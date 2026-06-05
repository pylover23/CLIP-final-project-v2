import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from DBManager import VectorDBManager
import tqdm
import json
from utils import *

PROMPT_KEYS = ["type", "material", "brand", "brand_material", "full"]

class ECommerceCLIPLinearProbe(nn.Module):
    """
        利用linear probe（线性探测）方式对CLIP模型进行轻量微调
        完全冻结CLIP的vision and text encoder，仅在vision和text encoder的输出上添加一个线性层
    """
    def __init__(self, model_name="ViT-B/32", embedding_dim=512, projection_mode="both"):
        """
            projection_mode: "both", "text_only", "vision_only" 是为了配合后续ablation实验设计的参数，能够直接兼容不同的投影方案
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(model_name, device=self.device)
        self.embedding_dim = embedding_dim
        self.projection_mode = projection_mode
        valid_modes = {"both", "text_only", "vision_only"}
        if projection_mode not in valid_modes:
            raise ValueError(f"projection_mode must be one of {valid_modes}")
        
        # 冻结CLIP模型的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 添加线性层，将CLIP的输出映射到指定的embedding_dim
        image_dim = self.clip_model.visual.output_dim
        text_dim = self.clip_model.transformer.width
        if projection_mode in {"both", "vision_only"}:
            self.image_projection = nn.Linear(image_dim, embedding_dim).to(self.device)
        else:
            if embedding_dim != image_dim:
                raise ValueError("text_only mode requires embedding_dim to match CLIP image output dimension.")
            # 当投影模式为text_only时，图像侧编码器不进行投影，直接使用原始输出，直接设置为恒等映射
            self.image_projection = nn.Identity().to(self.device)
        if projection_mode in {"both", "text_only"}:
            self.text_projection = nn.Linear(text_dim, embedding_dim).to(self.device)
        else:
            if embedding_dim != text_dim:
                raise ValueError("vision_only mode requires embedding_dim to match CLIP text output dimension.")
            # 当投影模式为vision_only时，文本侧编码器不进行投影，直接使用原始输出，直接设置为恒等映射
            self.text_projection = nn.Identity().to(self.device)

    def forward(self, images, texts):
        """
            输入图像和tokenized文本tensor，返回它们的embedding
        """
        image_embeddings = None
        text_embeddings = None
        
        with torch.no_grad():
            # 在评估时，images和texts会只传入其中一个参数，需要分别判断是否为空，不然会报错
            if images is not None:
                image_features = self.clip_model.encode_image(images).float()
            if texts is not None:
                text_features = self.clip_model.encode_text(texts).float()

        # projection层需要梯度，放在no_grad外
        if images is not None:
            image_embeddings = self.image_projection(image_features)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        if texts is not None:
            text_embeddings = self.text_projection(text_features)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
        return image_embeddings, text_embeddings
    
    def save(self, path="checkpoints/model.pth"):
        """
            只保存线性层的权重，CLIP模型权重在初始化函数中会自动load    
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'projection_mode': self.projection_mode,
            'embedding_dim': self.embedding_dim,
            'image_projection': self.image_projection.state_dict(),
            'text_projection': self.text_projection.state_dict(),
        }, path)
        print(f"模型权重已保存到 {path}")

    def load(self, path="checkpoints/model.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.image_projection.load_state_dict(checkpoint['image_projection'])
        self.text_projection.load_state_dict(checkpoint['text_projection'])
        print(f"模型权重已从 {path} 加载")

class OriginalCLIP(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(model_name, device=self.device)
        self.clip_model.eval()
    
    def forward(self, images, texts):
        image_embeddings = None
        text_embeddings = None
        if images is not None:
            image_embeddings = self.clip_model.encode_image(images).float()
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        if texts is not None:
            text_embeddings = self.clip_model.encode_text(texts).float()
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings, text_embeddings

class LoRALinear(nn.Module):
    def __init__(self, base, r, a, p):
        super().__init__()
        
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
            
        self.rank = r
        self.scaling = a / r
        self.dropout = nn.Dropout(p)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A.to(device=device, dtype=dtype)
        self.lora_B.to(device=device, dtype=dtype)
    
    def forward(self, x):
        lora_delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return self.base(x) + lora_delta.to(dtype=self.base.weight.dtype)

class LoRAMultiheadAttention(nn.Module):
    def __init__(self, base, r, a, p):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = r
        self.scaling = a / r
        self.dropout = nn.Dropout(p)
        self.lora_A = nn.Linear(base.embed_dim, r, bias=False)
        self.lora_B = nn.Linear(r, base.embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        device = base.out_proj.weight.device
        dtype = base.out_proj.weight.dtype
        self.lora_A.to(device=device, dtype=dtype)
        self.lora_B.to(device=device, dtype=dtype)

    def forward(self, query, key, value, **kwargs):
        output, weights = self.base(query, key, value, **kwargs)
        delta = self.lora_B(self.lora_A(self.dropout(query))) * self.scaling
        return output + delta.to(dtype=output.dtype), weights

class ECommerceCLIPLoRA(nn.Module):
    def __init__(self, model_name="ViT-B/32", rank=8, alpha=16, dropout=0.05, target_last_blocks=4):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(model_name, device=self.device)
        self.clip_model.float()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_last_blocks = target_last_blocks
        self._inject_lora_modules()

    def _inject_lora_modules(self):
        text_blocks = list(self.clip_model.transformer.resblocks)
        self._inject_into_blocks(text_blocks[-self.target_last_blocks:])

        visual_transformer = getattr(self.clip_model.visual, "transformer", None)
        if visual_transformer is not None and hasattr(visual_transformer, "resblocks"):
            vision_blocks = list(visual_transformer.resblocks)
            self._inject_into_blocks(vision_blocks[-self.target_last_blocks:])

    def _inject_into_blocks(self, blocks):
        for block in blocks:
            self._replace_linear_children(block)

    def _replace_linear_children(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                setattr(module, name, LoRAMultiheadAttention(child, self.rank, self.alpha, self.dropout))
            elif isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, self.rank, self.alpha, self.dropout))
            else:
                self._replace_linear_children(child)

    def trainable_parameters(self):
        return [p for p in self.clip_model.parameters() if p.requires_grad]

    def train(self):
        self.clip_model.train()

    def eval(self):
        self.clip_model.eval()

    def forward(self, images, texts):
        image_embeddings = None
        text_embeddings = None
        if images is not None:
            image_embeddings = self.clip_model.encode_image(images).float()
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        if texts is not None:
            text_embeddings = self.clip_model.encode_text(texts).float()
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings, text_embeddings

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "rank": self.rank,
                "alpha": self.alpha,
                "dropout": self.dropout,
                "target_last_blocks": self.target_last_blocks,
                "state_dict": {
                    k: v.detach().cpu()
                    for k, v in self.clip_model.state_dict().items()
                    if "lora_" in k
                },
            },
            path,
        )
        print(f"LoRA weights saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.clip_model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"LoRA weights loaded from {path}")


def get_trainable_params(model: nn.Module):
    """
        linear probe 和 loRA 微调时要训练的参数不同
        在微调时都会冻结CLIP的原始参数
        只需要保留需要训练的参数即可
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return params if params else None
    

def train_contrastive(model: nn.Module, dataloader: DataLoader, 
                      epochs, lr, patience, temperature,
                      prompt_key, save_path=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(get_trainable_params(model), lr)
    
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        # 只使用detailed prompt进行微调训练
        # dataloader中的每个batch包含: images, product_ids, 不同细粒度的prompt集合
        # 训练过程中仅需图片和detailed prompt，使两者在向量空间中对齐，不需要ids
        for images, _, prompts in loop:
            optimizer.zero_grad()
            images = images.to(model.device)
            # 由于需要将text embedding移动到GPU上，所以先对文本进行编码
            text_inputs = clip.tokenize(prompts[prompt_key]).to(model.device)
            # 获取图像和文本的embedding
            image_embeddings, text_embeddings = model.forward(images, text_inputs)
            # 使用对比学习损失InfoNCE loss，正样本为同一batch中对应的图像和文本，负样本为同一batch中其他图像和文本
            logits_per_image = image_embeddings @ text_embeddings.t() / temperature  # 图像到文本的相似度矩阵,大小为(batch_size, batch_size)
            logits_per_text = text_embeddings @ image_embeddings.t() / temperature   # 文本到图像的相似度矩阵,大小为(batch_size, batch_size)
            labels = torch.arange(len(images), device=model.device)
            loss = (criterion(logits_per_image, labels) + criterion(logits_per_text, labels)) / 2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # 有更优的结果，重置patience计数器
            if save_path and hasattr(model, "save"):
                model.save(save_path)
        else:
            patience_counter += 1
            print(f"Early-stop counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                break
    
    model.eval()
    return best_loss


def round_metric(value):
    return round(float(value), 4)


def compute_rank_metrics(retrieved_ids, product_ids, top_k):
    """
        计算评价指标
    """
    recall = 0.0
    mrr = 0.0
    ndcg = 0.0
    for retrieved, product_id in zip(retrieved_ids, product_ids):
        retrieved = retrieved[:top_k]
        gt = f"{product_id.item()}_img"
        if gt in retrieved:
            recall += 1.0
        # MRR@K: 正确结果在检索结果中的排名的倒数, 没找到就是0
        # NDCG@K: 计算DCG和IDCG
        # idcg = 1.0  最理想的情况，正确结果在第一位，rank=0，dcg=log(2)=1
        for rank, rid in enumerate(retrieved):
            if gt == rid:
                mrr += 1.0 / (rank + 1)
                ndcg += float(1.0 / torch.log2(torch.tensor(rank + 2.0)))
                break
    n = len(product_ids)
    return recall / n, mrr / n, ndcg / n


def build_image_collection(model: nn.Module, dataloader: DataLoader, collection_name, persist_dir):
    """
        只保存图片的embedding，text embedding意义不大
    """
    db_manager = VectorDBManager(collection_name=collection_name, persist_dir=persist_dir)
    # 如果之前跑过，数据已经存到了数据库中，则直接返回
    if db_manager.collection.count() > 0:
        return db_manager
    
    tqdm_loader = tqdm.tqdm(dataloader, desc="Building image collection")
    model.eval()
    for images, product_ids, _ in tqdm_loader:
        images = images.to(model.device)
        with torch.no_grad():
            image_emb, _ = model.forward(images, None)
        db_manager.add_image(
            [f"{int(product_id)}_img" for product_id in product_ids],
            image_emb
        )
    
    return db_manager
    

def evaluate_prompts_text2image(model: nn.Module, dataloader: DataLoader,
                                baseline_name, collection_name, persist_dir,
                                top_k = 5):
    """ 
        评估模型在图像-文本检索任务上的性能
        评价指标包括: recall@K（在top K检索结果中包含正确图像的比例）
                        mean reciprocal rank (MRR)（正确图像在检索结果中的平均排名的倒数）
                        ndcg@K（考虑排名位置的检索结果质量指标）
                        三个指标都是越接近1越好
    """
    # 评估时，需要输入文本，搜索图片，所以需要先将图片存入向量数据库中
    db_manager = build_image_collection(model, dataloader, collection_name, persist_dir)
    totals = {key: {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "n": 0} for key in PROMPT_KEYS}
    
    tqdm_loader = tqdm.tqdm(dataloader, desc=f"Evaluating {baseline_name}")
    model.eval()
    for _, product_ids, prompts in tqdm_loader:
        for prompt_key in PROMPT_KEYS:
            text_inputs = clip.tokenize(prompts[prompt_key]).to(model.device)
            with torch.no_grad():
                _, text_embedding = model.forward(None, text_inputs)
            
            retrieved_ids = db_manager.search_images(text_embedding, top_k=top_k)
            recall, mrr, ndcg = compute_rank_metrics(retrieved_ids, product_ids, top_k)
            batch_size = len(product_ids)
            totals[prompt_key]["recall"] += recall * batch_size
            totals[prompt_key]["mrr"] += mrr * batch_size
            totals[prompt_key]["ndcg"] += ndcg * batch_size
            totals[prompt_key]["n"] += batch_size

    # 统计指标
    flat = {}  # 存放指标结果
    rows = []  # 用于写入csv和json文件
    for prompt_key in PROMPT_KEYS:
        n = totals[prompt_key]["n"]
        recall = round_metric(totals[prompt_key]["recall"] / n)
        mrr = round_metric(totals[prompt_key]["mrr"] / n)
        ndcg = round_metric(totals[prompt_key]["ndcg"] / n)
        
        flat[f"{prompt_key}_recall@{top_k}"] = recall
        flat[f"{prompt_key}_mrr@{top_k}"] = mrr
        flat[f"{prompt_key}_ndcg@{top_k}"] = ndcg
        
        rows.append({
            "baseline": baseline_name,
            "prompt": prompt_key,
            f"recall@{top_k}": recall,
            f"mrr@{top_k}": mrr,
            f"ndcg@{top_k}": ndcg,
        })
        
    print_table(rows, top_k)
    return flat, rows
    
        
def print_table(rows, top_k):
    print(f"| Baseline | Prompt | Recall@{top_k} | MRR@{top_k} | NDCG@{top_k} |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['baseline']} | {row['prompt']} | "
            f"{row[f'recall@{top_k}']:.4f} | {row[f'mrr@{top_k}']:.4f} | {row[f'ndcg@{top_k}']:.4f} |"
        )


def load_best_linear_model():
    embedding_dim = 512  # 默认值，与CLIP输出保持一致
    model = ECommerceCLIPLinearProbe(embedding_dim=embedding_dim)
    model.load("checkpoints/best_linear_probe.pth")
    return model


def train_lora(train_loader: DataLoader, epochs=20,
               rank=8, alpha=16, dropout=0.05, save_path="checkpoints/lora_model.pth"):
    model = ECommerceCLIPLoRA(rank=rank, alpha=alpha, dropout=dropout)
    params = json.load(open("grid_search_results/best_linear_probe_params.json", "r", encoding="utf-8"))["params"]
    train_contrastive(
        model,
        train_loader,
        epochs=epochs,
        lr=params["lr"],
        patience=params["patience"],
        temperature=params["temperature"],
        prompt_key="full",
        save_path=save_path,
    )
    return model


def load_lora_model(rank=8, alpha=16, dropout=0.05):
    model = ECommerceCLIPLoRA(rank=rank, alpha=alpha, dropout=dropout)
    model.load("checkpoints/lora_model.pth")
    return model


def evaluate_all(test_loader, top_k=5, linear_model=None, lora_model=None):
    all_rows = []
    baseline_rows = []

    print("\nEvaluating Original CLIP...")
    original = OriginalCLIP()
    _, rows = evaluate_prompts_text2image(
        original,
        test_loader,
        baseline_name="Original CLIP",
        collection_name=f"originalCLIP",
        persist_dir="./chroma_db_deep_eval",
        top_k=top_k,
    )
    all_rows.extend(rows)
    print("\nFinish evaluating Original CLIP ")
    
    print("\nEvaluating CLIP + Linear Probe...")
    if linear_model is None:
        linear_model = load_best_linear_model()
    _, rows = evaluate_prompts_text2image(
        linear_model,
        test_loader,
        baseline_name="CLIP+Linear Probe",
        collection_name=f"linearProbe",
        persist_dir="./chroma_db_deep_eval",
        top_k=top_k,
    )
    all_rows.extend(rows)
    print("\nFinish evaluating CLIP + Linear Probe ")
    
    print("\nEvaluating CLIP + LoRA...")
    if lora_model is None:
        lora_model = load_lora_model()
    _, rows = evaluate_prompts_text2image(
        lora_model,
        test_loader,
        baseline_name="CLIP+LoRA",
        collection_name=f"loRA",
        persist_dir="./chroma_db_deep_eval",
        top_k=top_k,
    )
    all_rows.extend(rows)
    print("\nFinish evaluating CLIP + LoRA ")
    for row in all_rows:
        if row["prompt"] == "full":
            baseline_rows.append(row)

    write_csv(f"results/prompt_granularity_results_{top_k}.csv", all_rows)
    write_csv(f"results/baseline_comparison_results_{top_k}.csv", baseline_rows)
    return all_rows
