import itertools
from torch.utils.data import DataLoader
from dataset import ECommerceDataset, train_test_split
from model import train_contrastive, ECommerceCLIPLinearProbe, evaluate_prompts_text2image
from DBManager import VectorDBManager
from utils import * 

def choose_best_linear_result(results, top_k=5):
    """
        按照细粒度最高的prompt对应的recall@top_k 和 mrr@top_k 的结果返回最优的参数对
    """
    recall_key = f"full_recall@{top_k}"
    mrr_key = f"full_mrr@{top_k}"
    return max(results, key=lambda r: (r.get(recall_key, 0.0), r.get(mrr_key, 0.0)))


def run_grid_search(train_loader: DataLoader, test_loader: DataLoader, epoch ,top_k=5):
    """
        在跑grid_search时，用的是linear probe微调的模型，训练模型选取的最完整的prompt，
        进行text与image的对齐训练
    """
    param_grid = {
        "lr": [1e-2, 1e-3, 1e-4],
        "embedding_dim": [512],
        "temperature": [0.07, 0.1, 0.2],
        "epoch": [epoch],
        "patience": [3]
    }
    # 获取params的所有排列组合
    combinations = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())
    rows = []
    results = []
    
    for idx, value in enumerate(combinations, start=1):
        params = dict(zip(keys, value))
        print(f"\n[{idx}/{len(combinations)}] Linear Probe params: {params}")
        
        model = ECommerceCLIPLinearProbe(model_name="ViT-B/32", embedding_dim=params["embedding_dim"])
        
        train_contrastive(
            model,
            train_loader,
            epochs=params['epoch'],
            lr=params["lr"],
            patience=params["patience"],
            temperature=params["temperature"],
            prompt_key="full",
            save_path="checkpoints/best_linear_probe.pth"
        )
        
        # 评估时，需要先初始化向量数据库，对于不同的参数，模型的encoder得到的向量不同
        # 采用不同的collection_name进行分片区分，防止每次的数据都保存在同一个数据库里混淆
        run_tag = "".join([f"{k}_{v}" for k, v in params.items()])
        metrics, _ = evaluate_prompts_text2image(
            model,
            test_loader,
            baseline_name="CLIP+Linear Probe",
            collection_name=f"g{run_tag}_{idx}",
            persist_dir="./chroma_db_deep_grid",
            top_k=top_k
        )
        
        result = {**metrics, "params": params}
        results.append(result)
        
        rows.append({
            **params,
            **metrics
        })
    
    # 将所有参数的结果保存到csv文件
    write_csv("grid_search_results/linear_probe_grid_search.csv", rows)
    best = choose_best_linear_result(results, top_k)
    best_params = best["params"]
    # 将最优参数保存到json文件
    write_json("grid_search_results/best_linear_probe_params.json", best)
    
    print(f"\nBest Linear Probe params: {best_params}")
    
    
    # 用最优参数再训练一遍模型，并保存对应模型权重
    final_model = ECommerceCLIPLinearProbe(embedding_dim=best_params["embedding_dim"])
    train_contrastive(
        final_model,
        train_loader,
        epochs=best_params["epoch"],
        lr=best_params["lr"],
        patience=best_params["patience"],
        temperature=best_params["temperature"],
        prompt_key="full",
        save_path="checkpoints/best_linear_probe.pth",
    )
    return best_params
        
        
if __name__ == "__main__":
    # 数据准备
    train_test_split(train_ratio=0.8)
    trainset = ECommerceDataset(csv_path="data_abo_subset_selected14_english/train.csv")
    testset = ECommerceDataset(csv_path="data_abo_subset_selected14_english/test.csv")
    batch_size = 128
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # grid search
    run_grid_search(train_loader, test_loader, epoch=20 ,top_k=5)
