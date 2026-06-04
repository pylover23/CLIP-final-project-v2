"""
错误分析脚本：使用 LoRA 模型在测试集上进行 text-to-image 检索，
逐样本记录检索结果，分析错误模式，输出分析报告。
"""

import os
import torch
import clip
import pandas as pd
import tqdm
from collections import defaultdict
from model import ECommerceCLIPLoRA, build_image_collection, PROMPT_KEYS
from dataset import ECommerceDataset, product_type_to_text
from torch.utils.data import DataLoader
from utils import write_csv, write_json


def collect_per_sample_results(model, dataloader, top_k=10):
    """
    逐样本收集检索结果。
    返回 list[dict]，每条包含:
      product_id, ProductType, Brand, Material, ProductTitle, Image,
      prompt_text, rank (0-based, -1=未命中), hit (0/1), retrieved_ids
    """
    # 构建/复用图片向量库
    db_manager = build_image_collection(
        model, dataloader,
        collection_name="loRA",   # 此处collection name与evaluate_prompts_text2image中一致
        persist_dir="./chroma_db_deep_eval",
    )

    # 读取测试集原始元数据
    test_df = pd.read_csv(
        "data_abo_subset_selected14_english/test.csv",
        usecols=["ProductId", "Brand", "ProductType", "Material", "ProductTitle", "Image"],
    )
    # 按 ProductId 建索引，方便后面查找
    meta_map = {}
    for _, row in test_df.iterrows():
        pid = int(row["ProductId"])
        if pid not in meta_map:
            meta_map[pid] = {
                "ProductType": str(row["ProductType"]),
                "Brand": str(row["Brand"]),
                "Material": str(row["Material"]),
                "ProductTitle": str(row["ProductTitle"]),
                "Image": str(row["Image"]),
            }

    records = []
    model.eval()
    tqdm_loader = tqdm.tqdm(dataloader, desc="Collecting per-sample results")

    for _, product_ids, prompts in tqdm_loader:
        # 只用 full prompt
        text_inputs = clip.tokenize(prompts["full"]).to(model.device)
        with torch.no_grad():
            _, text_embedding = model.forward(None, text_inputs)

        retrieved_ids = db_manager.search_images(text_embedding, top_k=top_k)

        for i, pid_tensor in enumerate(product_ids):
            pid = int(pid_tensor)
            gt_id = f"{pid}_img"
            retrieved = retrieved_ids[i]
            hit = 0
            rank = -1
            for r, rid in enumerate(retrieved):
                if rid == gt_id:
                    hit = 1
                    rank = r
                    break

            meta = meta_map.get(pid, {})
            records.append({
                "product_id": pid,
                "ProductType": meta.get("ProductType", ""),
                "Brand": meta.get("Brand", ""),
                "Material": meta.get("Material", ""),
                "ProductTitle": meta.get("ProductTitle", ""),
                "Image": meta.get("Image", ""),
                "prompt_text": prompts["full"][i],
                "hit": hit,
                "rank": rank,
                "retrieved_ids": "|".join(retrieved),
            })

    return records


def analyze_errors(records):
    """
    分析错误模式，返回分析结果 dict。
    """
    total = len(records)
    errors = [r for r in records if r["hit"] == 0]  # 未命中样本
    hits = [r for r in records if r["hit"] == 1]   # 命中样本
    error_count = len(errors)
    hit_count = len(hits)

    # 1. 按类别统计错误率
    category_stats = defaultdict(lambda: {"total": 0, "errors": 0})  # 字典：每一个类别的总数和错误数
    for r in records:
        pt = product_type_to_text(r["ProductType"])
        category_stats[pt]["total"] += 1
        if r["hit"] == 0:
            category_stats[pt]["errors"] += 1

    category_error_rates = {}
    # 按照错误率从高到低排序类别，并计算错误率
    for pt, stats in sorted(category_stats.items(), key=lambda x: x[1]["errors"] / max(x[1]["total"], 1), reverse=True):
        category_error_rates[pt] = {
            "total": stats["total"],
            "errors": stats["errors"],
            "error_rate": round(stats["errors"] / max(stats["total"], 1), 4),
        }

    # 2. 分析错误案例中 top-1 检索结果属于哪个类别
    #    需要从 retrieved_ids 反查 ProductType
    pid_to_type = {}
    for r in records:
        pid_to_type[r["product_id"]] = product_type_to_text(r["ProductType"])

    # 建立所有 product_id -> ProductType 的映射（从 test_df 获取不到的，从 records 获取）
    # retrieved_ids 中的 id 格式为 "{product_id}_img"
    wrong_top1_types = defaultdict(int)
    same_category_confusion = 0
    cross_category_confusion = 0
    for r in errors:
        retrieved = r["retrieved_ids"].split("|")
        if len(retrieved) > 0:
            top1_id = retrieved[0]
            top1_pid = int(top1_id.replace("_img", ""))
            top1_type = pid_to_type.get(top1_pid, "unknown")
            query_type = product_type_to_text(r["ProductType"])
            wrong_top1_types[top1_type] += 1
            if top1_type == query_type:
                same_category_confusion += 1
            else:
                cross_category_confusion += 1

    # 3. 按 rank 统计命中样本的排名分布
    rank_distribution = defaultdict(int)
    for r in hits:
        rank_distribution[r["rank"]] += 1

    # 4. 找出错误率最高的类别，选取典型错误案例
    worst_categories = sorted(
        category_error_rates.items(),
        key=lambda x: x[1]["error_rate"],
        reverse=True,
    )[:3]

    typical_errors = []
    for cat_name, _ in worst_categories:
        cat_errors = [r for r in errors if product_type_to_text(r["ProductType"]) == cat_name]
        # 选取 2 个典型案例
        for r in cat_errors[:2]:
            retrieved = r["retrieved_ids"].split("|")[:5]
            top1_pid = int(retrieved[0].replace("_img", "")) if retrieved else -1
            top1_type = pid_to_type.get(top1_pid, "unknown")
            typical_errors.append({
                "product_id": r["product_id"],
                "query_type": cat_name,
                "Brand": r["Brand"],
                "Material": r["Material"],
                "ProductTitle": r["ProductTitle"],
                "Image": r["Image"],
                "top1_retrieved_pid": top1_pid,
                "top1_retrieved_type": top1_type,
                "top5_retrieved": retrieved,
            })

    summary = {
        "total_samples": total,
        "hit_count": hit_count,
        "error_count": error_count,
        "overall_error_rate": round(error_count / max(total, 1), 4),
        "category_error_rates": category_error_rates,
        "same_category_confusion": same_category_confusion,
        "cross_category_confusion": cross_category_confusion,
        "wrong_top1_type_distribution": dict(wrong_top1_types),
        "hit_rank_distribution": {str(k): v for k, v in sorted(rank_distribution.items())},
        "worst_categories": [c for c, _ in worst_categories],
        "typical_errors": typical_errors,
    }

    return summary


def main():
    # 加载数据
    testset = ECommerceDataset(csv_path="data_abo_subset_selected14_english/test.csv")
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

    # 加载 LoRA 模型
    print("Loading LoRA model...")
    model = ECommerceCLIPLoRA(rank=8, alpha=16, dropout=0.05)
    model.load("checkpoints/lora_model.pth")
    model.eval()

    # 逐样本收集结果 (top_k=20 以获取更丰富的排名信息)
    print("Collecting per-sample results (top_k=20)...")
    records = collect_per_sample_results(model, test_loader, top_k=20)

    # 保存逐样本结果
    write_csv("results/error_analysis_per_sample.csv", records)
    print(f"Saved {len(records)} per-sample records to results/error_analysis_per_sample.csv")

    # 分析错误模式
    print("Analyzing error patterns...")
    summary = analyze_errors(records)
    write_json("results/error_analysis_summary.json", summary)
    print("Saved error analysis summary to results/error_analysis_summary.json")

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"错误分析摘要")
    print(f"{'='*60}")
    print(f"总样本数: {summary['total_samples']}")
    print(f"命中数: {summary['hit_count']}")
    print(f"未命中数: {summary['error_count']}")
    print(f"整体错误率: {summary['overall_error_rate']:.4f}")
    print(f"\n各类别错误率:")
    for cat, stats in summary["category_error_rates"].items():
        print(f"  {cat}: {stats['errors']}/{stats['total']} = {stats['error_rate']:.4f}")
    print(f"\n错误案例 top-1 检索结果中:")
    print(f"  同类别混淆: {summary['same_category_confusion']}")
    print(f"  跨类别混淆: {summary['cross_category_confusion']}")
    print(f"\n命中排名分布:")
    for rank, count in summary["hit_rank_distribution"].items():
        print(f"  rank {rank}: {count}")


if __name__ == "__main__":
    main()
