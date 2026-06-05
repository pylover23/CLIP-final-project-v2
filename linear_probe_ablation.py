import json
import os
import time
from torch.utils.data import DataLoader

from dataset import ECommerceDataset, train_test_split
from model import ECommerceCLIPLinearProbe, evaluate_prompts_text2image, train_contrastive
from utils import write_csv, write_json

# 消融实验不同的投影方案
ABLATION_MODES = [
    ("both", "Linear Probe (text+vision)"),
    ("text_only", "Linear Probe (text only)"),
    ("vision_only", "Linear Probe (vision only)"),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def load_best_params(path=None):
    """
        使用grid search得到的最优参数进行消融实验，默认从grid_search_results/best_linear_probe_params.json中加载
    """
    path = path or project_path("grid_search_results", "best_linear_probe_params.json")
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info["params"]


def normalize_metric_rows(rows, projection_mode, top_k):
    normalized = []
    for row in rows:
        normalized.append({
            "projection_mode": projection_mode,
            "baseline": row["baseline"],
            "prompt": row["prompt"],
            "top_k": top_k,
            "recall": row[f"recall@{top_k}"],
            "mrr": row[f"mrr@{top_k}"],
            "ndcg": row[f"ndcg@{top_k}"],
        })
    return normalized


def run_linear_probe_ablation(train_loader, test_loader, top_k_values=(5, 10, 20)):
    params = load_best_params()
    run_tag = time.strftime("%Y%m%d_%H%M%S")  # 时间戳作为本次消融实验的唯一标识
    all_rows = []
    summary = {
        "params": params,
        "top_k_values": list(top_k_values),
        "results": [],
    }

    for projection_mode, baseline_name in ABLATION_MODES:
        print(f"\nTraining {baseline_name} with params: {params}")
        model = ECommerceCLIPLinearProbe(
            model_name="ViT-B/32",
            embedding_dim=params.get("embedding_dim", 512),
            projection_mode=projection_mode,
        )
        checkpoint_path = project_path("checkpoints", f"linear_probe_{projection_mode}.pth")
        train_contrastive(
            model,
            train_loader,
            epochs=params["epoch"],
            lr=params["lr"],
            patience=params["patience"],
            temperature=params["temperature"],
            prompt_key="full",
            save_path=checkpoint_path,
        )

        # 评估不同top_k下的性能，并收集结果
        for top_k in top_k_values:
            metrics, rows = evaluate_prompts_text2image(
                model,
                test_loader,
                baseline_name=baseline_name,
                collection_name=f"linear_probe_ablation_{projection_mode}",
                persist_dir=project_path("chroma_db_linear_probe_ablation", run_tag),
                top_k=top_k,
            )
            normalized_rows = normalize_metric_rows(rows, projection_mode, top_k)
            all_rows.extend(normalized_rows)
            summary["results"].append({
                "projection_mode": projection_mode,
                "baseline": baseline_name,
                "top_k": top_k,
                "metrics": metrics,
            })

    write_csv(project_path("results", "linear_probe_ablation_results.csv"), all_rows)
    write_json(project_path("results", "linear_probe_ablation_summary.json"), summary)
    return all_rows


if __name__ == "__main__":
    train_test_split(train_ratio=0.8)
    trainset = ECommerceDataset(csv_path=project_path("data_abo_subset_selected14_english", "train.csv"))
    testset = ECommerceDataset(csv_path=project_path("data_abo_subset_selected14_english", "test.csv"))
    batch_size = 128
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    run_linear_probe_ablation(train_loader, test_loader)
