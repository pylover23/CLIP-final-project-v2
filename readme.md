# 基于 CLIP 的 ABO 电商商品图文检索实验

## 项目简介

本项目基于 OpenAI CLIP (ViT-B/32) 构建电商商品 text-to-image 检索系统，使用 Amazon Berkeley Objects (ABO) 数据集子集（14 类、5600 样本），对比原始 CLIP、Linear Probe 和 LoRA 三种模型在不同 prompt 细粒度下的检索性能。

## 项目结构

```
.
├── main.py                         # 主入口：训练 LoRA + 三模型评估对比
├── model.py                        # 模型定义（OriginalCLIP / LinearProbe / LoRA）、训练与评估逻辑
├── dataset.py                      # 数据集类、分层 train/test 划分
├── DBManager.py                    # ChromaDB 向量数据库管理器
├── grid_search.py                  # Linear Probe 超参数网格搜索
├── linear_probe_ablation.py        # Linear Probe 投影位置消融实验
├── error_analysis.py               # 错误分析：逐样本检索结果收集与错误模式分析
├── demo.py                         # FastAPI Web 演示（文本检索商品图片）
├── utils.py                        # CSV/JSON 写入工具函数
├── readme.md                       # 本文件
│
├── data_abo_subset_selected14_english/
│   ├── abo_subset.csv              # 原始数据集（5600 条）
│   ├── train.csv                   # 训练集
│   ├── test.csv                    # 测试集
│   └── images/                     # 商品图片目录（5600 张 JPG）
│
├── checkpoints/                    # 模型权重
│   ├── best_linear_probe.pth       # 最佳 Linear Probe（both 模式）
│   ├── linear_probe_both.pth       # Linear Probe 双侧投影
│   ├── linear_probe_text_only.pth  # Linear Probe 仅文本侧
│   ├── linear_probe_vision_only.pth# Linear Probe 仅图像侧
│   └── lora_model.pth              # LoRA 微调模型
│
├── results/                        # 评估结果
│   ├── baseline_comparison_results_{5,10,20}.csv     # full prompt 下三模型 Top-K 对比
│   ├── prompt_granularity_results_{5,10,20}.csv      # 五种 prompt × 三模型结果
│   ├── linear_probe_ablation_results.csv             # 消融实验结果
│   ├── linear_probe_ablation_summary.json
│   ├── error_analysis_per_sample.csv                 # 错误分析：逐样本结果
│   └── error_analysis_summary.json                   # 错误分析：汇总统计
│
├── grid_search_results/            # 网格搜索结果
│   ├── best_linear_probe_params.json
│   └── linear_probe_grid_search.csv
│
├── chroma_db_deep_eval/            # 评估阶段 ChromaDB 持久化
├── chroma_db_deep_grid/            # 网格搜索阶段 ChromaDB 持久化
├── chroma_db_linear_probe_ablation/# 消融实验 ChromaDB 持久化
```

## 运行步骤
### Step 0：环境配置
```bash
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Step 1：超参数网格搜索

```bash
uv run grid_search.py
```

对训练集和测试集进行划分，然后对 Linear Probe 的学习率和温度系数进行网格搜索。每个组合训练一个模型后，使用测试集评估并选出最优参数，保存至 `checkpoints/best_linear_probe.pth`，同时将测试集图片数据写入向量数据库（持久化至 `chroma_db_deep_grid`）。

!!! 注意：初次运行需要下载CLIP模型权重，启动时会比较慢

**产出文件：**
- `grid_search_results/best_linear_probe_params.json`：最优超参数
- `grid_search_results/linear_probe_grid_search.csv`：全部搜索结果

### Step 2：模型训练与评估对比

```bash
uv run main.py
```

使用 grid search 得到的参数训练 LoRA 模型（权重保存至 `checkpoints/lora_model.pth`），然后分别在 Top-K = 5, 10, 20 下对原始 CLIP、Linear Probe、LoRA 模型进行评估对比。评估时将测试集图片写入向量数据库（持久化至 `chroma_db_deep_eval`），结果保存至 `results/` 目录，包含两个维度：

- 不同 baseline 在不同 Top-K 下 full prompt 的评估指标
- 不同 baseline 在不同 Top-K 和不同细粒度 prompt 下的评估指标

### Step 3：Linear Probe 消融实验

```bash
uv run linear_probe_ablation.py
```

对 Linear Probe 的投影位置进行消融实验，比较 `text+vision`、`text only`、`vision only` 三种方案，结果保存至 `results/linear_probe_ablation.csv`，向量数据库持久化至 `chroma_db_linear_probe_ablation`。

### Step 4：错误分析

```bash
uv run error_analysis.py
```

使用 LoRA 模型（最佳模型）在测试集上以 `full` prompt 进行 Top-20 text-to-image 检索，逐样本记录检索结果，分析错误模式并输出统计报告。运行前请确保 `checkpoints/lora_model.pth` 和 `chroma_db_deep_eval/` 中的向量库已存在（即已运行过 Step 2）。

**产出文件：**
- `results/error_analysis_per_sample.csv`：每个测试样本的检索结果（product_id、类别、品牌、标题、是否命中、排名、Top-20 检索 ID）
- `results/error_analysis_summary.json`：错误分析汇总，包含按类别错误率、同类/跨类混淆统计、命中排名分布、典型错误案例

### Step 5：Web 演示

```bash
uv run demo.py
```

启动 FastAPI 服务，浏览器访问 `http://127.0.0.1:8000`，支持文本检索商品图片，提供单路检索和 text+image 双路融合检索两种模式。
