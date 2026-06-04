# 问题背景与研究目标

随着电商平台商品规模持续增长，用户越来越依赖自然语言描述进行商品搜索。传统检索方法往往依赖人工标签或关键词匹配，难以充分理解商品图片与文本描述之间的语义对应关系。CLIP 和 ALIGN 等视觉语言预训练方法通过大规模图文对比学习获得了较强的跨模态表征能力 ，但直接应用于细粒度商品检索时，仍会受到商品类别、品牌、材质、标题描述粒度等因素影响。

本实验希望回答以下问题：

1.  原始 CLIP 在 ABO 商品图文检索任务中的基础表现如何；

2.  轻量微调方法是否能提升商品 text-to-image 检索性能；

3.  不同细粒度 prompt 对检索结果的影响有多大；

4.  在 Top-5、Top-10、Top-20 等不同候选规模下，各模型性能是否保持一致趋势。

# 任务定义与整体思路

**任务定义：** 给定一个商品文本 prompt，在候选商品图片库中检索与该文本最匹配的图片。本文主要评测 text-to-image 方向，即文本查询到图片返回的检索任务。

**整体思路：** 首先使用 CLIP `ViT-B/32` 提取图片和文本特征；然后将测试集图片向量写入 ChromaDB 向量数据库；评测阶段将每个商品的不同 prompt 编码为文本向量，并在图片向量库中检索 Top-K 候选。训练阶段分别使用 Linear Probe 和 LoRA 两种轻量微调方法，使商品图片与 `full` prompt 在向量空间中更加接近。

# 数据集与预处理

## 数据来源与字段结构

数据文件 `abo_subset.csv` 包含如下字段：

<div class="center">

`ProductId, ItemId, Brand, Category, ProductType, Colour, Material, ProductTitle, Image, ImageURL, MainImageId`

</div>

数据集共 5600 条记录，`ProductId`、`ItemId`、`ProductTitle` 和 `Image` 均为唯一值。本实验选取 14 个商品类别，每个类别 400 条样本，类别分布均衡。

<div id="tab:dataset-stat">

| **项目**     | **数值** |
|:-------------|:--------:|
| 样本总数     |   5600   |
| 训练集样本数 |   4480   |
| 测试集样本数 |   1120   |
| 商品类别数   |    14    |
| 每类样本数   |   400    |
| 品牌数       |    72    |

ABO 子集基本统计

</div>

## 数据划分

为保持训练集和测试集类别分布一致，本文使用分层切分思想：先按商品类别分别收集 `ProductId`，在每个类别内部使用固定随机种子打乱，再按 $`8:2`$ 比例划分训练和测试样本，每个类别划分为 320 条训练样本和 80 条测试样本。

## Prompt 构造

本文构造五种不同细粒度的 prompt：

<div id="tab:prompt-design">

| **Prompt 类型** | **模板** | **信息粒度** |
|:---|:---|:---|
| `type` | `a photo of a {product_type}` | 商品类别 |
| `material` | `a photo of a {material} {product_type}` | 材质 + 类别 |
| `brand` | `a photo of a {brand} {product_type}` | 品牌 + 类别 |
| `brand_material` | `a photo of a {brand} {material} {product_type}` | 品牌 + 材质 + 类别 |
| `full` | `a photo of a {product_title}` | 完整商品标题 |

五种 prompt 细粒度设计

</div>

由于 `Colour` 字段中存在较多 `Unknown Color`、`Multi`、`Multicolor` 等不稳定取值，所以未使用颜色构造 prompt，以降低噪声。

# 模型设计与实现

## 原始 CLIP 模型

CLIP 由图像编码器和文本编码器两部分组成，其核心思想是通过大规模图文对比学习，将语义匹配的图像和文本映射到同一个向量空间中，并使二者的表示尽可能接近 。在文本分支中，CLIP 使用 Transformer 对输入 prompt 的 token 序列进行上下文建模，并取序列中的全局表示作为文本特征；在图像分支中，本文采用的 `ViT-B/32` 版本使用 Vision Transformer，将输入图像划分为固定大小的 patch，再将这些 patch 展平，从而建模图像的全局语义信息。

设图像样本 $`\{I_i\}_{i=1}^{N}`$ 和文本样本 $`\{T_i\}_{i=1}^{N}`$ 分别经过图像编码器 $`f_v(\cdot)`$ 与文本编码器 $`f_t(\cdot)`$ 后得到特征表示。为了使用余弦相似度进行跨模态匹配，编码后的向量需要进行 $`L_2`$ 归一化：
``` math
\begin{equation}
    \hat{\mathbf{v}}_i = \frac{f_v(I_i)}{\|f_v(I_i)\|_2}, \qquad
    \hat{\mathbf{t}}_i = \frac{f_t(T_i)}{\|f_t(T_i)\|_2}
\end{equation}
```

归一化后的图像特征和文本特征通过点积计算相似度矩阵：
``` math
\begin{equation}
    S_{ij}=\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j
\end{equation}
```
其中 $`S_{ij}`$ 表示第 $`i`$ 张图像与第 $`j`$ 条文本之间的语义相似度。原始 CLIP 在本实验中不更新任何参数，直接作为零样本检索 baseline，用于衡量预训练模型在 ABO 商品域上的基础表现。本文实现基于 OpenAI 发布的 CLIP 代码与模型权重 。

<figure id="fig:clip-original" data-latex-placement="!htbp">
![clip原始架构图](assets/report_figures/clip原始架构图.png)
<figcaption>原始 CLIP 模型结构</figcaption>
</figure>

## Linear Probe 微调

Linear Probe 是一种轻量级微调方法，该方法冻结 CLIP 原有的图像编码器和文本编码器，只在两端输出特征之后加入可训练的线性投影层。这样既能保留 CLIP 已有的通用视觉语言知识，又能通过少量参数适配 ABO 商品检索任务。本文的 Linear Probe 同时在图像侧和文本侧加入线性映射：
``` math
\begin{equation}
    \mathbf{z}_i^{(v)} = W_v \mathbf{v}_i + b_v, \qquad
    \mathbf{z}_i^{(t)} = W_t \mathbf{t}_i + b_t
\end{equation}
```

其中 $`W_v,b_v`$ 为图像侧投影层参数，$`W_t,b_t`$ 为文本侧投影层参数。投影后的图像和文本向量再次进行 $`L_2`$ 归一化，并使用双向 InfoNCE 损失进行训练：
``` math
\begin{equation}
    \mathcal{L} = \frac{1}{2}\left(
    -\frac{1}{N}\sum_i \log \frac{\exp(\tilde{S}_{ii}/\tau)}{\sum_j \exp(\tilde{S}_{ij}/\tau)}
    -\frac{1}{N}\sum_i \log \frac{\exp(\tilde{S}_{ii}/\tau)}{\sum_j \exp(\tilde{S}_{ji}/\tau)}
    \right)
\end{equation}
```

其中 $`\tau`$ 为温度系数。该损失同时约束 “图像到文本” 和 “文本到图像” 两个方向，使同一 batch 内匹配的图文对相似度更高、不匹配的图文对相似度更低。由于原始 CLIP 与 LoRA 模型的输出维度均为 512，为保证不同模型之间的评估结果具有可比性，本文将 Linear Probe 的投影维度也固定为 512。

<figure id="fig:clip-linear" data-latex-placement="!htbp">
![clip微调架构图](assets/report_figures/clip微调架构图.png)
<figcaption>Linear Probe 微调结构</figcaption>
</figure>

## LoRA 微调

同样冻结原始 CLIP 的大部分参数，但不再调整输出端的线性投影层，而是在模型内部的部分线性层和注意力模块中注入低秩可训练分支。对于原始权重矩阵 $`W`$，LoRA 不直接更新 $`W`$，而是使用两个低秩矩阵 $`A`$ 和 $`B`$ 学习权重增量：
``` math
\begin{equation}
    h = Wx + \frac{\alpha}{r}BAx
\end{equation}
```

其中 $`r`$ 为低秩矩阵的秩，$`\alpha`$ 为缩放系数。与 Linear Probe 只改变最终 embedding 空间不同，LoRA 可以对编码器内部的特征提取过程进行轻量调整，因此理论上具有更强的领域适配能力。本文在 CLIP 文本 Transformer 和视觉 Transformer 的后部 block 中注入 LoRA 模块，使模型在保持输出维度 512 不变的情况下适配 ABO 商品检索任务。

## 向量数据库检索

在数据规模较小时，可以直接将所有 text embedding 和 image embedding 相乘组成相似度矩阵，并在内存中完成 Top-K 排序。但在真实电商场景中，候选商品数量可能达到百万甚至亿级，直接保存和遍历完整相似度矩阵会带来很高的内存和计算开销。因此，在实际工业场景中，通常将商品向量写入向量数据库，并通过近似最近邻检索算法进行高效召回。常见工具包括 ChromaDB、Faiss 等 。

本文实验使用 ChromaDB 构建图片向量库和文本向量库，使用的是测试集的数据。离线阶段，模型先对测试集图片进行编码，并以 `ProductId_img` 作为向量 ID 将图片 embedding 写入数据库； 在线评估阶段，demo 中可以使用两种查询方式：第一种，直接将不同细粒度的文本 prompt 编码为查询向量，再基于余弦距离从图片向量库中检索 Top-K 候选图片。 第二种，启用 text+image 双路检索融合，将 prompt 编码为查询向量，图片向量数据和文本向量分别赋予 6:4 权重进行加权融合后再检索。理论上，当查询文本中存在品牌、颜色等信息时，文本侧的向量可以提供语义线索；当数据库中存在大量相似物品时，融合检索能够提高召回率。

<figure id="fig:vector-db" data-latex-placement="!htbp">
![clip向量数据库架构图](assets/report_figures/clip向量数据库架构图.png)
<figcaption>向量数据库检索流程</figcaption>
</figure>

# 实验设置

## 训练与超参数

Linear Probe 通过网格搜索选择超参数。考虑到原始 CLIP 与 LoRA 的输出均为 512 维，本文固定 `embedding_dim` 为 512，搜索学习率和温度系数。最终以 `full` prompt 的 `Recall@5` 和 `MRR@5` 作为主要选择依据。 网格搜索得到的参数将直接用于 LoRA 微调训练（并未针对 LoRA 单独搜索参数）。

<div id="tab:hyperparams">

| **参数** | **取值** |
|:---------|:--------:|
| 学习率   |  0.001   |
| 投影维度 |   512    |
| 温度系数 |   0.07   |
| 训练轮数 |    20    |

最佳 Linear Probe 超参数

</div>

## 评价指标

本文采用 `Recall@K`、`MRR@K` 和 `NDCG@K` 作为评价指标。

**Recall@K** 衡量正确图片是否出现在前 $`K`$ 个候选中：
``` math
\begin{equation}
    \mathrm{Recall}@K = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathrm{Hit}@K(q)
\end{equation}
```

**MRR@K** 衡量正确图片排名倒数的平均值：
``` math
\begin{equation}
    \mathrm{MRR}@K = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathrm{RR}@K(q)
\end{equation}
```

**NDCG@K** 在是否命中的基础上进一步考虑排名位置折损，能够衡量检索的质量：
``` math
\begin{equation}
    \mathrm{NDCG}@K = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}}
    \frac{\sum_{r=1}^{K}\mathrm{rel}_r(q)/\log_2(r+1)}{\mathrm{IDCG}@K(q)}
\end{equation}
```

# 实验结果

## Top-5 Prompt 细粒度结果

<div id="tab:top5-prompt">

| **模型**          | **Prompt**     | **Recall@5** | **MRR@5**  | **NDCG@5** |
|:------------------|:---------------|:------------:|:----------:|:----------:|
| Original CLIP     | type           |    0.0482    |   0.0220   |   0.0284   |
| Original CLIP     | material       |    0.0688    |   0.0333   |   0.0420   |
| Original CLIP     | brand          |    0.0527    |   0.0269   |   0.0332   |
| Original CLIP     | brand_material |    0.0786    |   0.0359   |   0.0463   |
| Original CLIP     | full           |    0.5446    |   0.3601   |   0.4059   |
| CLIP+Linear Probe | type           |    0.0366    |   0.0172   |   0.0219   |
| CLIP+Linear Probe | material       |    0.0741    |   0.0316   |   0.0420   |
| CLIP+Linear Probe | brand          |    0.1080    |   0.0530   |   0.0664   |
| CLIP+Linear Probe | brand_material |    0.1357    |   0.0683   |   0.0849   |
| CLIP+Linear Probe | full           |    0.8000    |   0.5970   |   0.6478   |
| CLIP+LoRA         | type           |    0.0446    |   0.0215   |   0.0272   |
| CLIP+LoRA         | material       |    0.0866    |   0.0431   |   0.0538   |
| CLIP+LoRA         | brand          |    0.0893    |   0.0447   |   0.0556   |
| CLIP+LoRA         | brand_material |    0.1214    |   0.0670   |   0.0805   |
| CLIP+LoRA         | full           |  **0.8339**  | **0.6457** | **0.6929** |

不同模型在五种 prompt 下的 Top-5 检索结果

</div>

## 完整标题 Prompt 下的 Top-K 对比

<div id="tab:full-topk">

| **Top-K** | **模型**          | **Recall@K** | **MRR@K**  | **NDCG@K** |
|:----------|:------------------|:------------:|:----------:|:----------:|
| 5         | Original CLIP     |    0.5446    |   0.3601   |   0.4059   |
| 5         | CLIP+Linear Probe |    0.8000    |   0.5970   |   0.6478   |
| 5         | CLIP+LoRA         |  **0.8339**  | **0.6457** | **0.6929** |
| 10        | Original CLIP     |    0.6705    |   0.3772   |   0.4470   |
| 10        | CLIP+Linear Probe |    0.8875    |   0.6091   |   0.6766   |
| 10        | CLIP+LoRA         |  **0.9009**  | **0.6549** | **0.7149** |
| 20        | Original CLIP     |    0.7705    |   0.3843   |   0.4724   |
| 20        | CLIP+Linear Probe |    0.9330    |   0.6123   |   0.6881   |
| 20        | CLIP+LoRA         |  **0.9357**  | **0.6575** | **0.7238** |

`full` prompt 下不同 Top-K 的模型对比

</div>

## 微调增益分析

以 `full` prompt 的 Top-5 结果为例，Linear Probe 相比原始 CLIP 的 `Recall@5` 从 $`0.5446`$ 提升到 $`0.8000`$，提升 $`0.2554`$；LoRA 进一步提升到 $`0.8339`$，相比原始 CLIP 提升 $`0.2893`$。MRR 和 NDCG 也呈现类似趋势，说明微调不仅提高了是否命中的概率，也改善了正确图片在候选列表中的排序位置。

<div id="tab:delta">

| **对比项** | **$`\Delta`$Recall@5** | **$`\Delta`$MRR@5** | **$`\Delta`$NDCG@5** |
|:---|:--:|:--:|:--:|
| Linear Probe $`-`$ Original CLIP | +0.2554 | +0.2369 | +0.2419 |
| LoRA $`-`$ Original CLIP | **+0.2893** | **+0.2856** | **+0.2870** |
| LoRA $`-`$ Linear Probe | +0.0339 | +0.0487 | +0.0451 |

`full` prompt 下 Top-5 微调增益

</div>

## Linear Probe 投影位置消融实验

为进一步分析 Linear Probe 中图像侧投影层和文本侧投影层各自的作用，本文补充设计了三组消融实验：`text+vision` 表示同时在图像编码器和文本编码器输出后加入线性层；`text only` 表示只在文本侧加入线性层，图像侧保持原始 CLIP embedding；`vision only` 表示只在图像侧加入线性层，文本侧保持原始 embedding。三组实验均使用 grid search 得到的最优参数，即学习率 $`0.001`$、温度系数 $`0.07`$、训练轮数 $`20`$，并统一使用 `full` prompt 进行训练。

<div id="tab:linear-ablation-top5">

| **投影方式** | **Prompt**     | **Recall@5** | **MRR@5**  | **NDCG@5** |
|:-------------|:---------------|:------------:|:----------:|:----------:|
| text+vision  | type           |    0.0348    |   0.0155   |   0.0203   |
| text+vision  | material       |    0.0670    |   0.0303   |   0.0393   |
| text+vision  | brand          |    0.1018    |   0.0494   |   0.0623   |
| text+vision  | brand_material |    0.1375    |   0.0702   |   0.0867   |
| text+vision  | full           |  **0.7875**  | **0.5902** | **0.6397** |
| text only    | type           |  **0.0429**  |   0.0190   |   0.0249   |
| text only    | material       |  **0.0777**  | **0.0359** | **0.0461** |
| text only    | brand          |  **0.1089**  | **0.0511** | **0.0653** |
| text only    | brand_material |  **0.1402**  | **0.0703** | **0.0874** |
| text only    | full           |    0.7366    |   0.5136   |   0.5693   |
| vision only  | type           |    0.0411    | **0.0198** | **0.0250** |
| vision only  | material       |    0.0741    |   0.0349   |   0.0445   |
| vision only  | brand          |    0.0973    |   0.0447   |   0.0576   |
| vision only  | brand_material |    0.1143    |   0.0586   |   0.0724   |
| vision only  | full           |    0.7438    |   0.5219   |   0.5770   |

Linear Probe 投影位置消融实验的 Top-5 结果

</div>

<div id="tab:linear-ablation-full-topk">

| **Top-K** | **投影方式** | **Recall@K** | **MRR@K**  | **NDCG@K** |
|:----------|:-------------|:------------:|:----------:|:----------:|
| 5         | text+vision  |  **0.7875**  | **0.5902** | **0.6397** |
| 5         | text only    |    0.7366    |   0.5136   |   0.5693   |
| 5         | vision only  |    0.7438    |   0.5219   |   0.5770   |
| 10        | text+vision  |  **0.8777**  | **0.6025** | **0.6691** |
| 10        | text only    |    0.8500    |   0.5287   |   0.6060   |
| 10        | vision only  |    0.8554    |   0.5371   |   0.6135   |
| 20        | text+vision  |  **0.9384**  | **0.6069** | **0.6847** |
| 20        | text only    |    0.9080    |   0.5330   |   0.6210   |
| 20        | vision only  |    0.9205    |   0.5419   |   0.6303   |

`full` prompt 下 Linear Probe 投影位置消融的 Top-K 对比

</div>

# 结果分析与讨论

## Prompt 细粒度影响显著

从表 <a href="#tab:top5-prompt" data-reference-type="ref" data-reference="tab:top5-prompt">4</a> 可以看出，`full` prompt 明显优于其它四类属性拼接 prompt。以原始 CLIP 为例，`type` 的 `Recall@5` 仅为 $`0.0482`$，而 `full` prompt 达到 $`0.5446`$。这说明在 ABO 商品检索任务中，完整标题包含了更丰富的商品名称、款式、规格和用途信息，能更有效地区分同一类别中的相似商品。

## 品牌和材质有帮助，但不足以替代标题

加入品牌和材质后，prompt 表现通常优于只使用类别。例如 Linear Probe 的 `brand_material` 在 Top-5 下达到 $`0.1357`$，高于 `type` 的 $`0.0366`$。但相比 `full` prompt 的 $`0.8000`$ 仍有巨大差距，说明品牌和材质只能提供部分判别信息，而商品标题仍是最有效的文本来源。

## LoRA 在完整标题条件下表现最佳

LoRA 在 `full` prompt 下取得所有模型中的最佳结果，Top-5、Top-10、Top-20 的 Recall 分别为 $`0.8339`$、$`0.9009`$、$`0.9357`$。这说明 LoRA 能在保持 CLIP 原有语义空间的基础上进一步适配 ABO 商品域，尤其有利于完整商品标题这种信息密度较高的查询形式。

## Linear Probe 投影位置的影响

从表 <a href="#tab:linear-ablation-top5" data-reference-type="ref" data-reference="tab:linear-ablation-top5">7</a> 可以看出，Linear Probe 的投影位置对不同类型 prompt 的影响并不完全相同。对于 `type`、`material`、`brand` 和 `brand_material` 等属性型 prompt，`text only` 通常优于 `vision only`，例如在 `brand_material` prompt 下，`text only` 的 `Recall@5` 为 $`0.1402`$，高于 `vision only` 的 $`0.1143`$。这与本文的预期基本一致：由于当前任务是 text-to-image 检索，查询端来自文本 prompt，因此只调整文本侧 embedding 可以更直接地改善不同细粒度 prompt 与图片向量之间的对齐关系。

但在 `full` prompt 下，最优结果来自 `text+vision` 双侧投影。表 <a href="#tab:linear-ablation-full-topk" data-reference-type="ref" data-reference="tab:linear-ablation-full-topk">8</a> 显示，`text+vision` 在 Top-5、Top-10 和 Top-20 下均取得最高的 Recall、MRR 和 NDCG。这说明当 prompt 包含完整商品标题时，文本信息已经足够丰富，仅调整文本侧虽然能够适配查询表达，但同时调整图像侧和文本侧可以进一步重塑两种模态的共同 embedding 空间，从而获得更好的整体排序质量。因此，消融实验表明：对于较粗粒度属性 prompt，文本侧投影更关键；而对于信息量更大的完整标题 prompt，双侧投影仍然是更稳定、更有效的选择。

## Top-K 增大时 Recall 稳定上升

从表 <a href="#tab:full-topk" data-reference-type="ref" data-reference="tab:full-topk">5</a> 可以看到，随着 $`K`$ 从 5 增加到 20，三类模型的 Recall 均稳定上升。例如 LoRA 的 Recall 从 $`0.8339`$ 增至 $`0.9357`$。MRR 和 NDCG 的提升幅度相对较小，这是因为它们更关注正确结果是否排在靠前位置；如果正确图片主要从第 6 到第 20 位被补充命中，Recall 会明显上升，而 MRR/NDCG 增益会较温和。

## 典型错误分析

为深入理解 LoRA 模型在 `full` prompt 下的检索失败原因，本文对测试集中全部 1120 个样本的 Top-20 检索结果进行了逐样本分析。共有 72 个样本在 Top-20 中未命中正确图片，整体错误率为 $`6.43\%`$。

### 按类别错误率分析

表 <a href="#tab:error-by-category" data-reference-type="ref" data-reference="tab:error-by-category">9</a> 展示了各类别的错误率分布。错误主要集中在 `home product`（$`13.70\%`$）、`fine jewelry ring`（$`11.76\%`$）和 `rug`（$`9.88\%`$）三个类别，而 `wall art` 和 `shoes` 的错误率为 $`0\%`$。

<div id="tab:error-by-category">

| **类别**                       | **测试样本数** | **错误数** | **错误率** |
|:-------------------------------|:--------------:|:----------:|:----------:|
| home product                   |       73       |     10     |   0.1370   |
| fine jewelry ring              |       85       |     10     |   0.1176   |
| rug                            |       81       |     8      |   0.0988   |
| table                          |       76       |     7      |   0.0921   |
| fine jewelry earrings          |       77       |     6      |   0.0779   |
| home furniture and decor       |       78       |     6      |   0.0769   |
| kitchen product                |       74       |     5      |   0.0676   |
| home bed and bath              |       78       |     5      |   0.0641   |
| chair                          |       82       |     5      |   0.0610   |
| fine jewelry necklace/bracelet |       82       |     5      |   0.0610   |
| cellular phone case            |       87       |     4      |   0.0460   |
| sofa                           |       74       |     1      |   0.0135   |
| wall art                       |       89       |     0      |   0.0000   |
| shoes                          |       84       |     0      |   0.0000   |

LoRA 模型各类别 Top-20 错误率（`full` prompt）

</div>

### 错误类型分析

对 72 个错误案例的 Top-1 检索结果进行分析，错误可归纳为以下四类：

**（1）同类别内视觉相似商品混淆（48/72，$`66.7\%`$）。** 这是最主要的错误类型。同一类别内的商品在视觉外观上高度相似，模型难以区分。例如：同类戒指仅在宝石形状、镶嵌方式上略有不同；同类地毯的图案、色调相近；同品牌手机壳的设计差异微小。这说明 CLIP 的全局语义表征能力虽强，但在细粒度视觉区分上仍有不足。

**（2）跨类别混淆（24/72，$`33.3\%`$）。** 跨类别错误主要源于以下三种情况：

- **非英文标题。** 原始数据集中存在大量西班牙语、法语、德语、意大利语和瑞典语的商品标题，在构建数据集时统一将其翻译为英文，但翻译过程可能引入语义损失。CLIP 的文本编码器主要在英文语料上训练，对非英文文本的编码质量较低，容易导致跨类别误匹配。例如，一个瑞典语标题的桌子被错误匹配到法语标题的沙发。

- **同品牌跨类别。** Amazon 自有品牌的产品线覆盖多个类别，同品牌不同类别的商品在标题中包含相同的品牌关键词，容易造成文本相似度过高而跨类别匹配。

- **语义模糊的通用描述。** 部分商品标题过于简短或通用，缺乏有效的类别区分信息。

**（3）数据质量问题。** 个别样本的 `ProductTitle` 字段存在噪声，例如 `product_id=4881` 的标题仅为 “24”，`product_id=5434` 的标题包含 “Unknown Color Unknown Material” 占位文本。这些低质量标题无法为模型提供有效的语义信息。

**（4）命中位置分析。** 在 1048 个命中的样本中，$`55.9\%`$ 的正确结果排在 Top-1 位置，$`17.7\%`$ 排在第 2 位。排名越靠后，说明模型对这些样本的区分置信度越低，存在潜在的检索质量风险。

### 典型错误案例

表 <a href="#tab:error-examples" data-reference-type="ref" data-reference="tab:error-examples">10</a> 列举了三类典型错误案例。

<div id="tab:error-examples">

| **错误类型** | **查询 PID** | **查询类别** | **Top-1 类别** | **原因** |
|:---|:--:|:--:|:--:|:--:|
| 同类视觉混淆 | 4503 | rug | rug | 同品牌同类别不同花纹地毯 |
| 同类视觉混淆 | 4494 | fine ring | fine ring | 同材质同类别不同款式戒指 |
| 非英文标题 | 4505 | table | furniture | 法语标题跨类别匹配 |
| 同品牌跨类 | 4701 | furniture | chair | Stone & Beam 品牌跨类别 |
| 数据质量 | 4881 | necklace | furniture | 标题仅为 “24”，无有效信息 |

典型错误案例

</div>

### 错误分析小结

错误分析表明，LoRA 模型在大多数类别上表现良好，但在以下场景中仍存在明显不足：(1) 同类别内视觉高度相似的商品细粒度区分；(2) 非英文商品标题的语义理解；(3) 低质量或缺失的标题字段。未来改进方向包括引入多语言 CLIP 模型、增加图像侧的细粒度特征学习，以及对数据进行更严格的清洗和标准化处理。

# 不足与改进方向

本实验仍存在以下不足：

1.  **视觉模型的选取。** 本实验 image encoder 使用的是 ViT-B/32 版本，虽然具有较好的性能和效率平衡，但在细粒度视觉区分上可能存在不足。未来可以尝试更强大的视觉编码器，例如 ViT-L/14 或 Swin Transformer，以提升同类别内相似商品的区分能力。

2.  **部分字段存在噪声。** 颜色字段中存在 `Unknown Color`、`Multi` 等不稳定值，因此本实验未使用颜色构造 prompt。未来可通过规则清洗或人工映射获得更可靠的颜色描述。

3.  **完整标题带来强语义优势。** `full` prompt 的性能远高于属性 prompt，实验结果很大程度依赖商品标题质量。后续可以进一步比较标题截断、属性重写和自然语言改写对检索性能的影响。

4.  **当前仅评测 text-to-image。** 后续可以补充 image-to-text 或 image-to-image 方向，使检索系统评估更加完整。

# 结论

本文基于 ABO 商品子集构建了一个 CLIP 电商图文检索实验系统，比较了原始 CLIP、Linear Probe 和 LoRA 三种模型，并系统分析了五种 prompt 细粒度对 text-to-image 检索性能的影响。实验结果表明，完整商品标题 prompt 是影响检索性能的关键因素；在此基础上，轻量微调能够显著提升模型在商品域上的匹配能力。其中 LoRA 在 `full` prompt 下取得最佳结果，`Recall@5`、`MRR@5` 和 `NDCG@5` 分别达到 $`0.8339`$、$`0.6457`$ 和 $`0.6929`$。此外，Linear Probe 消融实验进一步表明，文本侧投影对属性型 prompt 更关键，而在完整标题 prompt 下，同时调整文本侧和图像侧投影能够获得更稳定的最优结果。

总体来看，CLIP 具备较强的电商商品跨模态检索能力，而面向具体商品域的轻量微调和高质量 prompt 构造是进一步提升性能的关键。

<div class="thebibliography">

9 Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision\[C\]. ICML, 2021.

Jia C, Yang Y, Xia Y, et al. Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision\[C\]. ICML, 2021.

ChromaDB Documentation. <https://docs.trychroma.com/>

OpenAI CLIP. <https://github.com/openai/CLIP>

</div>

# AI使用情况

1\. 由于原始数据集中存在部分非英文内容，如日语、法语等等，为了方便，直接将原始数据集丢给AI，让其将所有非英文标题翻译成英文。翻译过程中，AI 可能会引入一些语义损失。

2\. 代码部分，由于时间有限，无法深入学习理解LoRA微调的原理细节，在中有关LoRA的模型类，是让AI参考我自己写的Linear Probe模型类，结合LoRA的原理，写出来的
