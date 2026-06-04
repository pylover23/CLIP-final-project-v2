# clip_course_presentation - Design Spec

> Human-readable design narrative - rationale, audience, style, color choices, content outline. Read once by downstream roles for context.
>
> Machine-readable execution contract: `spec_lock.md` (color / typography / icon / image short form). Executor re-reads `spec_lock.md` before every SVG page to resist context-compression drift. Keep both in sync; on divergence, `spec_lock.md` wins.

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | clip_course_presentation |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 15 pages |
| **Design Style** | B) General Consulting + dark tech classroom showcase |
| **Target Audience** | 课程老师与同学，具备基础机器学习认知，但不预设其熟悉 CLIP、LoRA 与向量数据库细节 |
| **Use Case** | 15 分钟课堂课程展示，介绍选题背景、模型设计、关键实现与实验结果 |
| **Created Date** | 2026-06-02 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 64px, top 52px, bottom 44px |
| **Content Area** | 1152×624 primary content field, allowing full-bleed decorative background beyond safe margin |

---

## III. Visual Theme

### Theme Style

- **Style**: B) General Consulting + dark tech classroom showcase
- **Theme**: Dark theme
- **Tone**: rational, modern, research-driven, presentation-first

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#07111F` | Deck main background, deep navy tech base |
| **Secondary bg** | `#0E1B2D` | Card background, content block background |
| **Primary** | `#1E66F5` | Titles, key dividers, structural emphasis |
| **Accent** | `#26C6DA` | Data highlights, model/result emphasis |
| **Secondary accent** | `#66E3FF` | Gradient transition, subtle glow details |
| **Body text** | `#EAF2FF` | Main readable text |
| **Secondary text** | `#A8B7CF` | Captions, annotations, supporting labels |
| **Tertiary text** | `#71839E` | Footnotes, low-priority metadata |
| **Border/divider** | `#20344F` | Card borders, grid lines, separators |
| **Success** | `#3DDC97` | Performance gains, positive deltas |
| **Warning** | `#FF8A4C` | Baseline contrast, limitations, error cues |

### Gradient Scheme (if needed, using SVG syntax)

```xml
<linearGradient id="titleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#1E66F5"/>
  <stop offset="100%" stop-color="#26C6DA"/>
</linearGradient>

<radialGradient id="bgDecor" cx="82%" cy="16%" r="56%">
  <stop offset="0%" stop-color="#26C6DA" stop-opacity="0.18"/>
  <stop offset="100%" stop-color="#26C6DA" stop-opacity="0"/>
</radialGradient>
```

---

## IV. Typography System

### Font Plan

**Typography direction**: modern CJK sans, clean classroom-tech presentation, high contrast and high readability

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Body** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Emphasis** | `"Microsoft YaHei"` | `Arial Black` | `sans-serif` |
| **Code** | - | `Consolas, "Courier New"` | `monospace` |

**Per-role font stacks**:

- Title: `"Microsoft YaHei", Arial, sans-serif`
- Body: `"Microsoft YaHei", Arial, sans-serif`
- Emphasis: `"Microsoft YaHei", "Arial Black", Arial, sans-serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 20px

| Purpose | Ratio to body | Example @ body=20 | Weight |
| ------- | ------------- | ----------------- | ------ |
| Cover title (hero headline) | 2.8-4x | 56-80px | Bold |
| Chapter / section opener | 2-2.4x | 40-48px | Bold |
| Page title | 1.6-1.9x | 32-38px | Bold |
| Hero number | 1.9-2.2x | 38-44px | Bold |
| Subtitle | 1.2-1.4x | 24-28px | SemiBold |
| **Body content** | **1x** | **20px** | Regular |
| Annotation / caption | 0.7-0.8x | 14-16px | Regular |
| Page number / footnote | 0.55-0.65x | 11-13px | Regular |

### Formula Rendering Policy

- **Policy**: `text-only`
- **Reason**: this is a classroom project showcase rather than a derivation-heavy academic talk; formulas should remain minimal, editable, and subordinate to the story of task-method-result.
- **Application**: only preserve short symbolic expressions where needed, such as `Recall@K`, `MRR@K`, `NDCG@K`, `Top-K`, and simple references to 512-d embeddings. Do not render block formulas as PNG assets.

---

## V. Layout Principles

### Page Structure

- **Header area**: 74-92px depending on page rhythm; contains page title, section tag, and subtle tech divider
- **Content area**: main 1152×624 usable region; asymmetric layouts preferred over rigid symmetry
- **Footer area**: 24-32px for page number and tiny descriptor only

### Layout Pattern Library (combine or break as content demands)

| Pattern | Suitable Scenarios |
| ------- | ----------------- |
| **Single column centered** | Cover, conclusion, core takeaways |
| **Asymmetric split (3:7 / 4:6)** | Method explanation, architecture + commentary |
| **Top-bottom split** | Result statement + chart/table |
| **Three/four column cards** | Prompt types, model comparison, key contributions |
| **Matrix grid (2×2)** | Error-type overview, comparative findings |
| **Center-radiating** | Overall pipeline or system relationship |
| **Negative-space-driven** | Chapter transition, final message |

### Spacing Specification

**Universal**:

| Element | Recommended Range | Current Project |
| ------- | ---------------- | --------------- |
| Safe margin from canvas edge | 40-60px | 52-64px |
| Content block gap | 24-40px | 28px |
| Icon-text gap | 8-16px | 10px |

**Card-based layouts**:

| Element | Recommended Range | Current Project |
| ------- | ---------------- | --------------- |
| Card gap | 20-32px | 24px |
| Card padding | 20-32px | 24px |
| Card border radius | 8-16px | 14px |
| Single-row card height | 530-600px | 548px |
| Double-row card height | 265-295px each | 276px |
| Three-column card width | 360-380px each | 368px |

**Non-card containers**:

- `breathing` pages should rely on whitespace, glow lines, oversized numerals, or a single highlighted statement rather than many rounded cards.
- Result pages may mix one dominant chart region with one smaller takeaway panel.
- Avoid a repeated “same three cards on every page” pattern; slide rhythm should visibly change with the argument.

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `templates/icons/`
- **Usage method**: SVG placeholder `<use data-icon="chunk-filled/icon-name" .../>`
- **Chosen library**: `chunk-filled`

### Recommended Icon List (fill as needed)

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Research question / target | `chunk-filled/target` | Slide 03 |
| Baseline vs improvement | `chunk-filled/arrow-trend-up` | Slides 12-15 |
| Retrieval / search | `chunk-filled/magnifying-glass` | Slides 04, 10 |
| Model / layers | `chunk-filled/layers` | Slides 07-09 |
| Vector database | `chunk-filled/database` | Slide 10 |
| Users / audience / scenario | `chunk-filled/users` | Slide 02 |
| Idea / insight | `chunk-filled/lightbulb` | Slides 03, 15 |
| Fine-tuning energy | `chunk-filled/bolt` | Slides 08-09 |
| Funnel / prompt filtering | `chunk-filled/funnel` | Slide 06 |
| Data/chart marker | `chunk-filled/chart-bar` | Slides 11-14 |

---

## VII. Visualization Reference List (if needed)

This deck intentionally uses **native custom SVG visualizations** rather than adapting a fixed chart template from `templates/charts/`.

Reason:
- the main value lies in aligning narrative emphasis with the exact experiment findings rather than fitting prebuilt chart shells;
- several pages combine numbers, annotations, and comparison callouts in one composition;
- classroom presentation readability is prioritized over strict template reuse.

**Catalog template usage**: none locked for this deck.

**Runners-up considered but not adopted**:

- `grouped_bar_chart` | rejected because the result pages require mixed bars + large numeric annotations + direct takeaway callouts, not a pure grouped bar layout
- `kpi_cards` | rejected because metrics need to coexist with cross-model reasoning, not only isolated score cards
- `process_flow` | rejected because the system page benefits from a more compact CLIP + ChromaDB retrieval loop than a long left-to-right template

---

## VIII. Image Resource List (if needed)

This deck uses **no external raster image assets by default**.

- Cover, section dividers, system diagrams, and result visuals are all generated as native SVG geometry, text, lines, and simple icons.
- Existing paper figures are used only as semantic references, not pasted screenshots.
- No AI image generation, web image sourcing, or formula PNG rendering is planned in this run.

---

## IX. Content Outline

### Part 1: 问题与任务

#### Slide 01 - Cover

- **Layout**: Single column centered with full-bleed dark tech background
- **Title**: 基于 CLIP 的 ABO 商品图文检索实验
- **Subtitle**: 轻量微调、Prompt 细粒度与检索性能分析
- **Info**: 课程展示 / 胡文煜 / 2026 春季学期

#### Slide 02 - 为什么做这个选题

- **Layout**: Asymmetric split (4:6), left concept cards + right scenario statement
- **Title**: 电商检索场景中的真实问题
- **Content**:
  - 用户会使用自然语言描述商品，而不仅仅是关键词
  - 同类商品外观接近，传统检索难以完成细粒度区分
  - 多模态预训练模型为商品图文对齐提供了新路径

#### Slide 03 - 我想回答的三个问题

- **Layout**: Three-column cards with large numeric chapter markers
- **Title**: 本次实验围绕三个核心问题展开
- **Content**:
  - 原始 CLIP 在商品检索里到底够不够用
  - Linear Probe 与 LoRA 是否能显著提升效果
  - Prompt 细粒度是否持续影响 text-to-image 检索质量

#### Slide 04 - 任务定义与整体流程

- **Layout**: Center-radiating process page
- **Title**: 任务定义：给定文本，返回最匹配商品图像
- **Content**:
  - 输入是商品 prompt，输出是候选图像 Top-K
  - CLIP 负责编码图像与文本
  - ChromaDB 负责向量存储与近邻检索
  - 最终使用 Recall@K、MRR@K、NDCG@K 评价效果

### Part 2: 数据与方法

#### Slide 05 - 数据集与样本构建

- **Layout**: Top-bottom split with top KPI strip + lower explanatory blocks
- **Title**: ABO 子集构建：14 类、5600 样本
- **Content**:
  - 每类 400 张图像，训练集 4480，测试集 1120
  - 分层划分保证各类分布均衡
  - 品牌、材质、标题等字段为 prompt 设计提供基础

#### Slide 06 - Prompt 设计与实验假设

- **Layout**: Five-column compact card row + lower takeaway panel
- **Title**: 从粗粒度到细粒度的五类 Prompt
- **Content**:
  - `type`
  - `material`
  - `brand`
  - `brand_material`
  - `full`
  - 假设：文本信息越完整，越有利于细粒度商品匹配

#### Slide 07 - Baseline：Original CLIP

- **Layout**: Asymmetric split (3:7), left concise bullets + right architecture sketch
- **Title**: 基线模型：直接使用预训练 CLIP
- **Content**:
  - 图像编码器与文本编码器共享对比学习语义空间
  - 不更新任何参数，作为 zero-shot baseline
  - 目标是先测出预训练模型在商品域上的原始表现

#### Slide 08 - 轻量微调一：Linear Probe

- **Layout**: Left formula-light explanation + right projection diagram
- **Title**: Linear Probe：冻结主干，只训练投影层
- **Content**:
  - 保留 CLIP 原有视觉语言知识
  - 在图像侧和文本侧后接线性投影
  - 参数少、训练快、适合验证“轻量适配”是否有效

#### Slide 09 - 轻量微调二：LoRA

- **Layout**: Asymmetric split with large emphasis block
- **Title**: LoRA：在后部 Transformer 模块中注入低秩适配
- **Content**:
  - 不直接重训大模型全部参数
  - 在保持 512 维输出的同时增强领域适配能力
  - 预期比纯投影层方法更能改善商品域对齐

#### Slide 10 - 检索系统实现

- **Layout**: Pipeline page with four stages and one result loop
- **Title**: 关键实现：CLIP Embedding + ChromaDB Top-K 检索
- **Content**:
  - 测试集图像先离线编码并写入向量库
  - 运行时将不同 prompt 编码为文本向量
  - 使用向量近邻返回候选商品图像
  - 支持分析不同模型与不同 prompt 的检索差异

### Part 3: 实验与发现

#### Slide 11 - 实验设置与评价指标

- **Layout**: Two-column layout, left hyperparameters, right metric definitions
- **Title**: 实验设置：统一比较、统一指标
- **Content**:
  - `embedding_dim = 512`
  - `lr = 0.001`
  - `temperature = 0.07`
  - `epoch = 20`
  - 指标采用 Recall@K、MRR@K、NDCG@K

#### Slide 12 - 结果一：Prompt 细粒度影响非常显著

- **Layout**: Dominant comparison chart + right takeaway panel
- **Title**: 完整商品标题远优于属性拼接 Prompt
- **Content**:
  - 原始 CLIP 下，`type` 的 Recall@5 仅 0.0482
  - 同一模型下，`full` Prompt 提升到 0.5446
  - 说明商品标题包含更充分的区分信息，是性能关键变量

#### Slide 13 - 结果二：轻量微调显著提升检索性能

- **Layout**: Top KPI band + lower Top-K comparison visualization
- **Title**: 在 `full` Prompt 下，LoRA 取得最佳结果
- **Content**:
  - Original CLIP：Recall@5 = 0.5446
  - Linear Probe：Recall@5 = 0.8000
  - LoRA：Recall@5 = 0.8339
  - LoRA 相对原始 CLIP 提升 +0.2893
  - Top-10 与 Top-20 下也保持领先

#### Slide 14 - 结果三：消融与误差分析

- **Layout**: 2×2 matrix grid
- **Title**: 进一步发现：文本侧更关键，但双侧协同最稳
- **Content**:
  - 属性型 Prompt 中，`text only` 通常优于 `vision only`
  - `brand_material` 下，text only Recall@5 = 0.1402，高于 vision only 的 0.1143
  - 完整标题 `full` 下，`text+vision` 反而最佳，Top-20 Recall 达到 0.9384
  - 典型错误主要来自同类商品视觉相似、跨类别混淆与低质量标题

#### Slide 15 - 结论与展望

- **Layout**: Single column centered with three conclusion blocks
- **Title**: 结论：Prompt 与轻量微调都决定了商品检索上限
- **Content**:
  - `full` Prompt 是最有效的文本表达形式
  - 轻量微调能显著提升商品域 text-to-image 检索性能
  - LoRA 在本实验中取得最佳综合表现
  - 后续可继续扩展到更大数据集、多模态推荐与更真实的线上检索场景

---

## X. Speaker Notes Requirements

One speaker note file per page, saved to `notes/`:

- **Filename**: match SVG name (e.g. `01_cover.md`)
- **Content**: script key points, transition phrases, and one-sentence summary per slide
- **Timing target**:
  - Slides 01-04: 3.5-4 minutes
  - Slides 05-10: 5.5-6 minutes
  - Slides 11-15: 4.5-5 minutes
- **Narration rule**: each page should first answer “这一页想说明什么”，再给 supporting detail; avoid reading tables line-by-line

---

## XI. Technical Constraints Reminder

### SVG Generation Must Follow:

1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` elements
3. Text wrapping uses `<tspan>` (`<foreignObject>` FORBIDDEN)
4. Transparency uses `fill-opacity` / `stroke-opacity`; `rgba()` FORBIDDEN
5. FORBIDDEN: `mask`, `<style>`, `class`, `foreignObject`
6. FORBIDDEN: `textPath`, `animate*`, `script`
7. Text characters: write typography and symbols as raw Unicode; XML reserved chars must be escaped
8. Keep all charts, blocks, and architecture diagrams editable as native SVG primitives rather than screenshots

### PPT Compatibility Rules:

- `<g opacity="...">` FORBIDDEN; apply opacity per child
- Inline styles only; no external CSS
- Prefer native vector shapes, icons, dividers, and text
- Avoid thin line weights that disappear in classroom projection
- Result pages should make numeric gains legible from distance, especially `0.8339`, `0.8000`, `0.5446`, and `+0.2893`
