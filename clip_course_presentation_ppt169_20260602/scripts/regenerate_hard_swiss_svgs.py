from __future__ import annotations

from pathlib import Path
from textwrap import dedent


ROOT = Path(r"E:\CLIP期末作业\version2\clip_course_presentation_ppt169_20260602")
OUT = ROOT / "svg_output"

W = 1280
H = 720

NAVY = "#0B1220"
NAVY_2 = "#111C2E"
NAVY_3 = "#1A2B44"
INK = "#111111"
INK_2 = "#3E4A59"
MUTED = "#6C7988"
LINE = "#D8DEE6"
PAPER = "#FBFBF8"
PAPER_2 = "#F1F4F7"
ACCENT = "#1DA9C2"
ACCENT_DARK = "#127A94"
SUCCESS = "#1E9F67"
WARN = "#C86E1D"
SKY = "#E6F7FB"

FONT = "Arial, Microsoft YaHei, sans-serif"
MONO = "Consolas, Courier New, monospace"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def wrap_svg(body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">\n'
        f"{body}\n"
        "</svg>\n"
    )


def t(x: float, y: float, text: str, size: int, *, weight: str = "400", fill: str = INK,
      anchor: str = "start", family: str = FONT, extra: str = "") -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" {extra}>'
        f"{esc(text)}</text>"
    )


def rect(x: float, y: float, w: float, h: float, *, fill: str = "none", stroke: str = "none",
         sw: int = 0, rx: int = 0, extra: str = "") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}" {extra}/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = INK, sw: int = 2,
         dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'


def body_frame(page_no: int, title: str, section: str, content: str) -> str:
    return wrap_svg(
        "\n".join(
            [
                rect(0, 0, W, H, fill=PAPER),
                rect(56, 50, 1168, 620, fill=PAPER, stroke=LINE, sw=2),
                line(86, 104, 1192, 104, stroke=INK, sw=2),
                t(86, 84, section.upper(), 15, family=MONO, fill=ACCENT_DARK, extra='letter-spacing="2.4"'),
                t(86, 148, title, 36, weight="700"),
                t(1200, 84, f"{page_no:02d}", 16, family=MONO, fill=MUTED, anchor="end", extra='letter-spacing="2"'),
                t(1200, 652, f"{page_no}", 16, family=MONO, fill=MUTED, anchor="end"),
                content,
            ]
        )
    )


def cover() -> str:
    body = [
        rect(0, 0, W, H, fill=NAVY),
        rect(72, 72, 1136, 576, fill=NAVY_2, stroke=NAVY_3, sw=2),
    ]
    for x in range(760, 1160, 44):
        body.append(line(x, 120, x, 602, stroke="#20304B", sw=1))
    for y in range(120, 603, 44):
        body.append(line(760, y, 1160, y, stroke="#20304B", sw=1))
    body.extend(
        [
            rect(86, 112, 140, 8, fill=ACCENT),
            t(86, 160, "课程汇报 / CLIP RETRIEVAL", 18, family=MONO, fill="#9DB6D3", extra='letter-spacing="1.8"'),
            t(86, 274, "基于 CLIP 的", 60, weight="700", fill="#F6FAFF"),
            t(86, 350, "ABO 商品图文检索实验", 62, weight="700", fill="#F6FAFF"),
            t(86, 416, "轻量微调、Prompt 粒度与检索性能分析", 28, fill="#AFBED0"),
            rect(86, 468, 534, 82, fill="#0E1A2C", stroke="#29415F", sw=2),
            t(114, 520, "胡文煜  ·  程序设计语言课程展示  ·  2026 春季学期", 24, fill="#EAF2FF"),
            t(86, 620, "核心主张：完整标题 Prompt 与 LoRA 微调共同决定性能上限", 18, fill="#88A1BD"),
            rect(820, 156, 260, 260, fill="none", stroke=ACCENT, sw=4),
            rect(874, 210, 260, 260, fill="none", stroke="#8FDDE8", sw=2),
            rect(928, 264, 180, 180, fill="#11253F", stroke="none"),
            t(1018, 360, "Recall@5", 22, family=MONO, fill="#9DD7E3", anchor="middle"),
            t(1018, 420, "0.8339", 56, weight="700", fill="#F5FBFF", anchor="middle"),
            line(820, 528, 1148, 528, stroke="#2F4968", sw=2),
            t(820, 560, "Original CLIP", 16, fill="#7F95AC"),
            t(994, 560, "Linear Probe", 16, fill="#7F95AC", anchor="middle"),
            t(1148, 560, "LoRA", 16, fill=ACCENT, anchor="end"),
        ]
    )
    return wrap_svg("\n".join(body))


def slide_2() -> str:
    content = "\n".join(
        [
            t(90, 222, "电商检索的难点不是“找不到商品”，而是难以准确区分高度相似的商品。", 30, weight="700"),
            t(90, 264, "本实验关注 text-to-image 检索：给定商品描述，返回最匹配的商品图像。", 22, fill=INK_2),
            rect(90, 318, 334, 258, fill="#FFFFFF", stroke=INK, sw=2),
            rect(460, 318, 334, 258, fill="#FFFFFF", stroke=INK, sw=2),
            rect(830, 318, 334, 258, fill="#FFFFFF", stroke=INK, sw=2),
            t(116, 366, "01  语义表达更自然", 24, weight="700"),
            t(116, 406, "用户更习惯输入", 22, fill=INK_2),
            t(116, 438, "“黑色皮质高跟鞋”", 26, weight="700", fill=ACCENT_DARK),
            t(116, 472, "而不是只输入单个关键词。", 22, fill=INK_2),
            t(486, 366, "02  视觉相似度很高", 24, weight="700"),
            t(486, 406, "同类商品外观接近，", 22, fill=INK_2),
            t(486, 438, "品牌 / 材质 / 标题", 26, weight="700", fill=ACCENT_DARK),
            t(486, 472, "共同决定检索区分能力。", 22, fill=INK_2),
            t(856, 366, "03  预训练模型可迁移", 24, weight="700"),
            t(856, 406, "CLIP 提供跨模态对齐基础，", 22, fill=INK_2),
            t(856, 438, "但是否足够适应商品域", 26, weight="700", fill=ACCENT_DARK),
            t(856, 472, "需要实验验证。", 22, fill=INK_2),
            rect(90, 600, 1074, 34, fill=SKY, stroke="none"),
            t(106, 623, "目标：比较原始 CLIP、Linear Probe 与 LoRA，并分析 Prompt 细粒度对 Top-K 检索表现的影响。", 20, fill=ACCENT_DARK),
        ]
    )
    return body_frame(2, "为什么做这个选题", "Background", content)


def slide_3() -> str:
    content = "\n".join(
        [
            rect(90, 222, 320, 340, fill="#FFFFFF", stroke=INK, sw=2),
            rect(482, 222, 320, 340, fill="#FFFFFF", stroke=INK, sw=2),
            rect(874, 222, 320, 340, fill="#FFFFFF", stroke=INK, sw=2),
            t(116, 282, "Q1", 56, weight="700", fill=ACCENT_DARK),
            t(116, 336, "原始 CLIP 在 ABO 商品", 28, weight="700"),
            t(116, 374, "检索任务里到底够不够用？", 28, weight="700"),
            t(116, 434, "先测 zero-shot 基线，", 22, fill=INK_2),
            t(116, 466, "判断预训练模型的原始上限。", 22, fill=INK_2),
            t(508, 282, "Q2", 56, weight="700", fill=ACCENT_DARK),
            t(508, 336, "Linear Probe 与 LoRA", 28, weight="700"),
            t(508, 374, "能否显著提升性能？", 28, weight="700"),
            t(508, 434, "比较两类轻量微调方案，", 22, fill=INK_2),
            t(508, 466, "看参数效率与效果差异。", 22, fill=INK_2),
            t(900, 282, "Q3", 56, weight="700", fill=ACCENT_DARK),
            t(900, 336, "Prompt 粒度是否持续影响", 28, weight="700"),
            t(900, 374, "text-to-image 检索质量？", 28, weight="700"),
            t(900, 434, "比较 type 到 full 五类 Prompt，", 22, fill=INK_2),
            t(900, 466, "分析哪些文本信息最关键。", 22, fill=INK_2),
            line(90, 598, 1190, 598, stroke=LINE, sw=2),
            t(90, 636, "整份汇报围绕这三个问题展开：基线能力、微调增益、Prompt 作用。", 20, fill=MUTED),
        ]
    )
    return body_frame(3, "本次实验围绕三个核心问题展开", "Question Set", content)


def slide_4() -> str:
    boxes = [
        (108, 320, 220, 110, "文本查询", "商品描述 / Prompt"),
        (382, 320, 220, 110, "文本编码", "CLIP Text Encoder"),
        (656, 320, 220, 110, "向量检索", "ChromaDB Top-K"),
        (930, 320, 220, 110, "结果返回", "最匹配商品图像"),
    ]
    content = [
        t(90, 222, "任务定义：给定商品文本描述，在候选图像库中返回最匹配的商品。", 28, weight="700"),
        t(90, 260, "核心流程是“编码 - 存储 - 近邻检索 - 指标评价”。", 22, fill=INK_2),
    ]
    for x, y, w, h, title, sub in boxes:
        content.append(rect(x, y, 220, 110, fill="#FFFFFF", stroke=INK, sw=2))
        content.append(t(x + 20, y + 44, title, 24, weight="700"))
        content.append(t(x + 20, y + 78, sub, 18, fill=INK_2))
    content.extend(
        [
            line(328, 374, 382, 374, stroke=ACCENT, sw=4),
            line(602, 374, 656, 374, stroke=ACCENT, sw=4),
            line(876, 374, 930, 374, stroke=ACCENT, sw=4),
            rect(166, 500, 900, 90, fill=SKY, stroke=ACCENT_DARK, sw=2),
            t(190, 540, "评价指标：Recall@K 衡量是否命中，MRR@K 衡量正确结果是否排在前面，NDCG@K 兼顾命中与排序质量。", 21, fill=ACCENT_DARK),
            t(190, 570, "本实验统一比较 Top-5 / Top-10 / Top-20 三种候选规模。", 21, fill=ACCENT_DARK),
        ]
    )
    return body_frame(4, "任务定义与整体流程", "Task Pipeline", "\n".join(content))


def slide_5() -> str:
    content = [
        rect(90, 212, 240, 112, fill="#FFFFFF", stroke=INK, sw=2),
        rect(350, 212, 240, 112, fill="#FFFFFF", stroke=INK, sw=2),
        rect(610, 212, 240, 112, fill="#FFFFFF", stroke=INK, sw=2),
        rect(870, 212, 320, 112, fill="#111111", stroke=INK, sw=2),
        t(112, 248, "样本总数", 18, fill=MUTED),
        t(112, 298, "5600", 52, weight="700"),
        t(372, 248, "商品类别", 18, fill=MUTED),
        t(372, 298, "14", 52, weight="700"),
        t(632, 248, "训练 / 测试", 18, fill=MUTED),
        t(632, 298, "4480 / 1120", 40, weight="700"),
        t(892, 248, "品牌数", 18, fill="#A6F1FD"),
        t(892, 298, "72", 52, weight="700", fill="#FFFFFF"),
        rect(90, 364, 520, 232, fill="#FFFFFF", stroke=INK, sw=2),
        rect(650, 364, 540, 232, fill="#FFFFFF", stroke=INK, sw=2),
        t(116, 408, "数据构建", 24, weight="700"),
        t(116, 448, "• 从 ABO 中选择 14 个类别，每类 400 条样本", 21, fill=INK_2),
        t(116, 482, "• ProductId / ItemId / ProductTitle / Image 唯一", 21, fill=INK_2),
        t(116, 516, "• 保留品牌、材质、标题等字段供 Prompt 构造", 21, fill=INK_2),
        t(676, 408, "划分策略", 24, weight="700"),
        t(676, 448, "• 按类别分层划分，保持分布均衡", 21, fill=INK_2),
        t(676, 482, "• 每类 320 条训练样本 + 80 条测试样本", 21, fill=INK_2),
        t(676, 516, "• 固定随机种子，保证实验可复现", 21, fill=INK_2),
        rect(90, 620, 1100, 24, fill=SKY, stroke="none"),
        t(106, 638, "为什么重要：类内高相似 + 跨类弱差异，正适合检验 CLIP 的商品域迁移能力。", 19, fill=ACCENT_DARK),
    ]
    return body_frame(5, "ABO 子集构建：14 类、5600 样本", "Dataset", "\n".join(content))


def slide_6() -> str:
    labels = [
        ("type", "仅类别"),
        ("material", "材质 + 类别"),
        ("brand", "品牌 + 类别"),
        ("brand_material", "品牌 + 材质 + 类别"),
        ("full", "完整标题"),
    ]
    content = [t(90, 220, "Prompt 从粗粒度到细粒度逐步增加文本信息量。", 28, weight="700")]
    start_x = 90
    widths = [170, 190, 170, 250, 170]
    for idx, ((name, desc), w) in enumerate(zip(labels, widths)):
        x = start_x + sum(widths[:idx]) + idx * 18
        fill = "#111111" if name == "full" else "#FFFFFF"
        text_fill = "#FFFFFF" if name == "full" else INK
        muted_fill = "#B7FAFF" if name == "full" else MUTED
        content.append(rect(x, 278, w, 204, fill=fill, stroke=INK, sw=2))
        content.append(t(x + 18, 320, name, 22, family=MONO, fill=muted_fill))
        content.append(t(x + 18, 368, desc, 26, weight="700", fill=text_fill))
        if name == "full":
            content.append(t(x + 18, 418, "a photo of a {product_title}", 18, fill="#C9F7FF"))
        else:
            content.append(t(x + 18, 418, "属性模板化表达", 18, fill=muted_fill))
    content.extend(
        [
            line(176, 524, 1112, 524, stroke=ACCENT, sw=4),
            t(1112, 514, "信息密度上升", 16, family=MONO, fill=ACCENT_DARK, anchor="end"),
            rect(90, 568, 1100, 76, fill=SKY, stroke=ACCENT_DARK, sw=2),
            t(114, 612, "实验假设：文本信息越完整，越有利于区分外观相似、类别接近的商品。", 24, weight="700", fill=ACCENT_DARK),
        ]
    )
    return body_frame(6, "Prompt 设计：从粗粒度到细粒度", "Prompt Design", "\n".join(content))


def slide_7() -> str:
    content = [
        rect(90, 212, 390, 382, fill="#FFFFFF", stroke=INK, sw=2),
        t(116, 258, "方法定位", 24, weight="700"),
        t(116, 302, "• 直接使用预训练 CLIP ViT-B/32", 21, fill=INK_2),
        t(116, 336, "• 不更新任何参数，作为 zero-shot baseline", 21, fill=INK_2),
        t(116, 370, "• 先观察商品域上的原始跨模态对齐能力", 21, fill=INK_2),
        t(116, 432, "问题意识", 24, weight="700"),
        t(116, 476, "预训练语义很强，但商品图像的", 21, fill=INK_2),
        t(116, 508, "品牌、材质、款式差异更细，", 21, fill=INK_2),
        t(116, 540, "可能需要任务域适配。", 21, fill=INK_2),
        rect(536, 230, 250, 110, fill=PAPER_2, stroke=INK, sw=2),
        rect(536, 438, 250, 110, fill=PAPER_2, stroke=INK, sw=2),
        rect(898, 334, 230, 110, fill="#111111", stroke=INK, sw=2),
        t(660, 278, "图像编码器", 28, weight="700", anchor="middle"),
        t(660, 312, "Vision Transformer", 18, fill=MUTED, anchor="middle"),
        t(660, 486, "文本编码器", 28, weight="700", anchor="middle"),
        t(660, 520, "Transformer", 18, fill=MUTED, anchor="middle"),
        t(1012, 382, "共享语义空间", 28, weight="700", fill="#FFFFFF", anchor="middle"),
        t(1012, 416, "余弦相似度匹配", 18, fill="#A9F4FF", anchor="middle"),
        line(786, 286, 898, 356, stroke=ACCENT, sw=4),
        line(786, 492, 898, 422, stroke=ACCENT, sw=4),
    ]
    return body_frame(7, "基线模型：直接使用预训练 CLIP", "Original CLIP", "\n".join(content))


def slide_8() -> str:
    content = [
        rect(90, 212, 360, 390, fill="#FFFFFF", stroke=INK, sw=2),
        t(116, 258, "核心思想", 24, weight="700"),
        t(116, 302, "冻结原始 CLIP 编码器，", 22, fill=INK_2),
        t(116, 336, "只在图像侧和文本侧输出后", 22, fill=INK_2),
        t(116, 370, "接入可训练线性投影层。", 22, fill=INK_2),
        t(116, 432, "优点", 24, weight="700"),
        t(116, 472, "• 参数量小，训练快", 21, fill=INK_2),
        t(116, 506, "• 保留 CLIP 原有通用视觉语义", 21, fill=INK_2),
        t(116, 540, "• 适合验证“轻量适配”是否有效", 21, fill=INK_2),
        rect(520, 238, 220, 96, fill=PAPER_2, stroke=INK, sw=2),
        rect(520, 466, 220, 96, fill=PAPER_2, stroke=INK, sw=2),
        rect(834, 238, 240, 96, fill="#FFFFFF", stroke=ACCENT_DARK, sw=4),
        rect(834, 466, 240, 96, fill="#FFFFFF", stroke=ACCENT_DARK, sw=4),
        rect(1112, 350, 90, 96, fill="#111111", stroke=INK, sw=2),
        t(630, 294, "图像编码器", 24, weight="700", anchor="middle"),
        t(630, 520, "文本编码器", 24, weight="700", anchor="middle"),
        t(954, 294, "Linear Head", 24, weight="700", fill=ACCENT_DARK, anchor="middle"),
        t(954, 520, "Linear Head", 24, weight="700", fill=ACCENT_DARK, anchor="middle"),
        t(1157, 408, "512D", 24, weight="700", fill="#FFFFFF", anchor="middle"),
        line(740, 286, 834, 286, stroke=INK, sw=3),
        line(740, 514, 834, 514, stroke=INK, sw=3),
        line(1074, 286, 1112, 378, stroke=ACCENT, sw=4),
        line(1074, 514, 1112, 418, stroke=ACCENT, sw=4),
    ]
    return body_frame(8, "Linear Probe：冻结主干，只训练投影层", "Fine-tune I", "\n".join(content))


def slide_9() -> str:
    content = [
        rect(90, 212, 412, 390, fill="#111111", stroke=INK, sw=2),
        t(116, 258, "LoRA 的关键点", 24, weight="700", fill="#FFFFFF"),
        t(116, 302, "不直接更新全部参数，而是在后部", 22, fill="#D9E4F1"),
        t(116, 336, "Transformer Block 中插入低秩适配分支。", 22, fill="#D9E4F1"),
        t(116, 396, "优点", 24, weight="700", fill="#A6F1FD"),
        t(116, 438, "• 适配能力强于只改输出头", 21, fill="#D9E4F1"),
        t(116, 472, "• 仍保持参数高效", 21, fill="#D9E4F1"),
        t(116, 506, "• 更适合商品域细粒度对齐", 21, fill="#D9E4F1"),
        rect(578, 234, 530, 320, fill="#FFFFFF", stroke=INK, sw=2),
        rect(624, 280, 180, 70, fill=PAPER_2, stroke=INK, sw=2),
        rect(624, 374, 180, 70, fill=PAPER_2, stroke=INK, sw=2),
        rect(878, 280, 184, 70, fill="#E8F9FC", stroke=ACCENT_DARK, sw=3),
        rect(878, 374, 184, 70, fill="#E8F9FC", stroke=ACCENT_DARK, sw=3),
        t(714, 323, "Frozen Block", 24, weight="700", anchor="middle"),
        t(714, 417, "Frozen Block", 24, weight="700", anchor="middle"),
        t(970, 311, "LoRA A / B", 24, weight="700", fill=ACCENT_DARK, anchor="middle"),
        t(970, 343, "low-rank update", 18, fill=ACCENT_DARK, anchor="middle"),
        t(970, 405, "LoRA A / B", 24, weight="700", fill=ACCENT_DARK, anchor="middle"),
        t(970, 437, "task adaptation", 18, fill=ACCENT_DARK, anchor="middle"),
        line(804, 315, 878, 315, stroke=INK, sw=3),
        line(804, 409, 878, 409, stroke=INK, sw=3),
        t(578, 598, "直观理解：Linear Probe 改“出口”，LoRA 改“后部语义变换过程”。", 22, fill=ACCENT_DARK),
    ]
    return body_frame(9, "LoRA：在后部 Transformer 模块中注入低秩适配", "Fine-tune II", "\n".join(content))


def slide_10() -> str:
    xs = [90, 366, 642, 918]
    titles = ["离线编码", "向量入库", "在线查询", "Top-K 返回"]
    descs = ["测试图像编码为 embedding", "以 ProductId_img 写入 ChromaDB", "Prompt 编码为文本向量", "返回最匹配候选图像"]
    content = []
    for x, title, desc in zip(xs, titles, descs):
        content.extend(
            [
                rect(x, 262, 220, 170, fill="#FFFFFF", stroke=INK, sw=2),
                t(x + 22, 314, title, 28, weight="700"),
                t(x + 22, 356, desc, 20, fill=INK_2),
            ]
        )
    content.extend(
        [
            line(310, 347, 366, 347, stroke=ACCENT, sw=4),
            line(586, 347, 642, 347, stroke=ACCENT, sw=4),
            line(862, 347, 918, 347, stroke=ACCENT, sw=4),
            rect(166, 504, 950, 106, fill=SKY, stroke=ACCENT_DARK, sw=2),
            t(190, 548, "实现重点：统一使用 CLIP embedding 作为检索空间，保证不同模型和不同 Prompt 的结果可直接比较。", 22, weight="700", fill=ACCENT_DARK),
            t(190, 584, "这样做也更贴近实际工业场景中的向量检索流水线。", 22, fill=ACCENT_DARK),
        ]
    )
    return body_frame(10, "关键实现：CLIP Embedding + ChromaDB Top-K 检索", "System", "\n".join(content))


def slide_11() -> str:
    content = [
        rect(90, 212, 420, 388, fill="#FFFFFF", stroke=INK, sw=2),
        rect(560, 212, 630, 388, fill="#FFFFFF", stroke=INK, sw=2),
        t(116, 258, "训练设置", 24, weight="700"),
        rect(116, 294, 164, 90, fill=PAPER_2, stroke=INK, sw=2),
        rect(300, 294, 164, 90, fill=PAPER_2, stroke=INK, sw=2),
        rect(116, 404, 164, 90, fill=PAPER_2, stroke=INK, sw=2),
        rect(300, 404, 164, 90, fill=PAPER_2, stroke=INK, sw=2),
        t(136, 326, "lr", 18, family=MONO, fill=MUTED),
        t(136, 366, "0.001", 34, weight="700"),
        t(320, 326, "embedding", 18, family=MONO, fill=MUTED),
        t(320, 366, "512", 34, weight="700"),
        t(136, 436, "temperature", 18, family=MONO, fill=MUTED),
        t(136, 476, "0.07", 34, weight="700"),
        t(320, 436, "epoch", 18, family=MONO, fill=MUTED),
        t(320, 476, "20", 34, weight="700"),
        t(586, 258, "评价指标", 24, weight="700"),
        t(586, 314, "Recall@K", 24, weight="700", fill=ACCENT_DARK),
        t(586, 344, "是否在前 K 个候选中命中正确图像", 20, fill=INK_2),
        t(586, 404, "MRR@K", 24, weight="700", fill=ACCENT_DARK),
        t(586, 434, "正确结果排得越靠前，分数越高", 20, fill=INK_2),
        t(586, 494, "NDCG@K", 24, weight="700", fill=ACCENT_DARK),
        t(586, 524, "同时考虑命中与排序位置质量", 20, fill=INK_2),
        rect(90, 624, 1100, 20, fill=SKY),
        t(106, 639, "比较策略：所有模型统一在 Top-5 / Top-10 / Top-20 下评测，避免只看单一候选规模。", 18, fill=ACCENT_DARK),
    ]
    return body_frame(11, "实验设置：统一比较、统一指标", "Evaluation", "\n".join(content))


def slide_12() -> str:
    bars = [0.0482, 0.0688, 0.0527, 0.0786, 0.5446]
    labels = ["type", "material", "brand", "brand+mat.", "full"]
    max_h = 260
    base_y = 572
    scale = max_h / 0.60
    content = [
        rect(90, 210, 760, 410, fill="#FFFFFF", stroke=INK, sw=2),
        rect(882, 210, 308, 410, fill="#111111", stroke=INK, sw=2),
        t(118, 252, "Original CLIP 下不同 Prompt 的 Recall@5", 26, weight="700"),
        line(140, base_y, 786, base_y, stroke=INK, sw=2),
        line(140, 300, 140, base_y, stroke=INK, sw=2),
    ]
    x0 = 214
    gap = 94
    for i, (v, label) in enumerate(zip(bars, labels)):
        h = v * scale
        x = x0 + i * gap
        bar_fill = ACCENT if label == "full" else "#111111"
        content.extend(
            [
                rect(x, base_y - h, 46, h, fill=bar_fill),
                t(x + 23, base_y - h - 12, f"{v:.4f}", 16, weight="700", fill=bar_fill, anchor="middle"),
                t(x + 23, 602, label, 16, fill=MUTED, anchor="middle"),
            ]
        )
    content.extend(
        [
            t(908, 258, "关键发现", 24, weight="700", fill="#FFFFFF"),
            t(908, 318, "1. 属性型 Prompt", 20, fill="#EAF2FF"),
            t(908, 350, "提升有限，Recall@5", 20, fill="#EAF2FF"),
            t(908, 382, "普遍低于 0.08。", 20, fill="#EAF2FF"),
            t(908, 444, "2. full Prompt", 20, fill="#A6F1FD"),
            t(908, 476, "直接跃升到", 20, fill="#EAF2FF"),
            t(908, 526, "0.5446", 42, weight="700", fill="#FFFFFF"),
            t(908, 570, "说明完整标题是最关键的", 20, fill="#EAF2FF"),
            t(908, 598, "文本信息来源。", 20, fill="#EAF2FF"),
        ]
    )
    return body_frame(12, "结果一：完整标题 Prompt 显著优于属性拼接", "Result I", "\n".join(content))


def slide_13() -> str:
    ks = ["Top-5", "Top-10", "Top-20"]
    original = [0.5446, 0.6705, 0.7705]
    lp = [0.8000, 0.8875, 0.9330]
    lora = [0.8339, 0.9009, 0.9357]
    content = [
        rect(90, 202, 1100, 104, fill="#111111", stroke=INK, sw=2),
        t(118, 240, "最优结论", 18, family=MONO, fill="#A6F1FD"),
        t(118, 286, "在 full Prompt 下，LoRA 在 Recall / MRR / NDCG 三项指标上都取得最佳结果。", 28, weight="700", fill="#FFFFFF"),
        rect(90, 340, 1100, 270, fill="#FFFFFF", stroke=INK, sw=2),
        line(160, 560, 1130, 560, stroke=INK, sw=2),
    ]
    centers = [290, 620, 950]
    scale = 180 / 1.0
    for center, k, o, p, l in zip(centers, ks, original, lp, lora):
        x = center - 72
        content.extend(
            [
                t(center, 388, k, 22, weight="700", anchor="middle"),
                rect(x, 560 - o * scale, 36, o * scale, fill="#111111"),
                rect(x + 46, 560 - p * scale, 36, p * scale, fill="#7C8A99"),
                rect(x + 92, 560 - l * scale, 36, l * scale, fill=ACCENT),
                t(x + 18, 580, "O", 16, anchor="middle", fill=MUTED),
                t(x + 64, 580, "LP", 16, anchor="middle", fill=MUTED),
                t(x + 110, 580, "L", 16, anchor="middle", fill=ACCENT_DARK),
                t(x + 18, 560 - o * scale - 10, f"{o:.4f}", 14, anchor="middle"),
                t(x + 64, 560 - p * scale - 10, f"{p:.4f}", 14, anchor="middle"),
                t(x + 110, 560 - l * scale - 10, f"{l:.4f}", 14, anchor="middle", fill=ACCENT_DARK),
            ]
        )
    content.append(t(90, 646, "Top-5 相比 Original CLIP：Linear Probe +0.2554，LoRA +0.2893。", 20, fill=ACCENT_DARK))
    return body_frame(13, "结果二：轻量微调显著提升检索性能", "Result II", "\n".join(content))


def slide_14() -> str:
    content = [
        rect(90, 208, 520, 180, fill="#FFFFFF", stroke=INK, sw=2),
        rect(670, 208, 520, 180, fill="#FFFFFF", stroke=INK, sw=2),
        rect(90, 430, 520, 180, fill="#FFFFFF", stroke=INK, sw=2),
        rect(670, 430, 520, 180, fill="#FFFFFF", stroke=INK, sw=2),
        t(116, 252, "A. 投影位置消融", 24, weight="700"),
        t(116, 296, "属性型 Prompt 中，text only 通常优于 vision only。", 20, fill=INK_2),
        t(116, 330, "说明文本侧适配对 text-to-image 更直接。", 20, fill=INK_2),
        t(696, 252, "B. full Prompt 下的最优配置", 24, weight="700"),
        t(696, 296, "text+vision 在 Top-20 Recall 达到 0.9384。", 20, fill=INK_2),
        t(696, 330, "信息足够丰富时，双侧协同最稳定。", 20, fill=INK_2),
        t(116, 474, "C. 典型错误来源", 24, weight="700"),
        t(116, 518, "• 同类商品外观极其接近", 20, fill=INK_2),
        t(116, 550, "• 跨类别边界模糊", 20, fill=INK_2),
        t(116, 582, "• 标题描述质量不足", 20, fill=INK_2),
        t(696, 474, "D. 错误率较高类别", 24, weight="700"),
        rect(730, 528, 180, 18, fill="#111111"),
        rect(730, 560, 154, 18, fill="#444F5D"),
        rect(730, 592, 130, 18, fill=ACCENT),
        t(920, 542, "home product 13.70%", 16),
        t(894, 574, "ring 11.76%", 16),
        t(870, 606, "rug 9.88%", 16, fill=ACCENT_DARK),
    ]
    return body_frame(14, "进一步发现：文本侧更关键，但双侧协同更稳", "Ablation & Errors", "\n".join(content))


def slide_15() -> str:
    body = [
        rect(0, 0, W, H, fill=NAVY),
        rect(72, 72, 1136, 576, fill=NAVY_2, stroke=NAVY_3, sw=2),
        t(640, 140, "结论与展望", 18, family=MONO, fill="#9DB6D3", anchor="middle", extra='letter-spacing="2.6"'),
        t(640, 214, "回到最初的三个问题", 44, weight="700", fill="#F6FAFF", anchor="middle"),
        rect(120, 268, 1040, 76, fill="#11253F", stroke="none"),
        t(640, 316, "答案是：原始 CLIP 能用，但 Prompt 与轻量微调共同决定了性能上限。", 28, weight="700", fill="#FFFFFF", anchor="middle"),
        rect(120, 392, 292, 172, fill="none", stroke="#2E4768", sw=2),
        rect(494, 392, 292, 172, fill="none", stroke="#2E4768", sw=2),
        rect(868, 392, 292, 172, fill="none", stroke="#2E4768", sw=2),
        t(146, 438, "Q1  原始 CLIP 够吗？", 24, weight="700", fill="#A6F1FD"),
        t(146, 486, "能工作，但商品域细粒度", 20, fill="#EAF2FF"),
        t(146, 516, "区分能力仍有提升空间。", 20, fill="#EAF2FF"),
        t(520, 438, "Q2  微调有效吗？", 24, weight="700", fill="#A6F1FD"),
        t(520, 486, "有效，而且提升明显。", 20, fill="#EAF2FF"),
        t(520, 520, "LoRA Recall@5 = 0.8339", 28, weight="700", fill="#FFFFFF"),
        t(894, 438, "Q3  Prompt 重要吗？", 24, weight="700", fill="#A6F1FD"),
        t(894, 486, "非常重要。完整标题 Prompt", 20, fill="#EAF2FF"),
        t(894, 516, "远优于属性拼接。", 20, fill="#EAF2FF"),
        t(120, 618, "后续工作：更大规模商品库、多模态推荐场景、线上检索系统验证。", 22, fill="#99B0C8"),
        t(1160, 618, "谢谢", 22, fill="#FFFFFF", anchor="end"),
    ]
    return wrap_svg("\n".join(body))


GENERATORS = [
    ("01_cover.svg", cover),
    ("02_scene.svg", slide_2),
    ("03_questions.svg", slide_3),
    ("04_task_pipeline.svg", slide_4),
    ("05_dataset.svg", slide_5),
    ("06_prompt_design.svg", slide_6),
    ("07_original_clip.svg", slide_7),
    ("08_linear_probe.svg", slide_8),
    ("09_lora.svg", slide_9),
    ("10_system_impl.svg", slide_10),
    ("11_metrics_setup.svg", slide_11),
    ("12_prompt_results.svg", slide_12),
    ("13_model_results.svg", slide_13),
    ("14_ablation_errors.svg", slide_14),
    ("15_conclusion.svg", slide_15),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in GENERATORS:
        (OUT / name).write_text(builder(), encoding="utf-8")


if __name__ == "__main__":
    main()
