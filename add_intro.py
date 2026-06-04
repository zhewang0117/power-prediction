#!/usr/bin/env python3
"""Add introduction/background section to thesis outline docx."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('docs/thesis_outline.docx')

# Find the first paragraph to insert before
# We'll insert at the beginning - after the title
# Strategy: create a new doc with intro first, then copy original content
# Simpler: just add sections at the beginning

# Actually, let's rebuild the doc with intro first
template = doc

# Create new document
new_doc = Document()

# Copy styles
style = new_doc.styles['Normal']
style.font.name = '宋体'
style.font.size = Pt(11)

# ============================================================
# Title
# ============================================================
title = new_doc.add_heading('台风期间多源数据驱动的风功率预测', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

new_doc.add_paragraph()

# ============================================================
# 一、研究背景
# ============================================================
new_doc.add_heading('一、研究背景与问题', level=1)

# 1.1
new_doc.add_heading('1.1 风电发展与台风威胁', level=2)

p = new_doc.add_paragraph()
p.add_run(
    '截至2024年底，中国风电累计并网装机容量突破5.2亿千瓦，其中海上风电约3910万千瓦，'
    '连续多年位居全球第一。海上风电开发集中于广东、福建、浙江、江苏、海南等东南沿海省份，'
    '这些区域恰好是西北太平洋台风的主要影响区。据统计，年均约7个台风登陆中国沿海，'
    '影响范围覆盖了绝大部分海上风电基地。'
)

p = new_doc.add_paragraph()
p.add_run(
    '台风对风电场的影响是多维度的。在物理安全层面，超强台风可直接导致风机倒塔——'
    '2024年台风"摩羯"登陆海南文昌，造成多个风电场风机严重损毁。在运行层面，当风速超过'
    '风机切出风速（通常约25 m/s）时，机组自动停机保护，风电场出力在数小时内从满发骤降至零，'
    '对电网调度形成强烈冲击。在经济层面，台风季（6—11月）与非台风季的发电量差异可达上千万千瓦时。'
)

# 1.2
new_doc.add_heading('1.2 台风期间功率预测的挑战', level=2)

p = new_doc.add_paragraph()
p.add_run(
    '风功率预测是电网调度运行的基础。当前主流的预测方法——无论是基于数值天气预报（NWP）'
    '的物理方法，还是基于深度学习的时序方法（如LSTM、BiLSTM）——在常规天气条件下已达到较高精度'
    '（RMSE约10%—15%额定容量）。然而，在台风期间，预测误差显著增大，主要原因包括：'
)

challenges = [
    ('台风样本稀疏', '台风是极端稀有事件。以6年数据为例，台风影响时段仅占总时长的约3%，'
     '深度学习模型难以从如此少量的样本中学习到台风特定的预测模式。'),
    ('NWP风速偏差', '数值天气预报在台风内核区的风速估计存在系统性偏差。'
     '台风眼墙附近的风速梯度极大（数十米/秒量级的变化），而NWP的时空分辨率（如ERA5约31 km）'
     '不足以精确捕捉这种小尺度结构。'),
    ('控制策略扰动', '台风期间风机频繁切出/再切入，功率输出不再遵循标准功率曲线，'
     '而是受机组控制策略的支配。这种非物理的功率变化难以被纯气象驱动的预测模型捕捉。'),
    ('路径不确定性', '台风路径的微小偏移可导致风电场经历完全不同的风况。'
     '确定性预报给出的单一路径无法反映这种不确定性，导致预测的可靠性不足。'),
]

for title, desc in challenges:
    p = new_doc.add_paragraph()
    p.add_run(f'（{challenges.index((title, desc)) + 1}）{title}：').bold = True
    p.add_run(desc)

# 1.3
new_doc.add_heading('1.3 台风参数化风场模型的潜力', level=2)

p = new_doc.add_paragraph()
p.add_run(
    '台风参数化风场模型（Parametric Wind Field Model）是一种基于少量台风特征参数'
    '（中心气压、最大风速、移动速度等）计算空间风场分布的物理模型。经典的Holland（1980）'
    '梯度风模型和Yan Meng（1995）地表风模型已广泛用于台风风场模拟和风暴潮计算。近年来，'
    '多项研究对这些模型的关键参数化方案进行了改进：Willoughby等（2006）提出了更准确的眼墙半径'
    '（Rmax）公式，Powell等（2003）发现了高风速下海面拖曳系数的非单调变化，Lin和Chavas（2012）'
    '修正了台风移动速度对地表风的不对称影响。'
)

p = new_doc.add_paragraph()
p.add_run(
    '参数化模型的核心优势在于：它仅需要台风最佳路径数据（位置、气压、风速）即可计算风场，'
    '独立于NWP。这意味着它提供了NWP之外的独立风场信息源。更重要的是，参数模型不仅输出风速估计，'
    '还输出描述台风风场结构的物理参数——眼墙半径Rmax、Holland B参数（气压梯度陡度）、'
    '移速依赖的衰减系数等。这些结构参数刻画了每个台风的"个性"，而这是NWP所不显式提供的。'
)

# 1.4
new_doc.add_heading('1.4 现有研究的不足', level=2)

p = new_doc.add_paragraph()
p.add_run(
    '尽管参数化风场模型在台风风场模拟中已有成熟应用，但将其系统性地引入风功率预测的研究尚不充分。'
    '现有工作主要存在以下不足：'
)

gaps = [
    '参数模型输出的利用方式单一：现有研究多将参数模型风速直接替代NWP风速作为预测输入，'
    '但参数模型风速在远离台风中心的区域精度不如NWP，直接替代反而可能降低整体预测精度。'
    '如何更有效地利用参数模型数据——尤其是结构参数——是一个有待探索的问题。',
    '缺乏不同利用路径的系统对比：参数模型数据可以通过"外部校正"（后处理修正预测值）或'
    '"内部适应"（微调预测模型本身）两种范式进入预测框架，但两种范式的效果差异、适用条件'
    '和物理机制缺乏系统性研究。',
    '未考虑风机控制策略与台风物理过程的耦合：台风期间的风机切出/再切入行为是一种控制驱动的'
    '功率变化，而非气象驱动的功率变化。现有预测方法未将控制策略纳入建模框架。',
    '确定性预测未反映台风路径不确定性：台风集合预报已可提供路径概率信息，但如何将其与参数模型'
    '耦合，实现概率化的风功率预测，尚缺乏完整的方法框架。',
]

for i, gap in enumerate(gaps):
    p = new_doc.add_paragraph()
    p.add_run(f'({i+1}) ').bold = True
    p.add_run(gap)

# ============================================================
# 二、研究目标与内容
# ============================================================
new_doc.add_heading('二、研究目标与内容', level=1)

p = new_doc.add_paragraph()
p.add_run(
    '本文旨在建立一套系统的方法框架，将改进的台风参数化风场模型与深度学习功率预测有机结合，'
    '解决台风期间的功率预测精度和可靠性问题。具体包括三个递进的研究内容：'
)

p = new_doc.add_paragraph()
p.add_run('工作1——台风参数模型驱动的风功率预测方法：').bold = True
p.add_run(
    '探索参数模型数据进入预测框架的两种范式（外部校正与内部微调），系统对比风速参数与结构参数'
    '的预测价值，确定最优的模型-数据耦合方式。'
)

p = new_doc.add_paragraph()
p.add_run('工作2——考虑风电机组控制策略的台风功率预测：').bold = True
p.add_run(
    '从历史功率数据中识别台风期间的风机控制行为（切出/再切入），建立控制约束模型，'
    '将其融入预测框架以提升台风期的预测精度。'
)

p = new_doc.add_paragraph()
p.add_run('工作3——基于台风集合预报的概率化功率预测：').bold = True
p.add_run(
    '利用台风集合预报的多路径信息，通过参数模型生成概率风场，驱动概率化功率预测，'
    '为电网调度提供不确定性量化信息。'
)

# ============================================================
# Add a page break, then copy original content
# ============================================================
new_doc.add_page_break()

# Now copy the original document content
new_doc.add_heading('附录：工作方案详细说明', level=1)
new_doc.add_paragraph('（以下为与导师讨论使用的工作方案）')
new_doc.add_paragraph()

# Copy paragraphs from original doc
for para in template.paragraphs:
    new_para = new_doc.add_paragraph()
    for run in para.runs:
        new_run = new_para.add_run(run.text)
        if run.bold:
            new_run.bold = True
        if run.font.size:
            new_run.font.size = run.font.size

# Copy tables from original doc
for table in template.tables:
    new_table = new_doc.add_table(rows=len(table.rows), cols=len(table.columns))
    new_table.style = 'Light Grid Accent 1'
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            new_table.rows[r_idx].cells[c_idx].text = cell.text

new_doc.save('docs/thesis_outline.docx')
print('Done: docs/thesis_outline.docx (with introduction)')
