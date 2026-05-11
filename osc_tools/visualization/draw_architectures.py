"""Генерация концептуальных схем архитектур нейронных сетей для научной статьи.

Рисует модели из Фаз 2.6 (MLP, CNN, ResNet, SimpleKAN, ConvKAN, PhysicsKAN, cPhysicsKAN)
и Фазы 4 (BaselineTransformer, PhysicalKANTransformer).

Схемы — концептуальные (показывают принцип, а не точные гиперпараметры), но верные
по формулам и последовательности операций согласно исходному коду
(см. osc_tools/ml/models/kan.py, transformer.py, cnn.py, baseline.py).
"""
import graphviz
from pathlib import Path
import os

# Единая палитра
C_INPUT     = '#e6f2ff'   # Входы (амплитуды, обычные данные)
C_INPUT_PH  = '#e0e0eb'   # Входы-фазы (отличаются от амплитуд)
C_CONV      = '#e6e6fa'   # Свёрточные/линейные "классические" слои
C_ACTIV     = '#ffe6e6'   # Активации (ReLU, GELU)
C_UTIL      = '#f2f2f2'   # Утилиты (Flatten, Pool, LayerNorm)
C_PHYS      = '#ccffcc'   # Физические операции
C_PHYS_BG   = '#e6ffe6'   # Фон физического кластера
C_KAN       = '#fffacd'   # KAN-слои и сплайны
C_NORM      = '#e6b3ff'   # Специфическая нормализация/дропаут
C_ATTN      = '#ffe4b5'   # Слои attention
C_OUT       = '#d9f2d9'   # Выходные головы


def setup_graph(name, title, rankdir='LR'):
    dot = graphviz.Digraph(name, format='png')
    dot.attr(rankdir=rankdir, splines='polyline', nodesep='0.35', ranksep='0.55')
    dot.attr('node', fontname='Helvetica,Arial,sans-serif', shape='box',
             style='rounded,filled', fillcolor='white', fontsize='10')
    dot.attr('edge', fontname='Helvetica,Arial,sans-serif', fontsize='9')
    dot.attr(label=f'\n{title}', fontname='Helvetica-Bold', fontsize='14', labelloc='b')
    return dot


# =====================================================================
# Фаза 2.6
# =====================================================================

def draw_mlp(output_dir):
    dot = setup_graph('SimpleMLP', 'SimpleMLP (Многослойный перцептрон)')
    dot.node('In',   'Вход\n(B, C, T)',            fillcolor=C_INPUT)
    dot.node('Flat', 'Flatten\n→ (B, C·T)',         fillcolor=C_UTIL)
    dot.node('Hid',  'N× блоков:\nLinear → BN → ReLU → Dropout', fillcolor=C_ACTIV)
    dot.node('Head', 'Linear\n(logits)',            fillcolor=C_OUT)
    dot.edges([('In', 'Flat'), ('Flat', 'Hid'), ('Hid', 'Head')])
    dot.render(filename=os.path.join(output_dir, 'SimpleMLP'), cleanup=True)


def draw_cnn(output_dir):
    dot = setup_graph('SimpleCNN', 'SimpleCNN (Одномерная свёрточная сеть)')
    dot.node('In',   'Вход\n(B, C, T)',                 fillcolor=C_INPUT)
    dot.node('Blk',  'N× блоков:\nConv1D → BN → ReLU\n→ MaxPool → Dropout', fillcolor=C_CONV)
    dot.node('GAP',  'AdaptiveAvgPool1D\n→ Flatten',    fillcolor=C_UTIL)
    dot.node('FC',   'Linear → ReLU\n→ Linear',         fillcolor=C_ACTIV)
    dot.node('Out',  'Logits',                          fillcolor=C_OUT)
    dot.edges([('In', 'Blk'), ('Blk', 'GAP'), ('GAP', 'FC'), ('FC', 'Out')])
    dot.render(filename=os.path.join(output_dir, 'SimpleCNN'), cleanup=True)


def draw_resnet(output_dir):
    dot = setup_graph('ResNet1D', 'ResNet1D (Остаточная свёрточная сеть)')
    dot.node('In',   'Вход',                            fillcolor=C_INPUT)
    dot.node('Stem', 'Conv1D Stem\n(k=7, s=2) → BN → ReLU\n→ MaxPool', fillcolor=C_CONV)
    with dot.subgraph(name='cluster_res') as c:
        c.attr(label='ResBlock × N  (обычно 4 стадии × 2)',
               style='dashed', color='gray', fontname='Helvetica-Bold')
        c.node('C1',  'Conv1D → BN → ReLU', fillcolor=C_CONV)
        c.node('C2',  'Conv1D → BN',        fillcolor=C_CONV)
        c.node('Add', '+',  shape='circle', fillcolor='#ffdead')
        c.node('Act', 'ReLU',               fillcolor=C_ACTIV)
        c.edges([('C1', 'C2'), ('C2', 'Add'), ('Add', 'Act')])
    dot.node('GAP',  'GlobalAvgPool → Flatten', fillcolor=C_UTIL)
    dot.node('Out',  'Linear\n(logits)',        fillcolor=C_OUT)
    dot.edge('In',   'Stem')
    dot.edge('Stem', 'C1')
    dot.edge('Stem', 'Add', label='  skip', style='dotted', color='#1f4fe0')
    dot.edge('Act',  'GAP')
    dot.edge('GAP',  'Out')
    dot.render(filename=os.path.join(output_dir, 'ResNet1D'), cleanup=True)


def draw_simplekan(output_dir):
    dot = setup_graph('SimpleKAN', 'SimpleKAN (Полносвязный Колмогоров-Арнольд)')
    dot.node('In',   'Вход → Flatten',   fillcolor=C_INPUT)
    dot.node('S1',   'Σ', shape='circle', fillcolor='white')
    dot.node('S2',   'Σ', shape='circle', fillcolor='white')
    dot.node('Out',  'Logits',           fillcolor=C_OUT)
    kan = {'color': '#cc6600', 'fontcolor': '#cc6600',
           'label': '  B-spline + SiLU'}
    dot.edge('In', 'S1', **kan)
    dot.edge('S1', 'S2', **kan)
    dot.edge('S2', 'Out', **kan)
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Особенность KAN:\nнелинейность на рёбрах, узлы = суммы',
               style='dashed', color='#cc6600', fontsize='10')
    dot.render(filename=os.path.join(output_dir, 'SimpleKAN'), cleanup=True)


def draw_convkan(output_dir):
    dot = setup_graph('ConvKAN', 'ConvKAN (Сверточный Колмогоров-Арнольд)')
    dot.node('In',   'Вход\n(B, C, T)',  fillcolor=C_INPUT)
    dot.node('Blk',  'N× блоков:\nKANConv1D → BN\n→ MaxPool → Dropout',
             fillcolor=C_KAN)
    dot.node('GAP',  'AdaptiveAvgPool1D\n→ Flatten', fillcolor=C_UTIL)
    dot.node('Head', 'KANLinear → KANLinear\n(logits)', fillcolor=C_KAN)
    dot.edges([('In', 'Blk'), ('Blk', 'GAP'), ('GAP', 'Head')])
    dot.render(filename=os.path.join(output_dir, 'ConvKAN'), cleanup=True)


def draw_physicskan(output_dir):
    dot = setup_graph('PhysicsKAN', 'PhysicsKAN (Физически-информированный KAN)')

    with dot.subgraph(name='cluster_in') as c:
        c.attr(label='Вход  (B, C, T)', style='dashed', color='gray',
               fontname='Helvetica-Bold')
        c.node('I', 'Токи  I\n(1-я половина каналов)',        fillcolor=C_INPUT)
        c.node('U', 'Напряжения  U\n(2-я половина каналов)',  fillcolor=C_INPUT)

    with dot.subgraph(name='cluster_phys') as c:
        c.attr(label='Физический слой', style='filled', fillcolor=C_PHYS_BG,
               color='#98fb98', fontname='Helvetica-Bold')
        c.node('Mul', 'MultiplicationLayer\nS = I · U\n(поэлементно)',
               fillcolor=C_PHYS)
        c.node('Div', 'DivisionLayer\nZ = I / U\n(защита от 0)',
               fillcolor=C_PHYS)
        c.node('BN1', 'BatchNorm1D', fillcolor=C_UTIL)
        c.node('BN2', 'BatchNorm1D', fillcolor=C_UTIL)
        c.node('Cat', 'Concat по каналам\n[I, U, S, Z]\n→ 2C каналов',
               shape='cds', fillcolor='#b3ffb3')

    dot.node('KAN', 'Processing network\nConvKAN\n(N KAN-Conv блоков)\n→ GAP → KANLinear×2',
             fillcolor=C_KAN)
    dot.node('Out', 'Logits', fillcolor=C_OUT)

    dot.edge('I', 'Mul');   dot.edge('U', 'Mul')
    dot.edge('I', 'Div');   dot.edge('U', 'Div')
    dot.edge('Mul', 'BN1'); dot.edge('BN1', 'Cat')
    dot.edge('Div', 'BN2'); dot.edge('BN2', 'Cat')
    dot.edge('I', 'Cat', style='dotted', color='#1f4fe0', label='  skip')
    dot.edge('U', 'Cat', style='dotted', color='#1f4fe0', label='  skip')
    dot.edge('Cat', 'KAN')
    dot.edge('KAN', 'Out')
    dot.render(filename=os.path.join(output_dir, 'PhysicsKAN'), cleanup=True)


def draw_cphysicskan(output_dir):
    dot = setup_graph('cPhysicsKAN',
                      'cPhysicsKAN (Комплексный PhysicsKAN в полярных координатах)')

    with dot.subgraph(name='cluster_in') as c:
        c.attr(label='Вход:  [A, φ, A, φ, …]   (пары I и U)',
               style='dashed', color='gray', fontname='Helvetica-Bold')
        c.node('AI', 'Амплитуды тока  |I|',       fillcolor=C_INPUT)
        c.node('PI', 'Фазы тока  ∠I',             fillcolor=C_INPUT_PH)
        c.node('AU', 'Амплитуды напряж.  |U|',    fillcolor=C_INPUT)
        c.node('PU', 'Фазы напряж.  ∠U',          fillcolor=C_INPUT_PH)

    with dot.subgraph(name='cluster_phys') as c:
        c.attr(label='Комплексный физический слой', style='filled',
               fillcolor=C_PHYS_BG, color='#98fb98', fontname='Helvetica-Bold')
        c.node('cMul',
               'ComplexMultiplicationLayer\n'
               '|S| = |I|·|U|\n'
               '∠S = ∠I + ∠U + bφ',
               fillcolor=C_PHYS)
        c.node('cDiv',
               'ComplexDivisionLayer\n'
               '|Y| = |I| / max(|U|, ε)\n'
               '∠Y = ∠I − ∠U + bφ',
               fillcolor=C_PHYS)
        c.node('NormS',
               'BatchNorm1D\n(только |S|)\n+ ComplexPairDropout',
               fillcolor=C_NORM)
        c.node('NormZ',
               'BatchNorm1D\n(только |Y|)\n+ ComplexPairDropout',
               fillcolor=C_NORM)
        c.node('Cat',
               'Concat по каналам\n[x, S, Y]\n→ 2C каналов\n(чередование  A, φ)',
               shape='cds', fillcolor='#b3ffb3')

    dot.node('KAN',
             'Processing network\nConvKAN\n(обрабатывает A и φ\nкак обычные каналы)',
             fillcolor=C_KAN)
    dot.node('Out', 'Logits', fillcolor=C_OUT)

    # В комплексное умножение/деление идут обе пары (A и φ)
    for src in ('AI', 'AU'):
        dot.edge(src, 'cMul'); dot.edge(src, 'cDiv')
    for src in ('PI', 'PU'):
        dot.edge(src, 'cMul', style='dashed')
        dot.edge(src, 'cDiv', style='dashed')

    dot.edge('cMul', 'NormS'); dot.edge('NormS', 'Cat')
    dot.edge('cDiv', 'NormZ'); dot.edge('NormZ', 'Cat')
    # skip всех четырёх исходных каналов
    for src in ('AI', 'PI', 'AU', 'PU'):
        dot.edge(src, 'Cat', style='dotted', color='#1f4fe0')
    dot.edge('Cat', 'KAN')
    dot.edge('KAN', 'Out')
    dot.render(filename=os.path.join(output_dir, 'cPhysicsKAN'), cleanup=True)


# =====================================================================
# Фаза 4
# =====================================================================

def draw_baseline_transformer(output_dir):
    dot = setup_graph('BaselineTransformer',
                      'BaselineTransformer (Спектральный Transformer, Phase 4)')

    dot.node('In',    'Вход (спектр.)\n(B, C, T)',        fillcolor=C_INPUT)
    dot.node('San',   'DataSanitizer\n(NaN→0 + missing_token)', fillcolor=C_UTIL)
    dot.node('Stem',  'MLP-Stem\nLinear → GELU → LayerNorm → Dropout',
             fillcolor=C_CONV)
    dot.node('PE',    'Sinusoidal\nPositional Encoding', fillcolor=C_UTIL)

    with dot.subgraph(name='cluster_enc') as c:
        c.attr(label='TransformerEncoderBlock × N\n(Pre-LN, N = 4 по умолч.)',
               style='dashed', color='gray', fontname='Helvetica-Bold')
        c.node('LN1',  'LayerNorm',                        fillcolor=C_UTIL)
        c.node('MHA',  'MultiheadAttention',               fillcolor=C_ATTN)
        c.node('A1',   '+',  shape='circle', fillcolor='#ffdead')
        c.node('LN2',  'LayerNorm',                        fillcolor=C_UTIL)
        c.node('FFN',  'MLP-FFN\nLinear → GELU\n→ Linear', fillcolor=C_ACTIV)
        c.node('A2',   '+',  shape='circle', fillcolor='#ffdead')
        c.edges([('LN1', 'MHA'), ('MHA', 'A1'), ('A1', 'LN2'),
                 ('LN2', 'FFN'), ('FFN', 'A2')])

    dot.node('LNF',   'LayerNorm (final)', fillcolor=C_UTIL)
    dot.node('Head',  'Головы:\n• SSL: Linear (recon.)\n• CLS: zone-avg → Linear\n• (опц.) Future Head',
             fillcolor=C_OUT)

    dot.edges([('In', 'San'), ('San', 'Stem'), ('Stem', 'PE'),
               ('PE', 'LN1'),
               ('A2', 'LNF'), ('LNF', 'Head')])
    dot.edge('Stem', 'A1', style='dotted', color='#1f4fe0', label='  skip')
    dot.edge('A1',   'A2', style='dotted', color='#1f4fe0', label='  skip')
    dot.render(filename=os.path.join(output_dir, 'BaselineTransformer'), cleanup=True)


def draw_physical_kan_transformer(output_dir):
    dot = setup_graph('PhysicalKANTransformer',
                      'PhysicalKANTransformer (Физически-информированный Transformer, Phase 4)',
                      rankdir='TB')

    dot.node('In',   'Вход (спектр., полярн.)\n[A₁, φ₁, A₂, φ₂, …]\n(B, C, T)',
             fillcolor=C_INPUT)
    dot.node('San',  'DataSanitizer\n(маска NaN + missing_token)', fillcolor=C_UTIL)

    with dot.subgraph(name='cluster_stem') as c:
        c.attr(label='PhysicalStem  (inductive bias)',
               style='filled', fillcolor=C_PHYS_BG,
               color='#98fb98', fontname='Helvetica-Bold')
        c.node('Sep', 'Разделение:\nчётные → A,  нечётные → φ', fillcolor=C_UTIL)
        c.node('Rot', 'φ ← φ + bφ\n(обучаемый сдвиг)',          fillcolor=C_PHYS)
        c.node('Gate','DirectionalRelayGate\nangle_coeff ∈ [α, 1]', fillcolor=C_NORM)
        c.node('GAmp','A_gated = A · angle_coeff',              fillcolor=C_PHYS)
        c.node('CIB', 'ComplexInteractionBlock\n'
                      'z = polar(A, φ)\n'
                      '→ комплексное  · и ÷\n'
                      '(обучаемые пары)',
               fillcolor=C_PHYS)
        c.node('KAmp','FastKAN\n(только амплитудная ветвь)',    fillcolor=C_KAN)
        c.node('FuseA','Concat A | proj Linear → d_model/2',    fillcolor=C_CONV)
        c.node('FuseP','Concat φ | proj Linear → d_model/2',    fillcolor=C_CONV)
        c.node('Emb', 'Embedding = [A‖φ]\n(d_model)',           fillcolor=C_UTIL)
        c.node('LN0', 'AmpOnlyLayerNorm + Dropout',             fillcolor=C_NORM)

    dot.node('PE',   'Sinusoidal PE', fillcolor=C_UTIL)

    with dot.subgraph(name='cluster_enc') as c:
        c.attr(label='TransformerEncoderBlock × N  (Pre-LN, N = 4)',
               style='dashed', color='gray', fontname='Helvetica-Bold')
        c.node('LN1', 'AmpOnlyLayerNorm',           fillcolor=C_NORM)
        c.node('cAttn','ComplexMultiheadAttention\n'
                      'score = QₐKₐ + Q_φK_φ\n'
                      'раздельные W для A и φ',
               fillcolor=C_ATTN)
        c.node('A1',  '+', shape='circle', fillcolor='#ffdead')
        c.node('LN2', 'AmpOnlyLayerNorm',           fillcolor=C_NORM)

        with c.subgraph(name='cluster_ffn') as ff:
            ff.attr(label='PhysicalKANFeedForward', style='filled',
                    fillcolor=C_PHYS_BG, color='#98fb98')
            ff.node('Split','Split x → (A, φ)', fillcolor=C_UTIL)
            ff.node('KAN2','FastKAN × 2\n(d_c → d_ff/2 → d_c)', fillcolor=C_KAN)
            ff.node('Gate2','DirectionalRelayGate (по φ)\n→ h_A · gate', fillcolor=C_NORM)
            ff.node('CIB2','ComplexInteractionBlock (малый)\n+ σ(g)·proj', fillcolor=C_PHYS)
            ff.node('Phi', 'φ_out = bφ\n(обучаемый сдвиг)',   fillcolor=C_PHYS)
            ff.node('Merge','[h_A‖φ_out]',                    fillcolor=C_UTIL)
            ff.edges([('Split','KAN2'),('KAN2','Gate2'),('Gate2','Merge'),
                      ('Split','CIB2'),('CIB2','Gate2'),('Split','Phi'),
                      ('Phi','Merge')])

        c.node('A2',  '+', shape='circle', fillcolor='#ffdead')
        c.edges([('LN1','cAttn'),('cAttn','A1'),('A1','LN2'),('LN2','Split'),
                 ('Merge','A2')])

    dot.node('LNF',  'AmpOnlyLayerNorm (final)', fillcolor=C_NORM)
    dot.node('Head',
             'Головы:\n'
             '• SSL-reconstruction (Linear)\n'
             '• KAN-classifier: FastKAN → Linear\n'
             '• (опц.) FuturePredictionHead',
             fillcolor=C_OUT)

    dot.edges([
        ('In','San'),('San','Sep'),
        ('Sep','Rot'),('Rot','Gate'),('Gate','GAmp'),
        ('Sep','CIB'),('Rot','CIB'),      # CIB берёт НЕгейтированные A
        ('GAmp','KAmp'),
        ('KAmp','FuseA'),('CIB','FuseA'),
        ('Rot','FuseP'),('CIB','FuseP'),
        ('FuseA','Emb'),('FuseP','Emb'),
        ('Emb','LN0'),
        ('LN0','PE'),('PE','LN1'),
        ('A2','LNF'),('LNF','Head'),
    ])
    dot.edge('LN0','A1', style='dotted', color='#1f4fe0', label='  skip')
    dot.edge('A1','A2',  style='dotted', color='#1f4fe0', label='  skip')
    dot.render(filename=os.path.join(output_dir, 'PhysicalKANTransformer'), cleanup=True)


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    output_directory = Path(__file__).parent / 'architecture_images'
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f'Генерация схем в  {output_directory} …')

    # Фаза 2.6
    draw_mlp(output_directory)
    draw_cnn(output_directory)
    draw_resnet(output_directory)
    draw_simplekan(output_directory)
    draw_convkan(output_directory)
    draw_physicskan(output_directory)
    draw_cphysicskan(output_directory)

    # Фаза 4
    draw_baseline_transformer(output_directory)
    draw_physical_kan_transformer(output_directory)

    print('Готово.')
