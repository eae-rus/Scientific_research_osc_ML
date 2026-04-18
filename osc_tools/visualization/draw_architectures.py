import graphviz
from pathlib import Path
import os

def setup_graph(name, title):
    dot = graphviz.Digraph(name, format='png')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.4', ranksep='0.6')
    dot.attr('node', fontname='Helvetica,Arial,sans-serif', shape='box', style='rounded,filled', fillcolor='white', fontsize='10')
    dot.attr('edge', fontname='Helvetica,Arial,sans-serif', fontsize='9')
    dot.attr(label=f'\n{title}', fontname='Helvetica-Bold', fontsize='14')
    return dot

def draw_mlp(output_dir):
    dot = setup_graph('SimpleMLP', 'SimpleMLP (Многослойный перцептрон)')
    
    dot.node('In', 'Вход\n[Каналы, Длина]', fillcolor='#e6f2ff')
    dot.node('Flat', 'Развертка\n(Flatten)', fillcolor='#f2f2f2')
    dot.node('H1', 'Скрытый слой 1\n(Узлы с ReLU)', fillcolor='#ffe6e6')
    dot.node('H2', 'Скрытый слой 2\n(Узлы с ReLU)', fillcolor='#ffe6e6')
    dot.node('Out', 'Выход\n(Вектор классов)', fillcolor='#e6ffe6')

    dot.edge('In', 'Flat')
    dot.edge('Flat', 'H1', label=' Веса (Линейно)')
    dot.edge('H1', 'H2', label=' Веса (Линейно)')
    dot.edge('H2', 'Out', label=' Веса (Линейно)')

    dot.render(filename=os.path.join(output_dir, 'SimpleMLP'), cleanup=True)

def draw_cnn(output_dir):
    dot = setup_graph('SimpleCNN', 'SimpleCNN (Сверточная сеть)')
    
    dot.node('In', 'Вход\n[Каналы, Длина]', fillcolor='#e6f2ff')
    dot.node('C1', 'Conv1D + ReLU', fillcolor='#e6e6fa')
    dot.node('P1', 'MaxPooling\n(Сжатие времени)', fillcolor='#f2f2f2')
    dot.node('C2', 'Conv1D + ReLU', fillcolor='#e6e6fa')
    dot.node('Flat', 'Развертка\n(Flatten)', fillcolor='#f2f2f2')
    dot.node('Out', 'Выход\nЛинейный слой', fillcolor='#e6ffe6')

    dot.edges([('In', 'C1'), ('C1', 'P1'), ('P1', 'C2'), ('C2', 'Flat'), ('Flat', 'Out')])
    dot.render(filename=os.path.join(output_dir, 'SimpleCNN'), cleanup=True)

def draw_resnet(output_dir):
    dot = setup_graph('ResNet1D', 'ResNet1D (Остаточная сеть)')
    
    dot.node('In', 'Вход', fillcolor='#e6f2ff')
    dot.node('Stem', 'Conv1D Stem', fillcolor='#e6e6fa')
    
    with dot.subgraph(name='cluster_resblock') as c:
        c.attr(label='Residual Block (xN)', style='dashed', color='gray')
        c.node('Conv1', 'Conv1D -> BN -> ReLU', fillcolor='#e6e6fa')
        c.node('Conv2', 'Conv1D -> BN', fillcolor='#e6e6fa')
        c.node('Add', 'Сумма (+)', shape='circle', fillcolor='#ffdead')
        c.node('ReLU', 'ReLU', fillcolor='#ffe6e6')
        
        c.edge('Conv1', 'Conv2')
        c.edge('Conv2', 'Add')
        c.edge('Add', 'ReLU')

    dot.node('GAP', 'Global Avg Pool', fillcolor='#f2f2f2')
    dot.node('Out', 'Выход', fillcolor='#e6ffe6')

    dot.edge('In', 'Stem')
    dot.edge('Stem', 'Conv1')
    dot.edge('Stem', 'Add', label=' Skip Connection', style='dotted', color='blue')
    dot.edge('ReLU', 'GAP')
    dot.edge('GAP', 'Out')

    dot.render(filename=os.path.join(output_dir, 'ResNet1D'), cleanup=True)

def draw_simplekan(output_dir):
    dot = setup_graph('SimpleKAN', 'SimpleKAN (Полносвязная сеть Колмогорова-Арнольда)')
    
    dot.node('In', 'Вход', fillcolor='#e6f2ff')
    dot.node('Flat', 'Развертка', fillcolor='#f2f2f2')
    dot.node('H1', 'Σ', shape='circle')
    dot.node('H2', 'Σ', shape='circle')
    dot.node('Out', 'Выход', fillcolor='#e6ffe6')

    dot.edge('In', 'Flat')
    
    kan_edge = {'color': '#ffa500', 'fontcolor': '#cc6600', 'label': '1D B-spline'}
    dot.edge('Flat', 'H1', **kan_edge)
    dot.edge('H1', 'H2', **kan_edge)
    dot.edge('H2', 'Out', **kan_edge)

    dot.render(filename=os.path.join(output_dir, 'SimpleKAN'), cleanup=True)

def draw_convkan(output_dir):
    dot = setup_graph('ConvKAN', 'ConvKAN (Сверточная сеть Колмогорова-Арнольда)')
    
    dot.node('In', 'Вход', fillcolor='#e6f2ff')
    dot.node('CK1', 'ConvKAN Layer 1\n(Сплайны по локальному окну)', fillcolor='#fffacd')
    dot.node('P1', 'Pooling', fillcolor='#f2f2f2')
    dot.node('CK2', 'ConvKAN Layer 2', fillcolor='#fffacd')
    dot.node('Flat', 'Развертка', fillcolor='#f2f2f2')
    dot.node('Out', 'Выход (SimpleKAN)', fillcolor='#e6ffe6')

    dot.edges([('In', 'CK1'), ('CK1', 'P1'), ('P1', 'CK2'), ('CK2', 'Flat')])
    dot.edge('Flat', 'Out', color='#ffa500', label=' B-spline')

    dot.render(filename=os.path.join(output_dir, 'ConvKAN'), cleanup=True)

def draw_physicskan(output_dir):
    dot = setup_graph('PhysicsKAN', 'PhysicsKAN (Физ.-информированный KAN)')
    
    with dot.subgraph(name='cluster_in') as c:
        c.attr(label='Входы', color='gray', style='dashed')
        c.node('I', 'Токи (I)', fillcolor='#e6f2ff')
        c.node('U', 'Напряжения (U)', fillcolor='#e6f2ff')

    with dot.subgraph(name='cluster_phys') as c:
        c.attr(label='Физический слой', fillcolor='#e6ffe6', style='filled', color='#98fb98')
        c.node('Mul', 'S ~ I × U', shape='box', fillcolor='#ccffcc')
        c.node('Div', 'Z ~ U / I', shape='box', fillcolor='#ccffcc')
        c.node('Cat', 'Конкатенация\n[I, U, S, Z]', fillcolor='#b3ffb3', shape='cds')

    dot.node('KAN', 'Обработка\n(ConvKAN / SimpleKAN)', fillcolor='#fffacd')
    dot.node('Out', 'Выход', fillcolor='#e6ffe6')

    dot.edge('I', 'Mul')
    dot.edge('U', 'Mul')
    dot.edge('I', 'Div')
    dot.edge('U', 'Div')
    
    dot.edge('I', 'Cat', style='dotted', color='blue')
    dot.edge('U', 'Cat', style='dotted', color='blue')
    dot.edge('Mul', 'Cat')
    dot.edge('Div', 'Cat')
    
    dot.edge('Cat', 'KAN')
    dot.edge('KAN', 'Out')

    dot.render(filename=os.path.join(output_dir, 'PhysicsKAN'), cleanup=True)

def draw_cphysicskan(output_dir):
    dot = setup_graph('cPhysicsKAN', 'cPhysicsKAN (Комплексный PhysicsKAN в полярных координатах)')
    
    with dot.subgraph(name='cluster_in') as c:
        c.attr(label='Полярные Входы', color='gray', style='dashed')
        c.node('IA', 'Амплитуда Тока (|I|)', fillcolor='#e6f2ff')
        c.node('IP', 'Фаза Тока (∠I)', fillcolor='#e0e0eb')
        c.node('UA', 'Амплитуда Напр. (|U|)', fillcolor='#e6f2ff')
        c.node('UP', 'Фаза Напр. (∠U)', fillcolor='#e0e0eb')

    with dot.subgraph(name='cluster_phys') as c:
        c.attr(label='Комплексный Физический слой', fillcolor='#e6ffe6', style='filled', color='#98fb98')
        c.node('cMul', 'Комплексное Умножение S\n|S| = |I|×|U|\n∠S = ∠I + ∠U', fillcolor='#ccffcc')
        c.node('cDiv', 'Комплексное Деление Y\n|Y| = |I|/|U|\n∠Y = ∠I - ∠U', fillcolor='#ccffcc')
        c.node('Cat', 'Конкатенация\nс исходными I и U')
        c.node('Norm', 'BN & Dropout\n(ТОЛЬКО для Амплитуд)', shape='box', fillcolor='#e6b3ff')

    dot.node('KAN', 'Комплексная обработка\n(KAN)', fillcolor='#fffacd')
    dot.node('Out', 'Выход', fillcolor='#e6ffe6')

    dot.edge('IA', 'cMul')
    dot.edge('UA', 'cMul')
    dot.edge('IP', 'cMul', style='dashed')
    dot.edge('UP', 'cMul', style='dashed')

    dot.edge('IA', 'cDiv')
    dot.edge('UA', 'cDiv')
    dot.edge('IP', 'cDiv', style='dashed')
    dot.edge('UP', 'cDiv', style='dashed')

    dot.edge('cMul', 'Cat')
    dot.edge('cDiv', 'Cat')
    dot.edge('IA', 'Cat', style='dotted')
    dot.edge('IP', 'Cat', style='dotted')
    
    dot.edge('Cat', 'Norm')
    dot.edge('Norm', 'KAN')
    dot.edge('KAN', 'Out')

    dot.render(filename=os.path.join(output_dir, 'cPhysicsKAN'), cleanup=True)

if __name__ == '__main__':
    output_directory = Path(__file__).parent / 'architecture_images'
    output_directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Генерация схем архитектур в директорию {output_directory}...")
    
    draw_mlp(output_directory)
    draw_cnn(output_directory)
    draw_resnet(output_directory)
    draw_simplekan(output_directory)
    draw_convkan(output_directory)
    draw_physicskan(output_directory)
    draw_cphysicskan(output_directory)
    
    print("Готово!")
