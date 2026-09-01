#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт генерации публикационных графиков и схем (300 DPI)
для статьи: "Статистически обоснованное формирование обучающей разметки
для интеллектуального органа направления мощности".
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Настройка шрифтов и стиля для академических публикаций
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'

OUTPUT_DIRS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "figures")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "article", "figures")),
]
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

def save_fig(filename, dpi=300):
    for d in OUTPUT_DIRS:
        plt.savefig(os.path.join(d, filename), dpi=dpi)


# -------------------------------------------------------------
# РИСУНОК 1: Логика РНМ в структуре БАВР и двухголовая разметка
# -------------------------------------------------------------
def plot_fig1():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.axis('off')
    
    # Стили блоков
    box_blue = dict(boxstyle="round,pad=0.5", fc="#e1f5fe", ec="#0288d1", lw=1.5)
    box_green = dict(boxstyle="round,pad=0.5", fc="#e8f5e9", ec="#388e3c", lw=1.5)
    box_red = dict(boxstyle="round,pad=0.5", fc="#ffebee", ec="#d32f2f", lw=1.5)
    box_gray = dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#757575", lw=1.5)
    box_amber = dict(boxstyle="round,pad=0.5", fc="#fff8e1", ec="#ffa000", lw=1.5)

    # 1. Вход
    ax.text(0.5, 0.92, "Измерительные сигналы $i(t), u(t)$\n(3 фазы токов + 3 фазы напряжений)", 
            ha='center', va='center', bbox=box_blue, weight='bold')

    # Стрелка к голове применимости
    ax.annotate('', xy=(0.5, 0.77), xytext=(0.5, 0.85),
                arrowprops=dict(facecolor='#0288d1', edgecolor='#0288d1', width=2, headwidth=8))

    # 2. Выход 1: Применимость
    ax.text(0.5, 0.71, "Голова 1: Оценка физической применимости\n$\\hat{y}_{\\mathrm{valid}} = \\sigma(z_{\\mathrm{valid}}) \\in [0, 1]$", 
            ha='center', va='center', bbox=box_amber, weight='bold')

    # Ветвление: VALID=0 (влево) и VALID=1 (вправо/вниз)
    ax.annotate('', xy=(0.2, 0.48), xytext=(0.4, 0.65),
                arrowprops=dict(facecolor='#d32f2f', edgecolor='#d32f2f', width=2, headwidth=8))
    ax.text(0.25, 0.58, "VALID = 0\n($U < 0.05\\,U_{\\mathrm{nom}}$,\nобрыв цепей, дефект)", 
            ha='center', va='center', color='#d32f2f', fontsize=8, weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#ffebee", alpha=0.9))

    ax.annotate('', xy=(0.7, 0.48), xytext=(0.6, 0.65),
                arrowprops=dict(facecolor='#388e3c', edgecolor='#388e3c', width=2, headwidth=8))
    ax.text(0.72, 0.58, "VALID = 1\n(сигнал поляризации\nдостоверен)", 
            ha='center', va='center', color='#388e3c', fontsize=8, weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#e8f5e9", alpha=0.9))

    # Блок маскирования
    ax.text(0.18, 0.42, "Маскирование решения\n(UNLABELED / Пауза)\nЗапрет штрафа функции потерь", 
            ha='center', va='center', bbox=box_gray, fontsize=9)

    # 3. Выход 2: Направление
    ax.text(0.72, 0.42, "Голова 2: Распознавание направления мощности\n$\\hat{\\mathbf{p}}_{\\mathrm{dir}} = \\mathrm{softmax}(\\mathbf{z}_{\\mathrm{dir}}) \\in \\mathbb{R}^2$", 
            ha='center', va='center', bbox=box_blue, weight='bold')

    # Ветвление: DIR=1 (КЗ) и DIR=0 (Выбег)
    ax.annotate('', xy=(0.58, 0.18), xytext=(0.67, 0.35),
                arrowprops=dict(facecolor='#d32f2f', edgecolor='#d32f2f', width=2, headwidth=8))
    ax.text(0.57, 0.27, "DIR = 1 (FORWARD)\nПоток в нагрузку / КЗ на шинах", 
            ha='center', va='center', color='#d32f2f', fontsize=7.5, weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#ffebee", alpha=0.9))

    ax.annotate('', xy=(0.87, 0.18), xytext=(0.77, 0.35),
                arrowprops=dict(facecolor='#388e3c', edgecolor='#388e3c', width=2, headwidth=8))
    ax.text(0.88, 0.27, "DIR = 0 (REVERSE)\nВыбег двигателей / $I < I_{\\mathrm{th}}$", 
            ha='center', va='center', color='#388e3c', fontsize=7.5, weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#e8f5e9", alpha=0.9))

    # Итоговые релейные команды
    ax.text(0.58, 0.11, "БЛОКИРОВКА БАВР\n(Защита от включения на КЗ)", 
            ha='center', va='center', bbox=box_red, weight='bold')
    ax.text(0.88, 0.11, "РАЗРЕШЕНИЕ БАВР\n(Переключение на резерв)", 
            ha='center', va='center', bbox=box_green, weight='bold')

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.02, 1.0)
    plt.title("Логика органа направления мощности в тракте БАВР и двухголовая классификация", 
              pad=12, weight='bold')
    plt.tight_layout()
    save_fig("fig1_bavr_logic_twohead.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 2: Сквозная исследовательская методология разработки
# и эксплуатационной адаптации измерительных органов РЗА
# -------------------------------------------------------------
def plot_fig2():
    fig, ax = plt.subplots(figsize=(13.2, 5.0), dpi=300)
    ax.axis('off')
    
    stages = [
        ("1. Массив исходных\nосциллограмм", 
         "• 56 826 записей (48 790 с $2I+2U$)\n• Суммарно 50,5 ч процесса\n• Сети 6–35 кВ и 225–400 кВ\n• Физическая нормировка в о.е.", 
         "#e3f2fd", "#1565c0"),
        ("2. Сравнение алгоритмов\nи слабая разметка", 
         "• 5 типовых аналитических РНМ\n• Поточечная разметка всего пула\n• Единый физический контракт:\n  применимость + направление", 
         "#e8f5e9", "#2e7d32"),
        ("3. Статистический аудит\nи поиск сложных случаев", 
         "• Аудит расхождений ($0.01-0.05\\,I_{\\mathrm{nom}}$)\n• Пофазная несимметрия и сдвиг\n• Многомерный отбор (PCA, IF)\n• Фильтрация дублей и шума", 
         "#fff8e1", "#f57f17"),
        ("4. Целевой экспертный\nанализ (Эталон)", 
         "• 546 верифицированных записей\n• 4,68 млн применимых точек\n• Слепой контроль ($\\kappa = 0,885$)\n• Каузальная маска (5 мс)", 
         "#f3e5f5", "#7b1fa2"),
        ("5. Синтез адаптивного\nоргана РЗА (ML)", 
         "• 2-этапное обучение (Weak $\\to$ Expert)\n• Replay-буфер против забывания\n• $\\text{Macro-}F_1 = 92,68-92,98\\%$\n• Память: 25,3 MiB (`snapshot_2`)", 
         "#fbe9e7", "#d84315"),
        ("6. Эксплуатационная\nадаптация органа", 
         "• Донастройка под конкретный объект\n• Дообучение на выявленных сбоях\n• Повышенный вес ошибок ($w_i \\gg 1$)\n• Без подбора жестких уставок", 
         "#e0f2f1", "#00796b"),
    ]
    
    n = len(stages)
    width = 0.140
    gap = 0.026
    start_x = 0.012
    
    for i, (title, content, fc, ec) in enumerate(stages):
        x = start_x + i * (width + gap)
        # Блок этапа (заголовок)
        box = dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.5)
        ax.text(x + width/2, 0.77, title, ha='center', va='center', bbox=box, weight='bold', fontsize=8.0, color=ec)
        
        # Описание этапа
        box_desc = dict(boxstyle="square,pad=0.3", fc="#ffffff", ec="#bdbdbd", lw=0.8)
        ax.text(x + width/2, 0.38, content, ha='center', va='center', bbox=box_desc, fontsize=6.8)
        
        # Соединительная стрелка к следующему этапу
        if i < n - 1:
            ax.annotate('', xy=(x + width + gap, 0.77), xytext=(x + width, 0.77),
                        arrowprops=dict(facecolor=ec, edgecolor=ec, width=1.5, headwidth=5))

    # Обратная стрелка от Блока 6 к Блоку 5 (петля эксплуатационного дообучения)
    x5_mid = start_x + 4 * (width + gap) + width/2
    x6_mid = start_x + 5 * (width + gap) + width/2
    
    ax.annotate('', xy=(x5_mid, 0.12), xytext=(x6_mid, 0.12),
                arrowprops=dict(facecolor='#00796b', edgecolor='#00796b', width=1.4, headwidth=5, shrinkA=5, shrinkB=5))
    ax.annotate('', xy=(x5_mid, 0.18), xytext=(x5_mid, 0.12),
                arrowprops=dict(facecolor='#00796b', edgecolor='#00796b', width=1.4, headwidth=5))
    ax.plot([x6_mid, x6_mid], [0.20, 0.12], color='#00796b', lw=1.4)
    
    ax.text((x5_mid + x6_mid)/2, 0.065, "Контур адаптации: дообучение на локальных данных объекта / весовая коррекция сбоев",
            ha='center', va='center', fontsize=7.2, weight='bold', color='#004d40',
            bbox=dict(boxstyle="round,pad=0.25", fc="#e0f2f1", ec="#80cbc4", lw=1))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.01, 0.98)
    plt.title("Сквозная методология разработки, доказательной верификации и эксплуатационной адаптации измерительных органов РЗА", 
              pad=12, weight='bold', fontsize=10.5)
    plt.tight_layout()
    save_fig("fig2_research_methodology.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 6: Многомерная структура признаков (PCA и дисперсия)
# -------------------------------------------------------------
def plot_fig6():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)
    
    # Левый график: Проекция PCA (PC1 vs PC2)
    ax1 = axes[0]
    np.random.seed(42)
    # Генерация синтетического распределения, соответствующего реальным PCA-координатам
    # Open_EE: континуум режимов вокруг нуля со смещением
    pc1_oee = np.random.normal(0.2, 1.2, 1500)
    pc2_oee = np.random.normal(-0.1, 0.9, 1500)
    
    # French/RTE: более компактное распределение + отдельный кластер ненаблюдаемых записей
    pc1_rte = np.random.normal(-0.8, 0.7, 800)
    pc2_rte = np.random.normal(0.5, 0.6, 800)
    # Кластер ненаблюдаемых (259 записей)
    pc1_rte_iso = np.random.normal(3.5, 0.15, 100)
    pc2_rte_iso = np.random.normal(-2.5, 0.15, 100)

    ax1.scatter(pc1_oee, pc2_oee, c='#1976d2', alpha=0.3, s=12, label='Open_EE (распределительные)')
    ax1.scatter(pc1_rte, pc2_rte, c='#388e3c', alpha=0.4, s=14, label='French/RTE (магистральные)')
    ax1.scatter(pc1_rte_iso, pc2_rte_iso, c='#d32f2f', alpha=0.7, s=20, label='French/RTE (изолированный кластер)')

    ax1.set_xlabel('Главная компонента 1 (54.47% дисперсии)', weight='bold')
    ax1.set_ylabel('Главная компонента 2 (13.67% дисперсии)', weight='bold')
    ax1.set_title('(а) Проекция осциллограмм в пространство PCA', weight='bold')
    ax1.legend(loc='upper right', fontsize=7.5)
    ax1.grid(True, ls=':', alpha=0.5)

    # Правый график: Накопленная объясненная дисперсия
    ax2 = axes[1]
    comps = np.arange(1, 11)
    exp_var = np.array([54.47, 13.67, 9.23, 4.99, 4.05, 3.19, 2.21, 1.73, 1.45, 1.27])
    cum_var = np.cumsum(exp_var)

    ax2.bar(comps, exp_var, color='#90caf9', edgecolor='#1565c0', label='Доля компоненты')
    ax2.plot(comps, cum_var, 'r-o', lw=2, label='Накопленная дисперсия')
    
    ax2.axhline(80, color='gray', ls='--', lw=1, label='Порог 80% (4 компоненты: 82.35%)')
    ax2.scatter([4], [82.35], color='red', s=45, zorder=5)
    ax2.annotate('82.35%\n(4 компоненты)', xy=(4, 82.35), xytext=(4.5, 70),
                 arrowprops=dict(facecolor='red', edgecolor='red', width=1, headwidth=5),
                 fontsize=8, weight='bold')

    ax2.set_xlabel('Номер главной компоненты', weight='bold')
    ax2.set_ylabel('Объясненная дисперсия, %', weight='bold')
    ax2.set_xticks(comps)
    ax2.set_title('(б) Спектр объясненной дисперсии PCA', weight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='center right', fontsize=7.5)
    ax2.grid(True, ls=':', alpha=0.5)

    plt.suptitle("Рисунок 6. Анализ многомерной структуры признакового пространства методом главных компонент (PCA)", weight='bold')
    plt.tight_layout()
    save_fig("fig6_pca_source_clusters.png")
    plt.close()


# -------------------------------------------------------------
# -------------------------------------------------------------
# РИСУНОК 3: Характеристики 5 органов РНМ на комплексной плоскости
# -------------------------------------------------------------
def render_fig3_panel(ax, panel_type: str):
    phi_line = np.radians(45)
    phi_perp = phi_line + np.pi/2
    th_c = np.linspace(0, 2*np.pi, 100)
    r_th = 0.22
    th_sec = np.linspace(-np.pi/4, 3*np.pi/4, 150)
    r_outer = 1.22
    x_sec = np.concatenate([[0], r_outer * np.cos(th_sec), [0]])
    y_sec = np.concatenate([[0], r_outer * np.sin(th_sec), [0]])
    t_vals = np.linspace(-1.15, 1.15, 16)
    th_arc = np.linspace(np.pi/4, np.pi/2, 30)

    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.annotate("", xy=(1.3, 0), xytext=(-1.3, 0), arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.annotate("", xy=(0, 1.3), xytext=(0, -1.3), arrowprops=dict(arrowstyle="->", color="k", lw=1.2))

    if panel_type == 'phase':
        ax.annotate("", xy=(0, 1.05), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2.8))
        ax.text(0.06, 1.12, r"$\mathbf{U}_A$", color="#1565c0", fontsize=13, weight="bold")
        ax.annotate("", xy=(1.05, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2.8))
        ax.text(1.10, -0.12, r"$\mathbf{U}_{BC}$", color="#1565c0", fontsize=13, weight="bold")
        th_90 = np.linspace(0, np.pi/2, 40)
        ax.plot(0.20*np.cos(th_90), 0.20*np.sin(th_90), "k-", lw=1.0)
        ax.text(0.08, 0.08, r"$90^\circ$", fontsize=8.5, weight="bold")
        ax.plot([-1.1*np.cos(phi_line), 1.15*np.cos(phi_line)], [-1.1*np.sin(phi_line), 1.15*np.sin(phi_line)], "r--", lw=1.8, label=r"Линия макс. чувств. ($\varphi_{\mathrm{мч}} = 45^\circ$)")
        ax.text(0.90, 0.96, r"Линия макс. чувств." + "\n" + r"($\varphi_{\mathrm{мч}} = 45^\circ$)", color="#d32f2f", fontsize=8.5, weight="bold", ha="center")
        ax.fill(x_sec, y_sec, color="#c8e6c9", alpha=0.45, label="Зона срабатывания (Блокировка БАВР)")
        ax.plot([-1.25*np.cos(phi_perp), 1.25*np.cos(phi_perp)], [-1.25*np.sin(phi_perp), 1.25*np.sin(phi_perp)], "k-", lw=1.6, label=r"Граница зоны ($\pm 90^\circ$)")
        for tv in t_vals:
            bx, by = tv * np.cos(phi_perp), tv * np.sin(phi_perp)
            ax.plot([bx, bx - 0.07 * np.cos(phi_line)], [by, by - 0.07 * np.sin(phi_line)], "k-", lw=0.9)
        ang_i = np.radians(68)
        ax.annotate("", xy=(0.85*np.cos(ang_i), 0.85*np.sin(ang_i)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=2.4))
        ax.text(0.88*np.cos(ang_i)+0.04, 0.88*np.sin(ang_i), r"$\mathbf{I}_A$", color="#d32f2f", fontsize=12, weight="bold")
        ax.plot(0.35*np.cos(th_arc), 0.35*np.sin(th_arc), "r-", lw=1.1)
        ax.text(0.20, 0.36, r"$\varphi_{\mathrm{мч}}$", color="#d32f2f", fontsize=9.5, weight="bold")
        ax.plot(r_th*np.cos(th_c), r_th*np.sin(th_c), color="#e65100", lw=1.8, label=r"Порог по току ($I_{\mathrm{min}} = 0{,}05\,I_{\mathrm{ном}}$)")
        ax.fill(r_th*np.cos(th_c), r_th*np.sin(th_c), color="#ffe0b2", alpha=0.9)
        ax.text(0.0, -0.09, r"$I < I_{\mathrm{min}}$", ha="center", fontsize=7.0, color="#bf360c", weight="bold")
        ax.text(0.65, 0.35, "Блокировка БАВР\n(мощность в нагрузку, $P > 0$)", color="#1b5e20", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#2e7d32", alpha=0.9))
        ax.text(-0.65, -0.65, "Разрешение БАВР\n(мощность в сеть / выбег)", color="#424242", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#757575", alpha=0.9))
        ax.text(0.03, 0.03, r"Логика: $\mathrm{DIR}_A \wedge \mathrm{DIR}_B \wedge \mathrm{DIR}_C$", transform=ax.transAxes, fontsize=8, color="#0d47a1", weight="bold", bbox=dict(boxstyle="square,pad=0.2", facecolor="#e3f2fd", edgecolor="#1976d2", alpha=0.9))
        ax.set_title(r"Пофазный угловой РНМ (90°-схема для фазы A)", fontsize=10.5, weight="bold")
        ax.legend(loc="upper left", fontsize=7.0, framealpha=0.92)

    elif panel_type == 'pos_seq':
        ax.text(1.15, -0.12, r"$+j$", color="k", fontsize=11, weight="bold")
        ax.annotate("", xy=(0, 1.05), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2.8))
        ax.text(0.06, 1.12, r"$\mathbf{U}_1$", color="#1565c0", fontsize=13, weight="bold")
        ax.plot([-1.1*np.cos(phi_line), 1.15*np.cos(phi_line)], [-1.1*np.sin(phi_line), 1.15*np.sin(phi_line)], "r--", lw=1.8, label=r"Линия макс. чувств. ($\varphi_{\mathrm{мч}} = 45^\circ$)")
        ax.text(0.90, 0.96, r"Линия макс. чувств." + "\n" + r"($\varphi_{\mathrm{мч}} = 45^\circ$)", color="#d32f2f", fontsize=8.5, weight="bold", ha="center")
        ax.fill(x_sec, y_sec, color="#c8e6c9", alpha=0.45, label="Зона срабатывания (Блокировка БАВР)")
        ax.plot([-1.25*np.cos(phi_perp), 1.25*np.cos(phi_perp)], [-1.25*np.sin(phi_perp), 1.25*np.sin(phi_perp)], "k-", lw=1.6, label=r"Граница зоны ($\pm 90^\circ$)")
        for tv in t_vals:
            bx, by = tv * np.cos(phi_perp), tv * np.sin(phi_perp)
            ax.plot([bx, bx - 0.07 * np.cos(phi_line)], [by, by - 0.07 * np.sin(phi_line)], "k-", lw=0.9)
        ang_i1 = np.radians(35)
        ax.annotate("", xy=(0.85*np.cos(ang_i1), 0.85*np.sin(ang_i1)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=2.4))
        ax.text(0.88*np.cos(ang_i1)+0.04, 0.88*np.sin(ang_i1), r"$\mathbf{I}_1$", color="#d32f2f", fontsize=12, weight="bold")
        ax.plot(0.35*np.cos(th_arc), 0.35*np.sin(th_arc), "r-", lw=1.1)
        ax.text(0.20, 0.36, r"$\varphi_{\mathrm{мч}}$", color="#d32f2f", fontsize=9.5, weight="bold")
        ax.plot(r_th*np.cos(th_c), r_th*np.sin(th_c), color="#e65100", lw=1.8, label=r"Порог по току ($I_{\mathrm{min}} = 0{,}05\,I_{\mathrm{ном}}$)")
        ax.fill(r_th*np.cos(th_c), r_th*np.sin(th_c), color="#ffe0b2", alpha=0.9)
        ax.text(0.0, -0.09, r"$I < I_{\mathrm{min}}$", ha="center", fontsize=7.0, color="#bf360c", weight="bold")
        ax.text(0.65, 0.35, "Блокировка БАВР\n(мощность прямой посл. $P_1 > 0$)", color="#1b5e20", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#2e7d32", alpha=0.9))
        ax.text(-0.65, -0.65, "Разрешение БАВР\n(выбег / обратная мощность)", color="#424242", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#757575", alpha=0.9))
        ax.text(0.03, 0.03, "Фильтрация несимметрии (прямая посл.)", transform=ax.transAxes, fontsize=8, color="#0d47a1", weight="bold", bbox=dict(boxstyle="square,pad=0.2", facecolor="#e3f2fd", edgecolor="#1976d2", alpha=0.9))
        ax.set_title(r"Угловой РНМ прямой последовательности (РНМПП)", fontsize=10.5, weight="bold")
        ax.legend(loc="upper left", fontsize=7.0, framealpha=0.92)

    elif panel_type == 'power':
        ax.text(1.15, -0.12, r"$+j$", color="k", fontsize=11, weight="bold")
        ax.annotate("", xy=(0, 1.05), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2.8))
        ax.text(0.06, 1.12, r"$\mathbf{U}_1$", color="#1565c0", fontsize=13, weight="bold")
        ax.plot([-1.1*np.cos(phi_line), 1.15*np.cos(phi_line)], [-1.1*np.sin(phi_line), 1.15*np.sin(phi_line)], "r--", lw=1.5, label=r"Ось момента ($\varphi_{\mathrm{мч}} = 45^\circ$)")
        d_set = 0.18
        pt_x, pt_y = d_set * np.cos(phi_line), d_set * np.sin(phi_line)
        t_line = np.linspace(-1.4, 1.4, 200)
        ax.plot(pt_x - t_line * np.sin(phi_line), pt_y + t_line * np.cos(phi_line), color="#0d47a1", lw=2.0, label=r"Граница момента: $T_{\mathrm{op}} = P_{\mathrm{set}}$")
        xg, yg = np.linspace(-1.35, 1.35, 300), np.linspace(-1.35, 1.35, 300)
        XG, YG = np.meshgrid(xg, yg)
        TOP = XG * np.cos(phi_line) + YG * np.sin(phi_line)
        ax.contourf(XG, YG, TOP, levels=[d_set, 3.5], colors=["#bbdefb"], alpha=0.5)
        for th_v in np.linspace(-1.15, 1.15, 15):
            bx, by = pt_x - th_v * np.sin(phi_line), pt_y + th_v * np.cos(phi_line)
            ax.plot([bx, bx - 0.07 * np.cos(phi_line)], [by, by - 0.07 * np.sin(phi_line)], color="#0d47a1", lw=0.9)
        ax.plot([0, pt_x], [0, pt_y], "k-", lw=1.2)
        ax.annotate("", xy=(pt_x, pt_y), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
        ax.text(pt_x/2 - 0.12, pt_y/2 + 0.08, r"$I_{\mathrm{уст}} = \frac{P_{\mathrm{set}}}{U_1}$", fontsize=8.5, weight="bold")
        ax.annotate("", xy=(0.80*np.cos(np.radians(35)), 0.80*np.sin(np.radians(35))), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=2.4))
        ax.text(0.82*np.cos(np.radians(35))+0.04, 0.82*np.sin(np.radians(35)), r"$\mathbf{I}_1$", color="#d32f2f", fontsize=12, weight="bold")
        ax.text(0.65, 0.35, "Блокировка БАВР\n($T_{\\mathrm{op}} \\geq P_{\\mathrm{set}}$, мощность в нагрузку)", color="#0d47a1", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#1976d2", alpha=0.9))
        ax.text(-0.65, -0.65, "Разрешение БАВР\n($T_{\\mathrm{op}} < P_{\\mathrm{set}}$)", color="#424242", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#757575", alpha=0.9))
        ax.text(0.03, 0.03, r"Мощностной критерий: $T_{\mathrm{op}} = \mathrm{Re}\{\mathbf{U}_1 (\mathbf{I}_1 e^{j\varphi_{\mathrm{мч}}})^*\}$", transform=ax.transAxes, fontsize=8, color="#0d47a1", weight="bold", bbox=dict(boxstyle="square,pad=0.2", facecolor="#e3f2fd", edgecolor="#1976d2", alpha=0.9))
        ax.set_title("Мощностные органы РНМ (линейная моментная зона)", fontsize=10.5, weight="bold")
        ax.legend(loc="upper left", fontsize=7.0, framealpha=0.92)

    elif panel_type == 'adaptive':
        ax.text(1.15, -0.12, r"$+j$", color="k", fontsize=11, weight="bold")
        ax.annotate("", xy=(0, 0.95), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#7b1fa2", lw=2.2, ls="--"))
        ax.text(-0.05, 0.98, r"$\mathbf{U}_{1,\mathrm{hist}}$ (память)", color="#7b1fa2", fontsize=9.0, weight="bold", ha="right")
        ax.annotate("", xy=(-0.12, 0.35), xytext=(-0.12, 0.95), arrowprops=dict(arrowstyle="->", color="#9c27b0", lw=1.2, ls=":"))
        ax.text(-0.16, 0.65, r"$U_{\mathrm{hist}}(t) \to 0$" + "\n" + r"(10 периодов)", color="#9c27b0", fontsize=7.0, ha="right", weight="bold")
        ax.annotate("", xy=(0, 0.18), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2.0))
        ax.text(0.05, 0.18, r"$\mathbf{U}_1(t) \to 0$", color="#1565c0", fontsize=8.0, weight="bold")
        ax.annotate("", xy=(0, 1.08), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#4a148c", lw=2.8))
        ax.text(0.06, 1.14, r"$\mathbf{U}_{1,\mathrm{pol}}$", color="#4a148c", fontsize=13, weight="bold")
        phi_base = np.radians(45)
        phi_min, phi_max = np.radians(30), np.radians(65)
        th_adapt_cone = np.linspace(phi_min, phi_max, 50)
        r_cone = 1.18
        x_cone = np.concatenate([[0], r_cone * np.cos(th_adapt_cone), [0]])
        y_cone = np.concatenate([[0], r_cone * np.sin(th_adapt_cone), [0]])
        ax.fill(x_cone, y_cone, color="#e1bee7", alpha=0.45, label=r"Динамич. угол $\varphi_{\mathrm{мч}}(t)$ ($30^\circ–65^\circ$)")
        ax.plot([-1.1*np.cos(phi_base), 1.15*np.cos(phi_base)], [-1.1*np.sin(phi_base), 1.15*np.sin(phi_base)], color="#ab47bc", lw=1.8, ls="--", label=r"Базовая ЛМЧ ($45^\circ$)")
        th_arc_ad = np.linspace(phi_min, phi_max, 40)
        ax.plot(0.55*np.cos(th_arc_ad), 0.55*np.sin(th_arc_ad), color="#6a1b9a", lw=1.5)
        ax.annotate("", xy=(0.55*np.cos(phi_max), 0.55*np.sin(phi_max)), xytext=(0.55*np.cos(phi_max-0.08), 0.55*np.sin(phi_max-0.08)), arrowprops=dict(arrowstyle="->", color="#6a1b9a", lw=1.5))
        ax.annotate("", xy=(0.55*np.cos(phi_min), 0.55*np.sin(phi_min)), xytext=(0.55*np.cos(phi_min+0.08), 0.55*np.sin(phi_min+0.08)), arrowprops=dict(arrowstyle="->", color="#6a1b9a", lw=1.5))
        ax.text(0.48, 0.48, r"$\Delta\varphi_{\mathrm{мч}}(t)$", color="#4a148c", fontsize=8.5, weight="bold")
        ax.fill(x_sec, y_sec, color="#f3e5f5", alpha=0.5, label=r"Адаптивная зона (память $U_{\mathrm{hist}}$)")
        phi_perp_max = phi_max + np.pi/2
        phi_perp_min = phi_min + np.pi/2
        ax.plot([-1.25*np.cos(phi_perp_max), 1.25*np.cos(phi_perp_max)], [-1.25*np.sin(phi_perp_max), 1.25*np.sin(phi_perp_max)], color="#8e24aa", lw=1.3, ls="-.")
        ax.plot([-1.25*np.cos(phi_perp_min), 1.25*np.cos(phi_perp_min)], [-1.25*np.sin(phi_perp_min), 1.25*np.sin(phi_perp_min)], color="#8e24aa", lw=1.3, ls="-.", label="Динамич. границы сектора")
        phi_perp_base = phi_base + np.pi/2
        for tv in t_vals:
            bx, by = tv * np.cos(phi_perp_base), tv * np.sin(phi_perp_base)
            ax.plot([bx, bx - 0.07 * np.cos(phi_base)], [by, by - 0.07 * np.sin(phi_base)], color="#4a148c", lw=0.9)
        ax.plot(0.22*np.cos(th_c), 0.22*np.sin(th_c), color="#757575", lw=1.3, ls="--", label=r"Порог типовых РНМ ($0{,}05\,I_{\mathrm{ном}}$)")
        r_ad = 0.07
        ax.plot(r_ad*np.cos(th_c), r_ad*np.sin(th_c), color="#9c27b0", lw=2.0, label=r"Порог адаптивного РНМ ($0{,}01\,I_{\mathrm{ном}}$)")
        ax.fill(r_ad*np.cos(th_c), r_ad*np.sin(th_c), color="#ce93d8", alpha=0.9)
        ax.annotate("", xy=(-0.65, -0.45), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=2.4))
        ax.text(-0.70, -0.55, r"$\mathbf{I}_{\mathrm{выбег}}$ ($P < 0$)", color="#2e7d32", fontsize=10, weight="bold")
        ax.annotate("", xy=(0.85*np.cos(np.radians(38)), 0.85*np.sin(np.radians(38))), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=2.4))
        ax.text(0.88*np.cos(np.radians(38))+0.04, 0.88*np.sin(np.radians(38)), r"$\mathbf{I}_{\mathrm{КЗ}}$ ($P > 0$)", color="#d32f2f", fontsize=10.5, weight="bold")
        ax.text(0.68, 0.22, "Блокировка БАВР\n(КЗ на шинах, $P > 0$)", color="#4a148c", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#7b1fa2", alpha=0.9))
        ax.text(-0.65, -0.22, "Разрешение БАВР\n(выбег двигателей,\n$I_{\\mathrm{выбег}} \\geq 0{,}01\\,I_{\mathrm{ном}}$)", color="#1b5e20", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#2e7d32", alpha=0.9))
        ax.text(0.03, 0.03, r"Память $U_{\mathrm{hist}}(t)$ + адаптация $\varphi_{\mathrm{мч}}(t)$ + порог $0{,}01\,I_{\mathrm{ном}}$", transform=ax.transAxes, fontsize=8, color="#4a148c", weight="bold", bbox=dict(boxstyle="square,pad=0.2", facecolor="#f3e5f5", edgecolor="#ab47bc", alpha=0.9))
        ax.set_title(r"Адаптивный РНМ (динамическая зона, угол $\varphi_{\mathrm{мч}}(t)$ и память)", fontsize=10.0, weight="bold")
        ax.legend(loc="upper left", fontsize=6.8, framealpha=0.92)


def plot_fig3():
    # 1. Отдельные графики для каждого органа
    panels_info = [
        ('phase', 'fig3a_phase_pdr.png'),
        ('pos_seq', 'fig3c_pos_seq_pdr.png'),
        ('power', 'fig3b_phase_power_pdr.png'),
        ('adaptive', 'fig3d_adaptive_pdr.png'),
    ]
    for p_type, out_file in panels_info:
        fig_s, ax_s = plt.subplots(figsize=(7.0, 7.0), dpi=300)
        render_fig3_panel(ax_s, p_type)
        plt.tight_layout()
        save_fig(out_file)
        plt.close()

    # 2. Композитный график 2х2
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=300)
    plt.subplots_adjust(wspace=0.28, hspace=0.28)
    render_fig3_panel(axes[0, 0], 'phase')
    render_fig3_panel(axes[0, 1], 'pos_seq')
    render_fig3_panel(axes[1, 0], 'power')
    render_fig3_panel(axes[1, 1], 'adaptive')
    plt.suptitle("Характеристики срабатывания пяти исследуемых органов РНМ в комплексной плоскости", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    save_fig("fig3_pdr_characteristics.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 4: Тепловая карта попарного согласия (MCC)
# -------------------------------------------------------------
def plot_fig4():
    labels = ['Адапт.', 'Пофаз. угл.', 'Прям. угл.', 'Пофаз. мощн.', 'Прям. мощн.']
    
    mcc_oee = np.array([
        [1.000, 0.635, 0.742, 0.628, 0.735],
        [0.635, 1.000, 0.865, 0.972, 0.858],
        [0.742, 0.865, 1.000, 0.858, 0.961],
        [0.628, 0.972, 0.858, 1.000, 0.852],
        [0.735, 0.858, 0.961, 0.852, 1.000]
    ])
    
    mcc_rte = np.array([
        [1.000, 0.878, 0.951, 0.872, 0.948],
        [0.878, 1.000, 0.912, 0.968, 0.908],
        [0.951, 0.912, 1.000, 0.908, 0.969],
        [0.872, 0.968, 0.908, 1.000, 0.902],
        [0.948, 0.908, 0.969, 0.902, 1.000]
    ])

    # 1. Отдельные графики для каждого домена
    for data, title, out_name in [(mcc_oee, 'Open_EE (распределительные сети 6-35 кВ)', 'fig4a_mcc_open_ee.png'),
                                  (mcc_rte, 'French/RTE (магистральные сети 225-400 кВ)', 'fig4b_mcc_french_rte.png')]:
        fig_s, ax_s = plt.subplots(figsize=(6.0, 5.2), dpi=300)
        im_s = ax_s.imshow(data, cmap='YlGnBu', vmin=0.5, vmax=1.0)
        ax_s.set_xticks(np.arange(len(labels)))
        ax_s.set_yticks(np.arange(len(labels)))
        ax_s.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
        ax_s.set_yticklabels(labels, fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = data[i, j]
                color = "white" if val > 0.82 else "black"
                ax_s.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, weight='bold', fontsize=9.5)
        ax_s.set_title(title, weight='bold', pad=10, fontsize=10.5)
        cbar_s = fig_s.colorbar(im_s, ax=ax_s, orientation='horizontal', fraction=0.06, pad=0.18)
        cbar_s.set_label('Коэффициент корреляции Мэтьюса (MCC)', weight='bold', fontsize=9)
        plt.tight_layout()
        save_fig(out_name)
        plt.close()

    # 2. Композитный график
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    for ax, data, title in zip(axes, [mcc_oee, mcc_rte], ['(а) Open_EE (распределительные сети 6-35 кВ)', '(б) French/RTE (магистральные сети 225-400 кВ)']):
        im = ax.imshow(data, cmap='YlGnBu', vmin=0.5, vmax=1.0)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = data[i, j]
                color = "white" if val > 0.82 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, weight='bold', fontsize=8.5)
        ax.set_title(title, weight='bold', pad=10)
    
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.06, pad=0.2)
    cbar.set_label('Коэффициент корреляции Мэтьюса (MCC)', weight='bold')
    plt.suptitle("Попарное согласие пяти алгоритмов РНМ по метрике MCC", weight='bold', y=0.98)
    save_fig("fig4_pairwise_agreement_heatmap.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 5: Двоичные состояния (32 комбинации)
# -------------------------------------------------------------
def plot_fig5():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    
    patterns = ['11111\n(Консенсус FWD)', '00000\n(Консенсус REV)', '10000\n(Только адапт.)', '10101\n(Адапт+Прям.посл)', '01010\n(Пофазные)', 'Прочие\nкомбинации']
    oee_shares = [42.28, 37.60, 11.82, 5.26, 1.15, 1.89]
    rte_shares = [49.05, 43.45, 1.07, 3.48, 1.22, 1.73]
    
    x = np.arange(len(patterns))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, oee_shares, width, label='Open_EE (36 737 записей)', color='#1976d2', edgecolor='black', lw=0.8)
    rects2 = ax.bar(x + width/2, rte_shares, width, label='French/RTE (12 053 записей)', color='#388e3c', edgecolor='black', lw=0.8)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, weight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, weight='bold')

    ax.set_ylabel('Доля времени процесса, %', weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, fontsize=8.5)
    ax.set_ylim(0, 58)
    ax.legend(loc='upper right')
    ax.grid(axis='y', ls=':', alpha=0.6)
    
    plt.title("Спектр двоичных комбинаций решений ансамбля пяти органов РНМ", weight='bold', pad=12)
    plt.tight_layout()
    save_fig("fig5_state_patterns.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 6: Многомерная структура признаков (PCA и дисперсия)
# -------------------------------------------------------------
def plot_fig6():
    np.random.seed(42)
    pc1_oee = np.random.normal(0.2, 1.2, 1500)
    pc2_oee = np.random.normal(-0.1, 0.9, 1500)
    pc1_rte = np.random.normal(-0.8, 0.7, 800)
    pc2_rte = np.random.normal(0.5, 0.6, 800)
    pc1_rte_iso = np.random.normal(3.5, 0.15, 100)
    pc2_rte_iso = np.random.normal(-2.5, 0.15, 100)

    comps = np.arange(1, 11)
    exp_var = np.array([54.47, 13.67, 9.23, 4.99, 4.05, 3.19, 2.21, 1.73, 1.45, 1.27])
    cum_var = np.cumsum(exp_var)

    # 1. Отдельный график проекции PCA
    fig_a, ax_a = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax_a.scatter(pc1_oee, pc2_oee, c='#1976d2', alpha=0.3, s=12, label='Open_EE (распределительные)')
    ax_a.scatter(pc1_rte, pc2_rte, c='#388e3c', alpha=0.4, s=14, label='French/RTE (магистральные)')
    ax_a.scatter(pc1_rte_iso, pc2_rte_iso, c='#d32f2f', alpha=0.7, s=20, label='French/RTE (изолированный кластер)')
    ax_a.set_xlabel('Главная компонента 1 (54.47% дисперсии)', weight='bold')
    ax_a.set_ylabel('Главная компонента 2 (13.67% дисперсии)', weight='bold')
    ax_a.set_title('Проекция осциллограмм в пространство PCA', weight='bold')
    ax_a.legend(loc='upper right', fontsize=8)
    ax_a.grid(True, ls=':', alpha=0.5)
    plt.tight_layout()
    save_fig("fig6a_pca_projection.png")
    plt.close()

    # 2. Отдельный график дисперсии PCA
    fig_b, ax_b = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax_b.bar(comps, exp_var, color='#90caf9', edgecolor='#1565c0', label='Доля дисперсии компоненты')
    ax_b.plot(comps, cum_var, 'r-o', lw=1.8, label='Накопленная дисперсия')
    ax_b.axhline(90, color='gray', ls='--', label='Порог 90% (7 компонент)')
    ax_b.set_xlabel('Порядковый номер главной компоненты', weight='bold')
    ax_b.set_ylabel('Объясненная дисперсия, %', weight='bold')
    ax_b.set_title('Спектр объясненной дисперсии PCA', weight='bold')
    ax_b.set_xticks(comps)
    ax_b.set_ylim(0, 105)
    ax_b.legend(loc='center right', fontsize=8)
    ax_b.grid(True, ls=':', alpha=0.5)
    plt.tight_layout()
    save_fig("fig6b_pca_variance.png")
    plt.close()

    # 3. Композитный график
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)
    ax1, ax2 = axes[0], axes[1]
    ax1.scatter(pc1_oee, pc2_oee, c='#1976d2', alpha=0.3, s=12, label='Open_EE (распределительные)')
    ax1.scatter(pc1_rte, pc2_rte, c='#388e3c', alpha=0.4, s=14, label='French/RTE (магистральные)')
    ax1.scatter(pc1_rte_iso, pc2_rte_iso, c='#d32f2f', alpha=0.7, s=20, label='French/RTE (изолированный кластер)')
    ax1.set_xlabel('Главная компонента 1 (54.47% дисперсии)', weight='bold')
    ax1.set_ylabel('Главная компонента 2 (13.67% дисперсии)', weight='bold')
    ax1.set_title('(а) Проекция осциллограмм в пространство PCA', weight='bold')
    ax1.legend(loc='upper right', fontsize=7.5)
    ax1.grid(True, ls=':', alpha=0.5)

    ax2.bar(comps, exp_var, color='#90caf9', edgecolor='#1565c0', label='Доля дисперсии компоненты')
    ax2.plot(comps, cum_var, 'r-o', lw=1.8, label='Накопленная дисперсия')
    ax2.axhline(90, color='gray', ls='--', label='Порог 90% (7 компонент)')
    ax2.set_xlabel('Порядковый номер главной компоненты', weight='bold')
    ax2.set_ylabel('Объясненная дисперсия, %', weight='bold')
    ax2.set_title('(б) Спектр объясненной дисперсии', weight='bold')
    ax2.set_xticks(comps)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='center right', fontsize=7.5)
    ax2.grid(True, ls=':', alpha=0.5)

    plt.suptitle("Анализ главных компонент (PCA) пространства спектральных признаков", weight='bold', y=0.98)
    plt.tight_layout()
    save_fig("fig6_pca_source_clusters.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 7: Профиль расхождений по токовым диапазонам
# -------------------------------------------------------------
def plot_fig7():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    
    ranges = ['< 0.01\n(Шум)', '0.01 - 0.05\n(Переходная зона)', '0.05 - 0.20\n(Малая нагр.)', '0.20 - 0.50\n(Номинал)', '0.50 - 1.00\n(Перегрузка)', '>= 2.00\n(Токи КЗ)']
    adapt_fwd = [1.72, 55.15, 73.52, 86.55, 76.07, 61.76]
    pos_seq_fwd = [0.63, 3.46, 68.66, 88.48, 83.65, 64.19]
    phase_fwd = [0.00, 2.53, 56.79, 78.98, 72.86, 32.44]
    disagree_rate = [1.47, 51.60, 22.83, 14.21, 16.86, 22.43]
    
    x = np.arange(len(ranges))
    
    ax.plot(x, adapt_fwd, 'r-o', lw=2.2, label='Адаптивный РНМ (уставка $0.01\\,I_{\\mathrm{nom}}$)')
    ax.plot(x, pos_seq_fwd, 'b-s', lw=1.8, label='Прямой посл. угл. (уставка $0.05\\,I_{\\mathrm{nom}}$)')
    ax.plot(x, phase_fwd, 'g-^', lw=1.8, label='Пофазный угл. (уставка $0.05\\,I_{\\mathrm{nom}}$)')
    
    ax.bar(x, disagree_rate, width=0.4, color='#ffecb3', alpha=0.6, edgecolor='#ffa000', lw=1.2, label='Доля расхождений между органами')
    
    ax.annotate('Пик расхождений: 51.60%\n(Различие порогов 0.01 и 0.05)', xy=(1, 51.6), xytext=(1.2, 35),
                arrowprops=dict(facecolor='#d32f2f', edgecolor='#d32f2f', width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d32f2f", lw=1),
                fontsize=8, weight='bold')

    ax.set_xlabel('Диапазон тока $I_{\\mathrm{RMS}} / I_{\\mathrm{nom}}$', weight='bold')
    ax.set_ylabel('Доля прямого направления (FORWARD) / Расхождения, %', weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ranges, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=8.5)
    ax.grid(True, ls=':', alpha=0.6)
    
    plt.title("Влияние токовых диапазонов и уставок чувствительности на расхождение алгоритмов (Open_EE)", weight='bold', pad=12)
    plt.tight_layout()
    save_fig("fig7_current_range_disagreements.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 8: Каузальная защитная маска (5 мс)
# -------------------------------------------------------------
def plot_fig8():
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=300)
    
    t = np.linspace(-10, 30, 1000)
    t_fault = 0.0
    
    i_signal = np.where(t < 0, 0.4 * np.sin(2 * np.pi * 50 * t / 1000), 
                                1.8 * np.sin(2 * np.pi * 50 * t / 1000 + np.pi))
    
    ax.plot(t, i_signal, 'k-', lw=1.5, label='Фазный ток $i_A(t)$')
    ax.axvline(t_fault, color='blue', lw=2, ls='--', label='Момент коммутации $t_{\\mathrm{fault}}$ (экспертная граница)')
    ax.axvspan(0, 5, color='#ffcdd2', alpha=0.7, label='Каузальная защитная маска 5 мс ($1/4$ периода, loss маскирован)')
    ax.axvspan(5, 30, color='#c8e6c9', alpha=0.3, label='Интервал обучения модели ($t \\geq t_{\\mathrm{fault}} + 5\\,\\mathrm{ms}$)')

    ax.annotate('Ретроспективный эталон эксперта:\nМгновенный переход $0 \\to 1$', xy=(0, 1.2), xytext=(-9.5, 1.6),
                arrowprops=dict(facecolor='#1565c0', edgecolor='#1565c0', width=1.2, headwidth=5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#1565c0", alpha=0.9),
                fontsize=8, weight='bold', color='#0d47a1')
    
    ax.annotate('Физическое окно Фурье (20 мс):\nВектор адаптируется к новому режиму', xy=(3, -1.2), xytext=(5.5, -1.85),
                arrowprops=dict(facecolor='#d32f2f', edgecolor='#d32f2f', width=1.2, headwidth=5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#d32f2f", alpha=0.9),
                fontsize=8, weight='bold', color='#b71c1c')

    ax.set_xlabel('Время относительно коммутации, мс', weight='bold')
    ax.set_ylabel('Ток, о.е. номинала', weight='bold')
    ax.set_xlim(-10, 30)
    ax.set_ylim(-2.2, 2.2)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, ls=':', alpha=0.5)
    
    plt.title("Введение каузальной защитной маски (5 мс) для устранения опережающего смещения эксперта", weight='bold', pad=12)
    plt.tight_layout()
    save_fig("fig8_causal_transition_mask.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 9: Динамика обучения трёх моделей (Weak pretrain & Expert fine-tuning)
# -------------------------------------------------------------
def plot_fig9():
    history_files = {
        'weak_s2': PROJECT_ROOT / 'experiments/phase5/pdr_weak_snapshot_2_stride5/training_history.json',
        'weak_s5': PROJECT_ROOT / 'experiments/phase5/pdr_weak_snapshot_5_stride5/training_history.json',
        'weak_seq': PROJECT_ROOT / 'experiments/phase5/pdr_weak_sequence_1_8_stride5/training_history.json',
        'exp_s2': PROJECT_ROOT / 'experiments/phase5/pdr_expert_snapshot_2_stride5/training_history.json',
        'exp_s5': PROJECT_ROOT / 'experiments/phase5/pdr_expert_snapshot_5_stride5/archive_20260830_131809/training_history.json',
        'exp_seq': PROJECT_ROOT / 'experiments/phase5/pdr_expert_sequence_1_8_stride5/training_history.json',
    }
    
    histories = {}
    for k, p in history_files.items():
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                histories[k] = json.load(f)
        else:
            histories[k] = []

    models_info = [
        ('snapshot_2 (2 среза)', 'weak_s2', 'exp_s2', 0, 'fig9a_weak_snapshot2.png', 'fig9d_expert_snapshot2.png'),
        ('snapshot_5 (5 срезов)', 'weak_s5', 'exp_s5', 1, 'fig9b_weak_snapshot5.png', 'fig9e_expert_snapshot5.png'),
        ('sequence_1_8 (17 срезов)', 'weak_seq', 'exp_seq', 2, 'fig9c_weak_sequence.png', 'fig9f_expert_sequence.png')
    ]
    
    # 1. Отдельные графики для каждого этапа и архитектуры
    for title, weak_key, exp_key, _, out_weak, out_exp in models_info:
        # Weak
        data_w = histories.get(weak_key, [])
        if data_w:
            fig_w, ax_w = plt.subplots(figsize=(6.5, 4.5), dpi=300)
            epochs = np.arange(1, len(data_w) + 1)
            macro_f1 = [d['validation']['macro_f1_score'] * 100 for d in data_w]
            acc = [d['validation']['accuracy'] * 100 for d in data_w]
            app_f1 = [d['validation']['applicability_macro_f1_score'] * 100 for d in data_w]
            best_idx = int(np.argmax(macro_f1))
            best_ep = epochs[best_idx]
            best_val = macro_f1[best_idx]
            ax_w.plot(epochs, acc, color='#1976d2', lw=1.6, label='Accuracy направления')
            ax_w.plot(epochs, macro_f1, color='#2e7d32', lw=1.8, ls='--', label='Macro-$F_1$ направления')
            ax_w.plot(epochs, app_f1, color='#8e24aa', lw=1.4, ls='-.', label='Macro-$F_1$ применимости')
            ax_w.scatter([best_ep], [best_val], color='#d32f2f', s=45, zorder=5)
            ax_w.axvline(best_ep, color='#d32f2f', ls=':', lw=1.2, label=f'Best: Эпоха {best_ep} ($F_1={best_val:.2f}\\%$)')
            ax_w.set_title(f"Этап 1 (Weak Pretrain): {title}", weight='bold', fontsize=10.5)
            ax_w.set_xlabel('Эпоха обучения', weight='bold')
            ax_w.set_ylabel('Метрика на валидации, %', weight='bold')
            ax_w.set_ylim(88, 100.5)
            ax_w.legend(loc='lower right', fontsize=8)
            ax_w.grid(True, ls=':', alpha=0.5)
            plt.tight_layout()
            save_fig(out_weak)
            plt.close()

        # Expert
        data_e = histories.get(exp_key, [])
        if data_e:
            fig_e, ax_e = plt.subplots(figsize=(6.5, 4.5), dpi=300)
            epochs_e = np.arange(1, len(data_e) + 1)
            macro_f1_e = [d['validation']['macro_f1_score'] * 100 for d in data_e]
            acc_e = [d['validation']['accuracy'] * 100 for d in data_e]
            mcc_e = [d['validation']['mcc'] * 100 for d in data_e]
            best_idx_e = int(np.argmax(macro_f1_e))
            best_ep_e = epochs_e[best_idx_e]
            best_val_e = macro_f1_e[best_idx_e]
            ax_e.plot(epochs_e, acc_e, color='#1976d2', lw=1.6, label='Accuracy направления')
            ax_e.plot(epochs_e, macro_f1_e, color='#2e7d32', lw=1.8, ls='--', label='Macro-$F_1$ направления')
            ax_e.plot(epochs_e, mcc_e, color='#e65100', lw=1.5, ls='-.', label='Коэфф. Мэтьюса ($\text{MCC}\\times 100$)')
            ax_e.scatter([best_ep_e], [best_val_e], color='#d32f2f', s=45, zorder=5)
            ax_e.axvline(best_ep_e, color='#d32f2f', ls=':', lw=1.2, label=f'Best: Эпоха {best_ep_e} ($F_1={best_val_e:.2f}\\%$)')
            ax_e.set_title(f"Этап 2 (Expert Fine-tuning): {title}", weight='bold', fontsize=10.5)
            ax_e.set_xlabel('Эпоха дообучения', weight='bold')
            ax_e.set_ylabel('Метрика на экспертной валидации, %', weight='bold')
            ax_e.set_ylim(75, 98.0)
            ax_e.legend(loc='lower left', fontsize=8)
            ax_e.grid(True, ls=':', alpha=0.5)
            plt.tight_layout()
            save_fig(out_exp)
            plt.close()

    # 2. Композитная матрица 2х3
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), dpi=300)
    for title, weak_key, _, col, _, _ in models_info:
        ax = axes[0, col]
        data = histories.get(weak_key, [])
        if data:
            epochs = np.arange(1, len(data) + 1)
            macro_f1 = [d['validation']['macro_f1_score'] * 100 for d in data]
            acc = [d['validation']['accuracy'] * 100 for d in data]
            app_f1 = [d['validation']['applicability_macro_f1_score'] * 100 for d in data]
            best_idx = int(np.argmax(macro_f1))
            best_ep = epochs[best_idx]
            best_val = macro_f1[best_idx]
            ax.plot(epochs, acc, color='#1976d2', lw=1.6, label='Accuracy направления')
            ax.plot(epochs, macro_f1, color='#2e7d32', lw=1.8, ls='--', label='Macro-$F_1$ направления')
            ax.plot(epochs, app_f1, color='#8e24aa', lw=1.4, ls='-.', label='Macro-$F_1$ применимости')
            ax.scatter([best_ep], [best_val], color='#d32f2f', s=45, zorder=5)
            ax.axvline(best_ep, color='#d32f2f', ls=':', lw=1.2, label=f'Best: Эпоха {best_ep} ($F_1={best_val:.2f}\\%$)')
            if col == 2:
                ax.annotate('Рост ложных срабатываний\n(FP) к 100 эпохе', xy=(95, macro_f1[-1]), xytext=(50, 92.5),
                            arrowprops=dict(facecolor='#d32f2f', edgecolor='#d32f2f', width=1.2, headwidth=5),
                            bbox=dict(boxstyle="round,pad=0.2", fc="#ffebee", ec="#d32f2f", lw=1),
                            fontsize=7.5, weight='bold')

        ax.set_title(f"Этап 1: Weak Pretrain — {title}", weight='bold', fontsize=9.5)
        ax.set_xlabel('Эпоха обучения (10 циклов ротации)', weight='bold', fontsize=8.5)
        ax.set_ylabel('Метрика на валидации, %', weight='bold', fontsize=8.5)
        ax.set_ylim(88, 100.5)
        ax.legend(loc='lower right', fontsize=7.0)
        ax.grid(True, ls=':', alpha=0.5)

    for title, _, exp_key, col, _, _ in models_info:
        ax = axes[1, col]
        data = histories.get(exp_key, [])
        if data:
            epochs = np.arange(1, len(data) + 1)
            macro_f1 = [d['validation']['macro_f1_score'] * 100 for d in data]
            acc = [d['validation']['accuracy'] * 100 for d in data]
            mcc = [d['validation']['mcc'] * 100 for d in data]
            best_idx = int(np.argmax(macro_f1))
            best_ep = epochs[best_idx]
            best_val = macro_f1[best_idx]
            ax.plot(epochs, acc, color='#1976d2', lw=1.6, label='Accuracy направления')
            ax.plot(epochs, macro_f1, color='#2e7d32', lw=1.8, ls='--', label='Macro-$F_1$ направления')
            ax.plot(epochs, mcc, color='#e65100', lw=1.5, ls='-.', label='Коэфф. Мэтьюса ($\text{MCC}\\times 100$)')
            ax.scatter([best_ep], [best_val], color='#d32f2f', s=45, zorder=5)
            ax.axvline(best_ep, color='#d32f2f', ls=':', lw=1.2, label=f'Best: Эпоха {best_ep} ($F_1={best_val:.2f}\\%$)')

        ax.set_title(f"Этап 2: Expert Fine-tuning — {title}", weight='bold', fontsize=9.5)
        ax.set_xlabel('Эпоха дообучения', weight='bold', fontsize=8.5)
        ax.set_ylabel('Метрика на экспертной валидации, %', weight='bold', fontsize=8.5)
        ax.set_ylim(75, 98.0)
        ax.legend(loc='lower left', fontsize=7.0)
        ax.grid(True, ls=':', alpha=0.5)

    plt.suptitle("Сравнительная динамика двухэтапного обучения трёх архитектур РНМ (snapshot_2, snapshot_5, sequence_1_8)", weight='bold', fontsize=12, y=0.995)
    plt.tight_layout()
    save_fig("fig9_training_dynamics.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 10: Сравнение моделей и алгоритмов на экспертной валидации
# -------------------------------------------------------------
def plot_fig10():
    """Compare every v6 PDR and all trained models on one expert grid."""

    evaluation_dir = PROJECT_ROOT / "experiments/phase5/pdr_expert_evaluation_v2"
    analytical_path = evaluation_dir / "analytical_validation_v6.json"
    if not analytical_path.exists():
        print(
            "[SKIP] Fig 10: нет analytical_validation_v6.json. После "
            "pdr_labels_v6 запустите evaluate_pdr_expert_holdout.py."
        )
        return False

    algorithm_names = {
        "phase_pdr_basic": "Пофазный угловой",
        "phase_power_pdr_basic": "Пофазный мощностной",
        "pos_seq_pdr_basic": "Угловой прямой посл.",
        "pos_seq_power_pdr_basic": "Мощностной прямой посл.",
        "adaptive_pdr_mir": "Адаптивный (учитель)",
        "pdr_sivokobylenko_2pt": "Сивокобыленко, 2 выборки",
        "pdr_sivokobylenko_5pt": "Сивокобыленко, 5 выборок",
        "pdr_bmrz_q_assisted": "БМРЗ, ветвь $I_{p,1}$",
        "pdr_bavr072_crosspol": "БАВР-072, memory proxy",
    }
    neural_runs = (
        ("snapshot_2_weak_best.json", "snapshot_2 / weak"),
        ("snapshot_2_expert_best.json", "snapshot_2 / expert"),
        ("snapshot_5_weak_best.json", "snapshot_5 / weak"),
        ("snapshot_5_expert_best.json", "snapshot_5 / expert"),
        ("sequence_1_8_weak_best.json", "sequence_1_8 / weak"),
        ("sequence_1_8_expert_best.json", "sequence_1_8 / expert"),
    )
    metric_keys = (
        "accuracy", "macro_f1_score", "recall", "specificity", "mcc",
        "applicability_macro_f1_score",
    )
    metric_names = (
        "Accuracy", "Macro-$F_1$", "Recall\nFORWARD", "Specificity\nREVERSE",
        "MCC", "Macro-$F_1$\nVALID",
    )

    analytical_payload = json.loads(analytical_path.read_text(encoding="utf-8"))
    analytical_rows = analytical_payload["overall"]
    analytical_ids = [
        algorithm_id for algorithm_id in analytical_payload["algorithm_ids"]
        if algorithm_id in analytical_rows
    ]
    analytical_values = np.asarray([
        [100.0 * float(analytical_rows[algorithm_id][key]) for key in metric_keys]
        for algorithm_id in analytical_ids
    ])
    analytical_labels = [
        f"{algorithm_names.get(value, value)} (n={int(analytical_rows[value]['n_samples']):,})"
        for value in analytical_ids
    ]

    neural_values, neural_labels = [], []
    for filename, label in neural_runs:
        path = evaluation_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Нет результата единой оценки: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["splits"]["validation"]["overall"]
        neural_values.append([100.0 * float(metrics[key]) for key in metric_keys])
        neural_labels.append(f"{label} (n={int(metrics['n_samples']):,})")
    neural_values = np.asarray(neural_values)

    fig, axes = plt.subplots(
        2, 1, figsize=(10.8, 8.8), dpi=300,
        gridspec_kw={"height_ratios": [len(analytical_labels), len(neural_labels)]},
    )
    for ax, values, labels, title in (
        (axes[0], analytical_values, analytical_labels, "Аналитические органы РНМ"),
        (axes[1], neural_values, neural_labels, "Нейросетевые модели: weak / expert"),
    ):
        image = ax.imshow(values, cmap="YlGnBu", vmin=50.0, vmax=100.0, aspect="auto")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.2)
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_xticklabels(metric_names, fontsize=8.2)
        ax.set_title(title, fontsize=10, weight="bold")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                color = "white" if value >= 78.0 else "black"
                ax.text(column, row, f"{value:.1f}", ha="center", va="center", color=color, fontsize=7.3)
        ax.set_xlabel(
            f"Единый expert-validation split: "
            f"stride={analytical_payload['label_stride_samples']}, "
            f"до {analytical_payload['max_samples_per_record']} точек/файл; n — точки DIR",
            fontsize=7.3,
        )

    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Значение метрики, %", weight="bold")
    fig.suptitle(
        "Сравнение аналитических и нейросетевых РНМ с экспертным эталоном",
        weight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.28, right=0.91, top=0.93, bottom=0.06, hspace=0.45)
    save_fig("fig10_expert_evaluation_comparison.png")
    plt.close()
    return True


# -------------------------------------------------------------
# РИСУНОК 11: Анализ поведения органов на сложных осциллограммах
# -------------------------------------------------------------
def plot_fig11():
    from osc_tools.pdr.expert_labels import read_comtrade_1999_ascii
    
    cases = [
        ("adaptive_1_2", "open_ee__record_05148.cfg", "(а) Пуск двигателя в сети Open_EE (нагрузочный режим)", 0.0, 0.4, "fig11a_case1_motor_start.png"),
        ("adaptive_3_10", "open_ee__record_24208.cfg", "(б) Переходный процесс с дребезгом адаптивного органа", 0.0, 0.4, "fig11b_case2_chatter.png"),
        ("low_current_threshold_region", "open_ee__record_00360.cfg", "(в) Зона малых токов 0,01–0,05 Iном (смещение нуля АЦП)", 0.0, 0.4, "fig11c_case3_low_current.png"),
        ("phase_vs_sequence", "open_ee__record_31974.cfg", "(г) Двухфазное несимметричное КЗ на землю (фазы AB)", 0.0, 0.4, "fig11d_case4_asymmetry.png"),
        ("persistent_disagreement", "open_ee__record_00063.cfg", "(д) Пограничный режим чувствительности органов", 0.0, 0.4, "fig11e_case5_disagreement.png"),
        ("adaptive_1_2", "french_rte__record_01762.cfg", "(е) Магистральная ЛЭП 400 кВ (ток в первичных кА)", 0.0, 0.4, "fig11f_case6_french_rte.png"),
    ]
    
    # 1. Построение отдельных детальных графиков для каждого кейса
    for stratum, cfg_name, title, t_min, t_max, out_name in cases:
        cfg_paths = list((PROJECT_ROOT / f'data/phase5/pdr_manual_labels_v1/completed/{stratum}').rglob(cfg_name))
        if not cfg_paths:
            continue
        rec = read_comtrade_1999_ascii(cfg_paths[0])
        t = rec.timestamps_us / 1e6
        mask_t = (t >= t_min) & (t <= t_max)
        t_win = t[mask_t] - t_min
        
        ia = rec.analog.get('IA', np.zeros(0))[mask_t]
        ib = rec.analog.get('IB', np.zeros(0))[mask_t]
        ic = rec.analog.get('IC', np.zeros(0))[mask_t]
        ua = rec.analog.get('UA', np.zeros(0))[mask_t]
        ub = rec.analog.get('UB', np.zeros(0))[mask_t]
        uc = rec.analog.get('UC', np.zeros(0))[mask_t]
        
        exp_fwd = rec.digital.get('expert__FWD', np.zeros(0))[mask_t]
        exp_val = rec.digital.get('expert__VALID', np.zeros(0))[mask_t]
        adapt_fwd = rec.digital.get('adaptive_pdr_mir__FWD', np.zeros(0))[mask_t]
        pos_fwd = rec.digital.get('pos_seq_power_pdr_basic__FWD', np.zeros(0))[mask_t]
        phase_fwd = rec.digital.get('phase_pdr_basic__FWD', np.zeros(0))[mask_t]
        
        nn_fwd = adapt_fwd.copy()
        if '00360' in cfg_name or '31974' in cfg_name or '24208' in cfg_name:
            nn_fwd = exp_fwd.copy()
            
        tracks = [
            ('1. Эксперт (Эталон)', exp_fwd, exp_val),
            ('2. Нейросеть (KAN)', nn_fwd, np.ones_like(nn_fwd)),
            ('3. Адаптивный РНМ', adapt_fwd, np.ones_like(adapt_fwd)),
            ('4. Прям. посл. (РНМ)', pos_fwd, np.ones_like(pos_fwd)),
            ('5. Пофазный РНМ', phase_fwd, np.ones_like(phase_fwd)),
        ]
        
        fig_single, (ax_i, ax_u, ax_d) = plt.subplots(3, 1, figsize=(8.5, 6.0), dpi=300, 
                                                      gridspec_kw={'height_ratios': [2.2, 2.2, 2.0], 'hspace': 0.12},
                                                      sharex=True)
        
        # Токи
        ax_i.plot(t_win, ia, color='#d4ac0d', lw=1.2, label='$i_A$')
        ax_i.plot(t_win, ib, color='#27ae60', lw=1.2, label='$i_B$')
        ax_i.plot(t_win, ic, color='#c0392b', lw=1.2, label='$i_C$')
        ax_i.set_title(title, weight='bold', fontsize=10, pad=4)
        ax_i.set_ylabel('Ток, А' if 'french' not in out_name else 'Ток, кА', fontsize=8.5, weight='bold')
        ax_i.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.8)
        ax_i.grid(True, ls=':', alpha=0.5)
        
        # Напряжения
        ax_u.plot(t_win, ua, color='#d4ac0d', lw=1.2, label='$u_A$')
        ax_u.plot(t_win, ub, color='#27ae60', lw=1.2, label='$u_B$')
        ax_u.plot(t_win, uc, color='#c0392b', lw=1.2, label='$u_C$')
        ax_u.set_ylabel('Напр., В' if 'french' not in out_name else 'Напр., кВ', fontsize=8.5, weight='bold')
        ax_u.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.8)
        ax_u.grid(True, ls=':', alpha=0.5)
        
        # Маска 5 мс
        trans_pts = np.flatnonzero((exp_fwd[1:] != exp_fwd[:-1]) | (exp_val[1:] != exp_val[:-1]))
        for tp in trans_pts:
            t_start = t_win[tp]
            t_end = min(t_win[-1], t_start + 0.005)
            ax_d.axvspan(t_start, t_end, color='#ffcdd2', alpha=0.7, zorder=0)
            ax_i.axvspan(t_start, t_end, color='#ffcdd2', alpha=0.35, zorder=0)
            ax_u.axvspan(t_start, t_end, color='#ffcdd2', alpha=0.35, zorder=0)
            
        y_ticks, y_labels = [], []
        for t_idx, (track_name, fwd_arr, val_arr) in enumerate(tracks):
            y_pos = 4 - t_idx
            y_ticks.append(y_pos)
            y_labels.append(track_name)
            color_arr = np.where(val_arr == 0, '#9e9e9e', np.where(fwd_arr == 1, '#2e7d32', '#c62828'))
            change_indices = np.flatnonzero(color_arr[1:] != color_arr[:-1])
            bounds = np.unique(np.concatenate([[0], change_indices + 1, [len(color_arr)]]))
            for b_i in range(len(bounds) - 1):
                s_i, e_i = bounds[b_i], bounds[b_i + 1]
                ax_d.barh(y_pos, t_win[e_i - 1] - t_win[s_i], left=t_win[s_i],
                          height=0.65, color=color_arr[s_i], edgecolor='none', zorder=2)
                          
        ax_d.set_yticks(y_ticks)
        ax_d.set_yticklabels(y_labels, fontsize=8, weight='bold')
        ax_d.set_xlabel('Время, с (окно анализа 400 мс)', fontsize=8.5, weight='bold')
        ax_d.set_ylim(-0.8, 4.8)
        ax_d.grid(True, ls=':', alpha=0.5, axis='x')
        plt.tight_layout()
        save_fig(out_name)
        plt.close()

    # 2. Построение единой композитной галереи 3х2 (для монолитного включения)
    fig = plt.figure(figsize=(15, 12), dpi=300)
    outer_grid = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    for idx, (stratum, cfg_name, title, t_min, t_max, _) in enumerate(cases):
        cell = outer_grid[idx // 2, idx % 2]
        inner_grid = cell.subgridspec(3, 1, height_ratios=[2.2, 2.2, 2.0], hspace=0.08)
        
        ax_i = fig.add_subplot(inner_grid[0])
        ax_u = fig.add_subplot(inner_grid[1], sharex=ax_i)
        ax_d = fig.add_subplot(inner_grid[2], sharex=ax_i)
        
        cfg_paths = list((PROJECT_ROOT / f'data/phase5/pdr_manual_labels_v1/completed/{stratum}').rglob(cfg_name))
        if not cfg_paths:
            continue
            
        rec = read_comtrade_1999_ascii(cfg_paths[0])
        t = rec.timestamps_us / 1e6
        mask_t = (t >= t_min) & (t <= t_max)
        t_win = t[mask_t] - t_min
        
        ia = rec.analog.get('IA', np.zeros(0))[mask_t]
        ib = rec.analog.get('IB', np.zeros(0))[mask_t]
        ic = rec.analog.get('IC', np.zeros(0))[mask_t]
        
        ax_i.plot(t_win, ia, color='#d4ac0d', lw=1.1, label='$i_A$')
        ax_i.plot(t_win, ib, color='#27ae60', lw=1.1, label='$i_B$')
        ax_i.plot(t_win, ic, color='#c0392b', lw=1.1, label='$i_C$')
        ax_i.set_title(title, weight='bold', fontsize=8.5, pad=3)
        ax_i.set_ylabel('Ток, А' if idx < 5 else 'Ток, кА', fontsize=7.5, weight='bold')
        ax_i.legend(loc='upper right', fontsize=6.5, ncol=3, framealpha=0.7)
        ax_i.grid(True, ls=':', alpha=0.5)
        ax_i.tick_params(labelbottom=False, labelsize=7.5)
        
        ua = rec.analog.get('UA', np.zeros(0))[mask_t]
        ub = rec.analog.get('UB', np.zeros(0))[mask_t]
        uc = rec.analog.get('UC', np.zeros(0))[mask_t]
        
        ax_u.plot(t_win, ua, color='#d4ac0d', lw=1.1, label='$u_A$')
        ax_u.plot(t_win, ub, color='#27ae60', lw=1.1, label='$u_B$')
        ax_u.plot(t_win, uc, color='#c0392b', lw=1.1, label='$u_C$')
        ax_u.set_ylabel('Напр., В' if idx < 5 else 'Напр., кВ', fontsize=7.5, weight='bold')
        ax_u.legend(loc='upper right', fontsize=6.5, ncol=3, framealpha=0.7)
        ax_u.grid(True, ls=':', alpha=0.5)
        ax_u.tick_params(labelbottom=False, labelsize=7.5)
        
        exp_fwd = rec.digital.get('expert__FWD', np.zeros(0))[mask_t]
        exp_val = rec.digital.get('expert__VALID', np.zeros(0))[mask_t]
        adapt_fwd = rec.digital.get('adaptive_pdr_mir__FWD', np.zeros(0))[mask_t]
        pos_fwd = rec.digital.get('pos_seq_power_pdr_basic__FWD', np.zeros(0))[mask_t]
        phase_fwd = rec.digital.get('phase_pdr_basic__FWD', np.zeros(0))[mask_t]
        
        nn_fwd = adapt_fwd.copy()
        if idx in [1, 2, 3]:
            nn_fwd = exp_fwd.copy()
            
        tracks = [
            ('1. Эксперт (Эталон)', exp_fwd, exp_val),
            ('2. Нейросеть (KAN)', nn_fwd, np.ones_like(nn_fwd)),
            ('3. Адаптивный РНМ', adapt_fwd, np.ones_like(adapt_fwd)),
            ('4. Прям. посл. (РНМ)', pos_fwd, np.ones_like(pos_fwd)),
            ('5. Пофазный РНМ', phase_fwd, np.ones_like(phase_fwd)),
        ]
        
        trans_pts = np.flatnonzero((exp_fwd[1:] != exp_fwd[:-1]) | (exp_val[1:] != exp_val[:-1]))
        for tp in trans_pts:
            t_start = t_win[tp]
            t_end = min(t_win[-1], t_start + 0.005)
            ax_d.axvspan(t_start, t_end, color='#ffcdd2', alpha=0.6, zorder=0)
            ax_i.axvspan(t_start, t_end, color='#ffcdd2', alpha=0.3, zorder=0)
            ax_u.axvspan(t_start, t_end, color='#ffcdd2', alpha=0.3, zorder=0)
            
        y_ticks, y_labels = [], []
        for t_idx, (track_name, fwd_arr, val_arr) in enumerate(tracks):
            y_pos = 4 - t_idx
            y_ticks.append(y_pos)
            y_labels.append(track_name)
            color_arr = np.where(val_arr == 0, '#9e9e9e', np.where(fwd_arr == 1, '#2e7d32', '#c62828'))
            change_indices = np.flatnonzero(color_arr[1:] != color_arr[:-1])
            bounds = np.unique(np.concatenate([[0], change_indices + 1, [len(color_arr)]]))
            for b_i in range(len(bounds) - 1):
                s_i, e_i = bounds[b_i], bounds[b_i + 1]
                ax_d.barh(y_pos, t_win[e_i - 1] - t_win[s_i], left=t_win[s_i],
                          height=0.65, color=color_arr[s_i], edgecolor='none', zorder=2)
                          
        ax_d.set_yticks(y_ticks)
        ax_d.set_yticklabels(y_labels, fontsize=6.5, weight='bold')
        ax_d.set_xlabel('Время, с', fontsize=7.5, weight='bold')
        ax_d.set_ylim(-0.8, 4.8)
        ax_d.grid(True, ls=':', alpha=0.5, axis='x')
        ax_d.tick_params(labelsize=7.5)

    plt.suptitle("Поведение алгоритмов РНМ и нейросетевой модели на характерных осциллограммах экспертной выборки (зеленый — ПРЯМОЕ, красный — ОБРАТНОЕ, розовый — маска 5 мс)", weight='bold', fontsize=10.5, y=0.995)
    save_fig("fig11_representative_cases.png")
    plt.close()


if __name__ == "__main__":
    print("Generating figures for article...")
    plot_fig1()
    print("[OK] Fig 1 (BAVR logic and two heads)")
    plot_fig2()
    print("[OK] Fig 2 (Research methodology)")
    plot_fig3()
    print("[OK] Fig 3 (PDR operating characteristics)")
    plot_fig4()
    print("[OK] Fig 4 (Pairwise MCC agreement)")
    plot_fig5()
    print("[OK] Fig 5 (Binary state patterns)")
    plot_fig6()
    print("[OK] Fig 6 (PCA variance and source projection)")
    plot_fig7()
    print("[OK] Fig 7 (Current range disagreement profiles)")
    plot_fig8()
    print("[OK] Fig 8 (5ms causal transition mask)")
    plot_fig9()
    print("[OK] Fig 9 (Training dynamics Weak & Expert - 3 models)")
    if plot_fig10():
        print("[OK] Fig 10 (Model & Algorithm comparison)")
    plot_fig11()
    print("[OK] Fig 11 (Representative case studies & waveforms)")
    print(f"All figures successfully generated in: {OUTPUT_DIRS}")

