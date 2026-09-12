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
    ax.text(0.25, 0.58, "VALID = 0\n(направление\nнеприменимо)",
            ha='center', va='center', color='#d32f2f', fontsize=8, weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#ffebee", alpha=0.9))

    ax.annotate('', xy=(0.7, 0.48), xytext=(0.6, 0.65),
                arrowprops=dict(facecolor='#388e3c', edgecolor='#388e3c', width=2, headwidth=8))
    ax.text(0.72, 0.58, "VALID = 1\n(сигнал поляризации\nдостоверен)", 
            ha='center', va='center', color='#388e3c', fontsize=8, weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#e8f5e9", alpha=0.9))

    # Блок маскирования
    ax.text(0.18, 0.42, "Нет решения направления\nОбучение VALID сохраняется\nDIR-loss: маска эталона",
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
    ax.text(0.88, 0.11, "УСЛОВИЕ ДЛЯ БАВР\n(не команда переключения)",
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
         "• 9 аналитических РНМ\n• Поточечная разметка всего пула\n• Единый физический контракт:\n  применимость + направление",
         "#e8f5e9", "#2e7d32"),
        ("3. Статистический аудит\nи поиск сложных случаев", 
         "• Аудит расхождений ($0.01-0.05\\,I_{\\mathrm{nom}}$)\n• Пофазная несимметрия и сдвиг\n• Многомерный отбор (PCA, IF)\n• Фильтрация дублей и шума", 
         "#fff8e1", "#f57f17"),
        ("4. Целевой экспертный\nанализ (Эталон)", 
         "• 830 рассмотренных записей\n• 6,806 млн применимых точек\n• 200 повторов ($\\kappa = 0,886$)\n• Каузальная маска (5 мс)",
         "#f3e5f5", "#7b1fa2"),
        ("5. Синтез адаптивного\nоргана РЗА (ML)", 
         "• Два этапа обучения\n• Буфер обычных режимов\n• F1 DIR: 92,73–97,22%\n• Контроль независимости",
         "#fbe9e7", "#d84315"),
        ("6. Дальнейшее развитие:\nадаптация под объект",
         "• Донастройка под конкретный объект\n• Дообучение на выявленных сбоях\n• Повышенный вес ошибок ($w_i \\gg 1$)\n• Отдельная проверка после адаптации",
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
        ax.text(-0.65, -0.22, "Разрешение БАВР\n(выбег двигателей,\n$I_{\\mathrm{выбег}} \\geq 0{,}01\\,I_{\\mathrm{ном}}$)", color="#1b5e20", fontsize=8.0, weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#2e7d32", alpha=0.9))
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
    plt.suptitle("Базовые семейства характеристик РНМ в комплексной плоскости", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    save_fig("fig3_pdr_characteristics.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 4: Тепловая карта попарного согласия (MCC)
# -------------------------------------------------------------
def plot_fig4():
    analysis_dir = PROJECT_ROOT / "data" / "phase5" / "pdr_analysis_v6"
    pairwise = pd.read_csv(analysis_dir / "pairwise_pointwise_agreement.csv")
    algorithm_ids = [
        "adaptive_pdr_mir", "phase_pdr_basic", "pos_seq_pdr_basic",
        "phase_power_pdr_basic", "pos_seq_power_pdr_basic",
        "pdr_sivokobylenko_2pt", "pdr_sivokobylenko_5pt",
        "pdr_bmrz_q_assisted", "pdr_bavr072_crosspol",
    ]
    labels = [
        "Адапт.", "Пофаз.\nугл.", "Прям.\nугл.", "Пофаз.\nмощн.",
        "Прям.\nмощн.", "Сивок.\n2 т.", "Сивок.\n5 т.", "БМРЗ", "БАВР-072",
    ]

    def matrix_for(source):
        matrix = np.eye(len(algorithm_ids), dtype=float)
        index = {name: i for i, name in enumerate(algorithm_ids)}
        for row in pairwise[pairwise["source"] == source].itertuples(index=False):
            if row.left in index and row.right in index:
                i, j = index[row.left], index[row.right]
                matrix[i, j] = matrix[j, i] = float(row.mcc)
        return matrix

    mcc_oee = matrix_for("open_ee")
    mcc_rte = matrix_for("french_rte")

    # 1. Отдельные графики для каждого домена
    for data, title, out_name in [(mcc_oee, 'Open_EE (распределительные сети 6-35 кВ)', 'fig4a_mcc_open_ee.png'),
                                  (mcc_rte, 'French/RTE (магистральные сети 225-400 кВ)', 'fig4b_mcc_french_rte.png')]:
        fig_s, ax_s = plt.subplots(figsize=(8.0, 6.8), dpi=300)
        im_s = ax_s.imshow(data, cmap='YlGnBu', vmin=0.5, vmax=1.0)
        ax_s.set_xticks(np.arange(len(labels)))
        ax_s.set_yticks(np.arange(len(labels)))
        ax_s.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax_s.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = data[i, j]
                color = "white" if val > 0.82 else "black"
                ax_s.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, weight='bold', fontsize=6.5)
        ax_s.set_title(title, weight='bold', pad=10, fontsize=10.5)
        cbar_s = fig_s.colorbar(im_s, ax=ax_s, orientation='horizontal', fraction=0.06, pad=0.18)
        cbar_s.set_label('Коэффициент корреляции Мэтьюса (MCC)', weight='bold', fontsize=9)
        plt.tight_layout()
        save_fig(out_name)
        plt.close()

    # 2. Композитный график
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), dpi=300)
    for ax, data, title in zip(axes, [mcc_oee, mcc_rte], ['(а) Open_EE (распределительные сети 6-35 кВ)', '(б) French/RTE (магистральные сети 225-400 кВ)']):
        im = ax.imshow(data, cmap='YlGnBu', vmin=0.5, vmax=1.0)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = data[i, j]
                color = "white" if val > 0.82 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, weight='bold', fontsize=5.8)
        ax.set_title(title, weight='bold', pad=10)
    
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.06, pad=0.2)
    cbar.set_label('Коэффициент корреляции Мэтьюса (MCC)', weight='bold')
    plt.suptitle("Попарное согласие девяти алгоритмов РНМ по метрике MCC", weight='bold', y=0.98)
    plt.subplots_adjust(bottom=0.28, wspace=0.25)
    save_fig("fig4_pairwise_agreement_heatmap.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 5: Двоичные состояния ансамбля девяти органов
# -------------------------------------------------------------
def plot_fig5():
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    table = pd.read_csv(
        PROJECT_ROOT / "data" / "phase5" / "pdr_analysis_v6" /
        "algorithm_state_patterns.csv", dtype={"state_pattern": str}
    )
    selected_patterns = [
        "111111111", "000000000", "100000000", "101011111",
        "000001000", "000000010",
    ]
    patterns = [
        "111111111\n(консенсус FWD)", "000000000\n(консенсус REV)",
        "100000000\n(только адапт.)", "101011111\n(кроме пофазных)",
        "000001000\n(только 2-точ.)", "000000010\n(только БМРЗ)",
        "Прочие\nкомбинации",
    ]

    def shares_for(source):
        source_rows = table[table["source"] == source].set_index("state_pattern")
        values = [
            100.0 * float(source_rows.loc[pattern, "point_fraction"])
            if pattern in source_rows.index else 0.0
            for pattern in selected_patterns
        ]
        values.append(max(0.0, 100.0 - sum(values)))
        return values

    oee_shares = shares_for("open_ee")
    rte_shares = shares_for("french_rte")
    
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
    ax.set_ylim(0, max(oee_shares + rte_shares) * 1.25)
    ax.legend(loc='upper right')
    ax.grid(axis='y', ls=':', alpha=0.6)
    
    plt.title("Доминирующие комбинации решений ансамбля девяти органов РНМ", weight='bold', pad=12)
    plt.tight_layout()
    save_fig("fig5_state_patterns.png")
    plt.close()


# -------------------------------------------------------------
# РИСУНОК 6: Многомерная структура признаков (PCA и дисперсия)
# -------------------------------------------------------------
def plot_fig6():
    analysis_dir = PROJECT_ROOT / "data" / "phase5" / "pdr_analysis_v6"
    coordinates = pd.read_csv(
        analysis_dir / "research_pca_coordinates.csv",
        usecols=["source", "PC1", "PC2"],
    )
    sampled = []
    for _, group in coordinates.groupby("source"):
        sampled.append(group.sample(n=min(5000, len(group)), random_state=42))
    coordinates = pd.concat(sampled, ignore_index=True)
    oee = coordinates[coordinates["source"] == "open_ee"]
    rte = coordinates[coordinates["source"] == "french_rte"]
    pc1_oee, pc2_oee = oee["PC1"].to_numpy(), oee["PC2"].to_numpy()
    pc1_rte, pc2_rte = rte["PC1"].to_numpy(), rte["PC2"].to_numpy()

    variance = pd.read_csv(analysis_dir / "research_pca_explained_variance.csv").head(10)
    comps = variance["component"].to_numpy(dtype=int)
    exp_var = 100.0 * variance["explained_variance_ratio"].to_numpy(dtype=float)
    cum_var = 100.0 * variance["cumulative_explained_variance"].to_numpy(dtype=float)
    first_pct, second_pct = exp_var[0], exp_var[1]
    threshold_component = int(comps[np.flatnonzero(cum_var >= 90.0)[0]])

    # 1. Отдельный график проекции PCA
    fig_a, ax_a = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax_a.scatter(pc1_oee, pc2_oee, c='#1976d2', alpha=0.3, s=12, label='Open_EE (распределительные)')
    ax_a.scatter(pc1_rte, pc2_rte, c='#388e3c', alpha=0.4, s=14, label='French/RTE (магистральные)')
    ax_a.set_xlabel(f'Главная компонента 1 ({first_pct:.2f}% дисперсии)', weight='bold')
    ax_a.set_ylabel(f'Главная компонента 2 ({second_pct:.2f}% дисперсии)', weight='bold')
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
    ax_b.axhline(90, color='gray', ls='--', label=f'Порог 90% ({threshold_component} компонент)')
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
    ax1.set_xlabel(f'Главная компонента 1 ({first_pct:.2f}% дисперсии)', weight='bold')
    ax1.set_ylabel(f'Главная компонента 2 ({second_pct:.2f}% дисперсии)', weight='bold')
    ax1.set_title('(а) Проекция осциллограмм в пространство PCA', weight='bold')
    ax1.legend(loc='upper right', fontsize=7.5)
    ax1.grid(True, ls=':', alpha=0.5)

    ax2.bar(comps, exp_var, color='#90caf9', edgecolor='#1565c0', label='Доля дисперсии компоненты')
    ax2.plot(comps, cum_var, 'r-o', lw=1.8, label='Накопленная дисперсия')
    ax2.axhline(90, color='gray', ls='--', label=f'Порог 90% ({threshold_component} компонент)')
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
    """Пофайловые токовые профили из текущего расчёта, не ручные константы."""
    data = pd.read_csv(PROJECT_ROOT / "data/phase5/pdr_analysis_v6/review_current_profiles.csv")
    bins = ["<0.01", "0.01-0.05", "0.05-0.20", "0.20-0.50", "0.50-1.00", "1.00-2.00", ">=2.00"]
    # Имена интервалов берём из таблицы, порядок — по нижней границе.
    def lower_bound(value):
        return -1.0 if str(value).startswith("<") else float(str(value).replace(">=", "").split("-")[0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for ax, source in zip(axes, ("open_ee", "french_rte")):
        rows = data[data.source == source].copy()
        rows["_sort"] = rows.current_bin_physical_pu.map(lower_bound)
        rows = rows.sort_values("_sort")
        x = np.arange(len(rows))
        for algorithm, name in (
            ("adaptive_pdr_mir", "Адаптивный"),
            ("pos_seq_pdr_basic", "Угловой прямой посл."),
            ("phase_pdr_basic", "Пофазный угловой"),
        ):
            ax.plot(x, 100 * rows[algorithm + "__mean_forward_fraction"], "o-", ms=4, label=name)
        ax.bar(x, 100 * rows.mean_disagreement_fraction, color="#ffd881", alpha=.5,
               label="Средняя доля расхождений")
        ax.set_xticks(x, [str(b) + "\nN=" + str(n) for b, n in zip(rows.current_bin_physical_pu, rows.records)],
                      rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(source)
        ax.set_xlabel("Пофайловый ток RMS / номинальный ток")
        ax.set_ylabel("Средняя по осциллограммам доля, %")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Токовые профили: доли решений и расхождений девяти РНМ")
    save_fig("fig7_current_range_disagreements.png")
    plt.close(fig)


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
        'exp_s5': PROJECT_ROOT / 'experiments/phase5/pdr_expert_snapshot_5_stride5/training_history.json',
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
        ('sequence_1_8', 'weak_seq', 'exp_seq', 2, 'fig9c_weak_sequence.png', 'fig9f_expert_sequence.png')
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
            best_idx = int(np.argmax([d["selection_score"] for d in data_w]))
            best_ep = epochs[best_idx]
            best_val = macro_f1[best_idx]
            ax_w.plot(epochs, acc, color='#1976d2', lw=1.6, label='Accuracy направления')
            ax_w.plot(epochs, macro_f1, color='#2e7d32', lw=1.8, ls='--', label='Macro-$F_1$ направления')
            ax_w.plot(epochs, app_f1, color='#8e24aa', lw=1.4, ls='-.', label='Macro-$F_1$ применимости')
            ax_w.scatter([best_ep], [best_val], color='#d32f2f', s=45, zorder=5)
            ax_w.axvline(best_ep, color='#d32f2f', ls=':', lw=1.2, label=f'Best: Эпоха {best_ep} (Macro-F1={best_val:.2f}%)')
            ax_w.set_title(f"Этап 1 (Автоматическая разметка): {title}", weight='bold', fontsize=10.5)
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
            best_idx_e = int(np.argmax([d["selection_score"] for d in data_e]))
            best_ep_e = epochs_e[best_idx_e]
            best_val_e = macro_f1_e[best_idx_e]
            ax_e.plot(epochs_e, acc_e, color='#1976d2', lw=1.6, label='Accuracy направления')
            ax_e.plot(epochs_e, macro_f1_e, color='#2e7d32', lw=1.8, ls='--', label='Macro-$F_1$ направления')
            ax_e.plot(epochs_e, mcc_e, color='#e65100', lw=1.5, ls='-.', label='MCC × 100')
            ax_e.scatter([best_ep_e], [best_val_e], color='#d32f2f', s=45, zorder=5)
            ax_e.axvline(best_ep_e, color='#d32f2f', ls=':', lw=1.2, label=f'Best: Эпоха {best_ep_e} (Macro-F1={best_val_e:.2f}%)')
            ax_e.set_title(f"Этап 2 (Экспертная адаптация): {title}", weight='bold', fontsize=10.5)
            ax_e.set_xlabel('Эпоха дообучения', weight='bold')
            ax_e.set_ylabel('Метрика на экспертной валидации, %', weight='bold')
            ax_e.set_ylim(65, 101)
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
            best_idx = int(np.argmax([d["selection_score"] for d in data]))
            best_ep = epochs[best_idx]
            best_val = macro_f1[best_idx]
            ax.plot(epochs, acc, color='#1976d2', lw=1.6, label='Accuracy направления')
            ax.plot(epochs, macro_f1, color='#2e7d32', lw=1.8, ls='--', label='Macro-$F_1$ направления')
            ax.plot(epochs, app_f1, color='#8e24aa', lw=1.4, ls='-.', label='Macro-$F_1$ применимости')
            ax.scatter([best_ep], [best_val], color='#d32f2f', s=45, zorder=5)
            ax.axvline(best_ep, color='#d32f2f', ls=':', lw=1.2, label=f'Best: Эпоха {best_ep} (Macro-F1={best_val:.2f}%)')

        ax.set_title(f"Этап 1: {title}", weight='bold', fontsize=9.5)
        ax.set_xlabel('Эпоха обучения (300 эпох)', weight='bold', fontsize=8.5)
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
            best_idx = int(np.argmax([d["selection_score"] for d in data]))
            best_ep = epochs[best_idx]
            best_val = macro_f1[best_idx]
            ax.plot(epochs, acc, color='#1976d2', lw=1.6, label='Accuracy направления')
            ax.plot(epochs, macro_f1, color='#2e7d32', lw=1.8, ls='--', label='Macro-$F_1$ направления')
            ax.plot(epochs, mcc, color='#e65100', lw=1.5, ls='-.', label='MCC × 100')
            ax.scatter([best_ep], [best_val], color='#d32f2f', s=45, zorder=5)
            ax.axvline(best_ep, color='#d32f2f', ls=':', lw=1.2, label=f'Best: Эпоха {best_ep} (Macro-F1={best_val:.2f}%)')

        ax.set_title(f"Этап 2: {title}", weight='bold', fontsize=9.5)
        ax.set_xlabel('Эпоха дообучения', weight='bold', fontsize=8.5)
        ax.set_ylabel('Метрика на экспертной валидации, %', weight='bold', fontsize=8.5)
        ax.set_ylim(65, 101)
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
    """Текущие веса и совпадающие экспертные сетки, с проверкой происхождения."""
    from scripts.phase5_experiments.evaluate_pdr_expert_holdout import _sha256
    root = PROJECT_ROOT / "experiments/phase5/pdr_expert_evaluation_v3"
    analytic_path = root / "analytical_validation_v6.json"
    if not analytic_path.exists():
        print("[Ожидание] Рисунок 10: сначала evaluate_pdr_expert_holdout.py")
        return False
    analytic = json.loads(analytic_path.read_text(encoding="utf-8"))
    names = {
        "adaptive_pdr_mir": "Адаптивный (учитель)", "phase_pdr_basic": "Пофазный угловой",
        "pos_seq_pdr_basic": "Угловой прямой посл.", "phase_power_pdr_basic": "Пофазный мощностной",
        "pos_seq_power_pdr_basic": "Мощностной прямой посл.", "pdr_sivokobylenko_2pt": "Двухвыборочный",
        "pdr_sivokobylenko_5pt": "Пятивыборочный", "pdr_bmrz_q_assisted": "БМРЗ (адаптация)",
        "pdr_bavr072_crosspol": "БАВР-072 (адаптация)",
    }
    panels = [[], []]
    for algorithm in analytic["algorithm_ids"]:
        metrics = dict(analytic["overall"][algorithm])
        if "joint_state_accuracy" not in metrics:
            metrics["joint_state_accuracy"] = sum(
                row[algorithm]["tp"] + row[algorithm]["tn"] + row[algorithm]["applicability_tn"]
                for row in analytic["by_source"].values()) / metrics["n_applicability_samples"]
        panels[0].append((names[algorithm], metrics))
    for mode in ("snapshot_2", "snapshot_5", "sequence_1_8"):
        for stage, name in (("weak_initial", "до адаптации"), ("expert_best", "после адаптации")):
            payload = json.loads((root / f"{mode}_{stage}.json").read_text(encoding="utf-8"))
            validation = payload["splits"]["validation"]
            if validation["grid_signature"] != analytic["grid_signature"]:
                raise ValueError("Рисунок 10: несовпадающие экспертные сетки")
            checkpoint = Path(payload["checkpoint"])
            if not checkpoint.exists() or _sha256(checkpoint) != payload["checkpoint_sha256"]:
                raise ValueError(f"Веса изменились после оценки: {checkpoint}")
            metrics = dict(validation["overall"])
            metrics["joint_state_accuracy"] = sum(
                row["joint_state_accuracy"] * row["n_applicability_samples"]
                for row in validation["by_source"].values()) / metrics["n_applicability_samples"]
            panels[1].append((f"{mode}: {name}", metrics))
    keys = ("accuracy", "macro_f1_score", "recall", "specificity",
            "applicability_macro_f1_score", "joint_state_accuracy")
    titles = ("Верные\nDIR", "Macro-F1\nDIR", "Полнота\nFORWARD",
              "Полнота\nREVERSE", "Macro-F1\nVALID", "Верное\nсостояние")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5),
                             gridspec_kw={"width_ratios": [6, 1], "height_ratios": [9, 6]})
    for i, rows in enumerate(panels):
        values = np.array([[metrics[key] * 100 for key in keys] for _, metrics in rows])
        mcc = np.array([[metrics["mcc"]] for _, metrics in rows])
        ax, corr = axes[i]
        ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        corr.imshow(mcc, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(rows)), [f"{name} (n={int(m['n_samples']):,})" for name, m in rows], fontsize=8)
        ax.set_xticks(range(len(keys)), titles, fontsize=8)
        corr.set_xticks([0], ["MCC\nDIR"], fontsize=8)
        corr.set_yticks([])
        for row in range(len(rows)):
            for col in range(len(keys)):
                ax.text(col, row, f"{values[row,col]:.1f}", ha="center", va="center",
                        color="white" if values[row,col] > 65 else "black", fontsize=8)
            corr.text(0, row, f"{mcc[row,0]:.3f}", ha="center", va="center",
                      color="white" if abs(mcc[row,0]) > .6 else "black", fontsize=8)
        ax.set_title(("Аналитические РНМ", "Нейромодели: фактическая инициализация → лучшая экспертная эпоха")[i], fontsize=10)
    points = sum(row["points"] for row in analytic["grid_signature"].values())
    records = sum(row["records"] for row in analytic["grid_signature"].values())
    fig.suptitle(f"Согласие с экспертом: {records} валидационных осциллограммы, {points:,} точек", fontsize=13)
    fig.text(.32, .025,
        "Все доли — в %, MCC — от −1 до 1. n — число точек оценки направления.\n"
        "DIR аналитических РНМ: совместная применимость; DIR ИИ: применимость эксперта.\n"
        "«Верное состояние» учитывает и VALID, и DIR на единой полной сетке.\n"
        "Ограничение: 2 validation-записи имеют копии в train; итоговая независимая проверка не выполнена.", fontsize=9)
    fig.subplots_adjust(left=.32, right=.98, bottom=.12, top=.92, hspace=.35, wspace=.08)
    save_fig("fig10_expert_evaluation_comparison.png")
    plt.close(fig)
    return True


# -------------------------------------------------------------
# РИСУНОК 11: Анализ поведения органов на сложных осциллограммах
# -------------------------------------------------------------
def _record_spectral_cache(raw, provenance, basis, timebase, mode, version, ends):
    """Точные признаки обучения с переиспользованием одинаковых окон Фурье.

    Только причинные окна, без изменения временной привязки/нормировки.
    Память ограничена одной осциллограммой; обучение этот путь не использует.
    """
    from osc_tools.ml.phase5_contracts import periods_to_samples, spectral_positions
    from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig
    builder = SpectralFeatureBuilder(SpectralFeatureConfig(version))
    history = max(builder.config.low_periods)
    total = periods_to_samples(history + timebase.window_periods, timebase.spp)
    local_positions = np.asarray(spectral_positions(
        total, timebase.spp, mode, history_periods=history,
        stride_fraction=timebase.stride_fraction,
    ))
    positions = np.asarray(ends)[:, None] - total + 1 + local_positions[None, :]
    unique, inverse = np.unique(positions, return_inverse=True)
    features, masks, metadata = builder.build(
        raw.T, timebase.spp, unique, voltage_basis=basis, channel_provenance=provenance,
    )
    feature_provenance = np.broadcast_to(
        np.asarray(metadata["feature_provenance"], dtype=np.int64), features.shape,
    ).copy()
    feature_provenance[masks] = 0
    return features, feature_provenance, inverse.reshape(positions.shape)


def _expert_state_track(valid, forward):
    """Знаковый тип обязателен: NumPy может привести -999 к uint8 (25)."""
    return np.where(np.asarray(valid, dtype=bool), np.asarray(forward, dtype=np.int16), -999)


def build_expert_gallery(
    checkpoint=None, *, output_dir=None, inference_stride=1, max_records=None,
    resume=True,
):
    """Полная галерея реальных предсказаний; рисунок 11 статьи НЕ изменяется.

    Сохранённые точки ИИ рассчитаны реально, а не скопированы с учителя.
    При stride>1 между точками показывается удержание предыдущего решения.
    """
    import csv
    import html
    import time
    from scripts.phase5_experiments.evaluate_pdr_expert_holdout import _config_from_json, _sha256
    from scripts.phase5_experiments.run_phase5_pdr_training import _build_model, _imports
    from osc_tools.ml.dataset_registry import create_source
    from osc_tools.ml.phase5_contracts import TimebaseContract, periods_to_samples
    from osc_tools.pdr.signal_analysis import derive_missing_currents, check_pdr_signal_sufficiency
    from osc_tools.pdr.expert_labels import read_comtrade_1999_ascii, _read_text
    from osc_tools.pdr.study import PDRStudyLabelStore

    torch, *_, = _imports()
    from osc_tools.pdr.pdr_trainer import extract_backbone_features
    torch.set_num_threads(2)
    if inference_stride < 1:
        raise ValueError("Шаг предсказаний должен быть положительным")
    checkpoint = Path(checkpoint or PROJECT_ROOT / "experiments/phase5/pdr_expert_snapshot_5_stride5/best_model.pt")
    cfg = _config_from_json(checkpoint.parent / "config.json")
    model, head, _ = _build_model(cfg, None, checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head = model.to(device).eval(), head.to(device).eval()
    output_dir = Path(output_dir or PROJECT_ROOT / "data/phase5/pdr_expert_gallery")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256(checkpoint)
    records = pd.read_csv(PROJECT_ROOT / "data/phase5/pdr_expert_labels_v1/record_audit.csv")
    splits = pd.read_csv(PROJECT_ROOT / "data/phase5/pdr_expert_labels_v1/records.csv")
    split_map = {(r.source, int(r.record_id)): r.split for r in splits.itertuples()}
    records = records.sort_values(["source", "record_id"])
    if max_records is not None:
        records = records.head(max_records)
    algorithms = {
        "adaptive_pdr_mir": "Адаптивный РНМ",
        "pos_seq_power_pdr_basic": "Мощностной прямой посл.",
        "phase_pdr_basic": "Пофазный угловой",
        "pdr_bavr072_crosspol": "Адаптация БАВР-072",
    }
    sources, stores, manifest_hashes = {}, {}, {}
    for source in records.source.unique():
        sources[source] = create_source(PROJECT_ROOT / "data/phase5/datasets_registry.json", source)
        root = PROJECT_ROOT / "data/phase5/pdr_labels_v6" / source
        manifest_hashes[source] = _sha256(root / "manifest.json")
        stores[source] = {key: PDRStudyLabelStore(root, algorithm_id=key) for key in algorithms}
    entries, started = [], time.monotonic()
    try:
        for number, row in enumerate(records.itertuples(), 1):
            source, record_id = row.source, int(row.record_id)
            stem = f"{source}__record_{record_id:05d}"
            cfg_path = Path(row.cfg)
            if not cfg_path.exists():
                matches = list((PROJECT_ROOT / "data/phase5/pdr_manual_labels").rglob(stem + ".cfg"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Неоднозначный/отсутствующий COMTRADE: {stem}")
                cfg_path = matches[0]
            split = split_map.get((source, record_id), "not_trainable")
            folder = output_dir / ("holdout_не_использовать_для_подбора" if split == "holdout" else split) / source
            folder.mkdir(parents=True, exist_ok=True)
            image_path, cache_path = folder / (stem + ".png"), folder / (stem + ".npz")
            provenance = {
                "schema": 2, "checkpoint_sha256": checkpoint_hash,
                "cfg_sha256": _sha256(cfg_path), "dat_sha256": _sha256(cfg_path.with_suffix(".dat")),
                "automatic_manifest_sha256": manifest_hashes[source],
                "inference_stride": inference_stride, "temporal_mode": cfg.temporal_mode,
                "split": split, "status": row.status,
            }
            sidecar = image_path.with_suffix(".json")
            if not (resume and image_path.exists() and cache_path.exists() and sidecar.exists()
                    and json.loads(sidecar.read_text(encoding="utf-8")) == provenance):
                cached_meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
                can_reuse = (resume and cache_path.exists() and bool(cached_meta)
                             and {k: v for k, v in cached_meta.items() if k != "schema"}
                             == {k: v for k, v in provenance.items() if k != "schema"})
                rec = read_comtrade_1999_ascii(cfg_path)
                t = rec.timestamps_us / 1e6
                raw = sources[source].load_signal(record_id)
                channel_prov = sources[source].get_provenance(record_id)
                meta = sources[source].get_metadata(record_id)
                basis = str(meta.get("voltage_basis", "phase"))
                tb = TimebaseContract.create(float(meta.get("f_adc", meta.get("sampling_rate_hz"))),
                                             float(meta.get("f_network", meta.get("network_frequency_hz"))))
                raw, channel_prov = derive_missing_currents(raw, channel_prov)
                if raw.shape[1] != rec.n_samples:
                    raise ValueError(f"Длина источника и COMTRADE различается: {stem}")
                # -998 — отсутствие полного окна/каналов; -999 — ответ VALID=0.
                states = np.full(rec.n_samples, -998, dtype=np.int16)
                first = periods_to_samples(20, tb.spp) - 1
                ends = np.arange(first, rec.n_samples, inference_stride, dtype=np.int64)
                if not check_pdr_signal_sufficiency(channel_prov, basis).can_run_phase_pdr:
                    ends = ends[:0]
                predictions, valid_probabilities = [], []
                if can_reuse:
                    with np.load(cache_path) as cached:
                        if not np.array_equal(cached["samples"], ends):
                            raise ValueError(f"Сетка сохранённых предсказаний изменилась: {stem}")
                        predictions = cached["direction"].tolist()
                        valid_probabilities = cached["probability_valid"].tolist()
                elif ends.size:
                    features, feature_prov, lookup = _record_spectral_cache(
                        raw, channel_prov, basis, tb, cfg.temporal_mode, cfg.feature_version, ends,
                    )
                    with torch.inference_mode():
                        for begin in range(0, len(ends), 256):
                            indices = lookup[begin:begin + 256]
                            batch = {
                                "features": torch.from_numpy(features[indices]),
                                "provenance": torch.from_numpy(feature_prov[indices]),
                            }
                            outputs = head(extract_backbone_features(model, batch, device))
                            predictions.extend(outputs["logits"].argmax(-1).cpu().tolist())
                            valid_probabilities.extend(outputs["applicability_logit"].sigmoid().cpu().tolist())
                if ends.size:
                    decoded = np.where(np.asarray(valid_probabilities) >= .5, predictions, -999)
                    nearest = np.searchsorted(ends, np.arange(first, rec.n_samples), side="right") - 1
                    states[first:] = decoded[nearest]
                np.savez_compressed(cache_path, samples=ends, direction=np.asarray(predictions, dtype=np.int8),
                                    probability_valid=np.asarray(valid_probabilities, dtype=np.float32))
                expert = _expert_state_track(rec.digital["expert__VALID"], rec.digital["expert__FWD"])
                tracks = [("Эксперт" + (" (спорно)" if row.status == "ambiguous" else ""), expert),
                          (f"Нейросеть {cfg.temporal_mode}", states)]
                for algorithm, name in algorithms.items():
                    track = np.full(rec.n_samples, -998, dtype=np.int16)
                    if stores[source][algorithm].has_record(record_id):
                        automatic = stores[source][algorithm].get_record(record_id)
                        samples = np.asarray(automatic["samples"], dtype=np.int64)
                        in_bounds = (samples >= 0) & (samples < rec.n_samples)
                        track[samples[in_bounds]] = automatic["directions"][in_bounds]
                    tracks.append((name, track))
                fig, axes = plt.subplots(3, 1, figsize=(17, 8), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 2, 3]}, layout="constrained")
                cfg_lines = list(csv.reader(_read_text(cfg_path).splitlines()))
                analog_count = int(cfg_lines[1][1].rstrip("Aa"))
                analog_info = {line[1].strip(): line for line in cfg_lines[2:2 + analog_count]}
                for ax, names, label in ((axes[0], ("IA", "IB", "IC"), "Токи"),
                                         (axes[1], ("UA", "UB", "UC", "UAB", "UBC", "UCA"), "Напряжения")):
                    units = set()
                    for name in names:
                        if name in rec.analog:
                            info = analog_info[name]
                            values = rec.analog[name] * float(info[5]) + float(info[6])
                            units.add(info[4])
                            ax.plot(t, values, lw=.55, label=name)
                    ax.set_ylabel(label + ", " + "/".join(sorted(units)))
                    ax.legend(loc="upper right", ncol=3, fontsize=8)
                    ax.grid(alpha=.2)
                from matplotlib.colors import ListedColormap, BoundaryNorm
                palette = ["#eeeeee", "#888888", "#d67b33", "#20834a"]
                matrix = np.array([np.select([s == -998, s == -999, s == 0, s == 1], [0, 1, 2, 3]) for _, s in tracks])
                edges = np.r_[t, t[-1] + 1.0 / rec.sample_rate_hz]
                axes[2].pcolormesh(edges, np.arange(len(tracks) + 1), matrix,
                                   cmap=ListedColormap(palette), norm=BoundaryNorm(np.arange(-.5, 4.5), 4),
                                   shading="flat", rasterized=True)
                axes[2].set_yticks(np.arange(len(tracks)) + .5, [name for name, _ in tracks])
                axes[2].invert_yaxis()
                axes[2].set_xlabel("Время, с — полная осциллограмма")
                axes[2].legend(handles=[patches.Patch(color=c, label=l) for c, l in zip(palette,
                    ("Нет контекста/каналов", "VALID=0", "REVERSE=0", "FORWARD=1"))],
                    loc="upper center", bbox_to_anchor=(.5, -.19), ncol=4, fontsize=9)
                axes[0].set_title(f"{stem} | {row.status} | {split} | ИИ: шаг {inference_stride} отсчёт(ов)")
                axes[2].set_xlim(t[0], edges[-1])
                fig.savefig(image_path, dpi=140)
                plt.close(fig)
                sidecar.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
            entries.append((stem, row.status, split, image_path.relative_to(output_dir).as_posix()))
            elapsed = time.monotonic() - started
            print(f"[Галерея] {number}/{len(records)}; {elapsed:.0f} с; {stem}", flush=True)
    finally:
        for group in stores.values():
            for store in group.values():
                store.close()
    with (output_dir / "index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("record", "status", "split", "image"))
        writer.writerows(entries)
    links = "\n".join(f'<li>{html.escape(status)} / {html.escape(split)}: <a href="{html.escape(path, quote=True)}">{stem}</a></li>'
                      for stem, status, split, path in entries)
    (output_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Полные осциллограммы РНМ</title>'
        '<h1>Галерея для выбора рисунка 11</h1><p>Реальные предсказания ИИ; текущий рисунок статьи не изменён. '
        'Не используйте holdout для подбора модели или порогов. Просмотр этих файлов раскрывает контрольную выборку. '
        'Спорные экспертные метки не являются эталоном.</p><ol>' + links + '</ol>', encoding="utf-8")
    print(f"[Готово] {output_dir / 'index.html'}", flush=True)


def plot_fig11():
    raise RuntimeError("Старый макет не содержал реального вывода ИИ. Используйте --gallery; рисунок статьи меняется только после выбора автора.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Рисунки статьи и отдельная галерея реальных предсказаний")
    parser.add_argument("--gallery", action="store_true", help="Полные осциллограммы, без изменения рисунка 11")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inference-stride", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    if args.gallery:
        build_expert_gallery(args.checkpoint, inference_stride=args.inference_stride, max_records=args.max_records)
        raise SystemExit(0)
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
    print("[Пропуск] Рисунок 11 сохраняется без изменений. Для выбора примеров: --gallery")
    print(f"All figures successfully generated in: {OUTPUT_DIRS}")

