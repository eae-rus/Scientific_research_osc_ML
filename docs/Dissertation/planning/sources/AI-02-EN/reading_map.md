# AI-02-EN: карта чтения

Дата: 05.10.2026. Шаг C02-01; карта не является приёмкой экспериментов.

Основной вход: `D:/Программирование/Fork/Scientific_research_osc_ML/docs/article/Engineering_Optimal_Feature_Spaces_OJIES/Engineering_Optimal_Feature_Spaces_OJIES v1.2.tex`.
Полный SHA-256: `eafaae8a56418286516957232b862ec35226c81011a8b98ebbb4711c131d7630`.

Пакет: 11 файлов; полный реестр и происхождение — [manifest.json](../manifest.json). Принятые для работы версии сохранены побайтово в `sources/AI-02-EN/<sha256>/`; старые версии не объединены с основной.

## Объём чтения

Прочитаны структура всего TEX, abstract L0041–L0043, инженерные рекомендации, ограничения и conclusion L0408–L0431, раздел профилирования L0371–L0405. Полный экспериментальный протокол и таблицы ещё предстоит сверить.

## Научная роль и проверки

Часть главы 4 о признаках; общее семейство AI-02. 990 осциллограмм и факторная сетка 9450 запусков — заявления источника до аудита. Сам источник относит physics-informed KAN к будущей работе: обычная KAN-семья в этой сетке не тождественна всем вариантам ФизКАН. Время CPU не равно полной задержке органа.

## Структура основного текста

| Локатор | Раздел |
|---|---|
| `L0050` | \section{Introduction} |
| `L0071` | \section{Related Work} |
| `L0073` | \subsection{Feature Selection for Machine Learning in Power Engineering} |
| `L0077` | \subsection{Deep Learning in Protection Tasks} |
| `L0081` | \subsection{Kolmogorov--Arnold Networks as a New Class of Architectures} |
| `L0085` | \section{Dataset and Methodology} |
| `L0094` | \subsection{Dataset and Problem Statement}\label{sec:dataset} |
| `L0102` | \subsection{Data processing} |
| `L0110` | \section{Feature Space Design} |
| `L0121` | \subsection{Time-domain representations} |
| `L0145` | \subsection{Spectral representations} |
| `L0175` | \subsection{Input compression} |
| `L0190` | \section{Neural Network Architectures} |
| `L0194` | \subsection{Classical Architectures} |
| `L0198` | \subsection{Recurrent Architectures} |
| `L0203` | \subsection{The Kolmogorov-Arnold Family} |
| `L0225` | \section{Experimental Protocol} |
| `L0227` | \subsection{Splitting and cross-validation} |
| `L0231` | \subsection{Training and evaluation} |
| `L0239` | \subsection{Statistical Tests} |
| `L0243` | \section{Results and Analysis} |
| `L0245` | \subsection{Comparison of Feature Spaces} |
| `L0299` | \subsection{The Compressed-Strategy Paradox} |
| `L0330` | \subsection{Architecture Comparison} |
| `L0343` | \subsection{Per-Class Analysis of the Best Configuration} |
| `L0369` | \subsection{Computational Efficiency} |
| `L0408` | \section{Engineering Recommendations} |
| `L0422` | \section{Discussion and Limitations} |
| `L0427` | \section{Conclusion} |

Локаторы L — строки сохранённого TEX/MD. Для DATA-SD это Article.tex с SHA-256 `220eb9b265bfc28430857a7cd6318a2ab03c58aa8c0aedbb1fdb1e3b6f27587e`.

## Объекты для следующего прохода

- L0090: \caption{Conceptual pipeline: a field oscillogram (8 channels) is transformed through six feature spaces and three time-samplings and fed to seven architectures for a four-class multi-label task.}
- L0117: \caption{The six feature representations computed on one real oscillogram of a fault event (class Fault Events, ML-3): (a) raw instantaneous values; (b) symmetric components in Cartesian coordinates; (c) symmetric components in polar coordinates; (d) per-phase phasors in polar coordinates; (e) instantaneous power; (f) Clarke transform. The dotted vertical line marks the event onset near 200 ms.}
- L0180: \caption{Two-point snapshot on a real fault oscillogram: (a) the pre-fault snapshot $t_{\mathrm{pre}}$ and the fault-instant snapshot $t_{\mathrm{event}}$ on the positive-sequence magnitude trace, shown against the phase-$a$ current; (b) the corresponding phasors and the change vector $\Delta$ in the complex plane.}
- L0209: \caption{Number of trainable parameters by complexity configuration (thousands)}\label{tab1}
- L0252: \caption{Macro-F1 by feature representation (marginal means over all models, samplings, complexities, folds and seeds). Polar spectral representations lead; instantaneous power is weakest.}
- L0261: \caption{Macro-F1 heatmap (feature representation $\times$ model). Note the collapse of non-recurrent models on the power representation (bottom row).}
- L0266: \caption{Macro-F1 (mean $\pm$ std) by representation and architecture}\label{tab2}
- L0283: \caption{Significance of key differences (fold level, $n=5$)}\label{tab3}
- L0308: \caption{Effect of input compression (full window / stride / two-point snapshot) on Macro-F1 per model. Two-point snapshot closely tracks the full window; the gap is largest for recurrent models and near-zero for CNN/MLP.}
- L0313: \caption{Effect of the sampling mode on Macro-F1 (mean over complexities, folds, seeds)}\label{tab4}
- L0339: \caption{Macro-F1 by model (marginal over all features, samplings and complexities), grouped as recurrent / KAN-family / other. Recurrent models lead; ConvKAN is the most stable (lowest spread).}
- L0352: \caption{Per-class Macro-F1 and AUPRC for the best configuration (symmetric-polar / GRU / Heavy / stride). Classes: No Event, Operational Switching, Abnormal Events, Fault Events. Fault Events is the hardest and most variable class.}
- L0357: \caption{Per-class metrics of the best configuration (symmetric-polar / GRU / Heavy / stride)}\label{tab5}
- L0376: \caption{Macro-F1 versus model size for the symmetric-polar representation on the full window; the horizontal axis is logarithmic and the dashed line is the Pareto front. SimpleMLP and SimpleKAN sit far off the front (millions of parameters).}
- L0385: \caption{Macro-F1 versus CPU latency per window, measured on an Intel Core Ultra 7 165U in a single thread; the horizontal axis is logarithmic and the dashed line is the Pareto front. Circles denote the full window and triangles the two-point snapshot; the compressed configurations form the front.}
- L0392: \caption{CPU latency per window (ms), heavy configuration: full window vs two snapshots}\label{tab6}

## Версии и замечания

| Снимок / вход | Абзацы | Word comments | Правки |
|---|---:|---:|---:|
| `Engineering_Optimal_Feature_Spaces_OJIES v1.2.tex` — `eafaae8a56418286516957232b862ec35226c81011a8b98ebbb4711c131d7630` | — | — | — |

Сырые комментарии и все XML/relationships сохранены. Техническая проверка экспорта: 24 снимка, 333 комментария, 11 отслеживаемых изменений, 359 XML/relationship-частей по всем пакетам; байты совпали. Смысл замечаний других пакетов ещё не рассмотрен; C03 не закрывается этим экспортом.

Следующий этап — паспорт постановки/данных/методов; затем численные тезисы, критическое чтение и роль. Графику принимать отдельно в C04.
