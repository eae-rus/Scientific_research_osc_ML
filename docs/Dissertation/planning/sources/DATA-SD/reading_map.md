# DATA-SD: карта чтения

Дата: 05.10.2026. Шаг C02-01; карта не является приёмкой экспериментов.

Основной вход: `D:/Программирование/Fork/Scientific_research_osc_ML/docs/article/(Scientific_Data) Подготовка датасета/A dataset of real-world oscillograms from electrical power grids.pdf`.
Полный SHA-256: `4ab58820b8c6fdf9f129b2c33fe2ce03ae792afd7abb99e085a27cbceb95acdc`.

Пакет: 129 файлов; полный реестр и происхождение — [manifest.json](../manifest.json). Принятые для работы версии сохранены побайтово в `sources/DATA-SD/<sha256>/`; старые версии не объединены с основной.

## Объём чтения

Полностью прочитан научный текст Article.tex, включая методы, обе таблицы, подписи рисунков, ограничения и авторский вклад; прочитаны обе текстовые рецензии. PDF извлечён, 20 страниц; отрендерены страницы 2–15 и просмотрены все 12 рисунков/две таблицы источника. Отдельная приёмка зависимостей и диссертационного отображения C04 ещё предстоит.

## Научная роль и проверки

Опубликованный COMTRADE-срез → общий ресурс главы 3. Раздельно хранить 50 765 записей, 20 873 записи с возмущениями и 480 с временными метками. Автоэнкодер и классификаторы демонстрируют пригодность меток для заданной задачи, не доказывают физическую истину каждого события. По поручению автора локальная версия принята как содержательная основа, сверка с издателем не требуется.

## Структура основного текста

| Локатор | Раздел |
|---|---|
| `L0041` | \section*{Background \& Summary} |
| `L0072` | \section*{Methods} |
| `L0090` | \subsection*{Data collection} |
| `L0094` | \subsection*{Anonymization and Standardization} |
| `L0158` | \subsection*{Data Normalization} |
| `L0210` | \subsection*{Manual Annotation} |
| `L0259` | \section*{Data Records} |
| `L0289` | \section*{Technical Validation} |
| `L0339` | \section*{Usage Notes} |
| `L0362` | \section*{Data Availability} |
| `L0366` | \section*{Code Availability} |
| `L0442` | \section*{Author contributions} |
| `L0446` | \section*{Competing interests} |
| `L0450` | \section*{Funding} |

Локаторы L — строки сохранённого TEX/MD. Для DATA-SD это Article.tex с SHA-256 `220eb9b265bfc28430857a7cd6318a2ab03c58aa8c0aedbb1fdb1e3b6f27587e`.

## Объекты для следующего прохода

- L0052:     \caption{Relay Protection and Automation (RPA) terminals}
- L0079:     \caption{Example of Fast Automatic Bus Transfer (FABT) operation oscillogram. Discrete signals represent the manual labeling.}
- L0086:     \caption{Oscillogram collection and preparation scheme}
- L0109:     \caption{Typical substation section layout and signal measurement points.}
- L0116:     \caption{Example of an analog signal name.}
- L0123:     \caption{The structure of analog signals.}
- L0135:     \caption{Standard substation layout and circuit breaker positions (Incoming feeder breaker – IFB, Bus coupler — BC).}
- L0142:     \caption{Discrete signal structure.}
- L0228: \caption{Distribution of annotated timestamps by event group}
- L0244:     \caption{An example of decoding an ML signal.}
- L0251:     \caption{Simplified classification hierarchy scheme.}
- L0266:     \caption{Folder hierarchy and file organization of the unified OscGrid database.}
- L0304:     \caption{t-SNE visualization of autoencoder latent vectors colored by the four expert-annotated event groups, demonstrating separability of the classes in the learned feature space.}
- L0320: \caption{Quantitative validation of expert annotations: per-class Precision, Recall and F1-score.}

## Версии и замечания

| Снимок / вход | Абзацы | Word comments | Правки |
|---|---:|---:|---:|
| `A dataset of real-world oscillograms from electrical power grids.pdf` — `4ab58820b8c6fdf9f129b2c33fe2ce03ae792afd7abb99e085a27cbceb95acdc` | — | — | — |
| `Ревью-1 (вопрос 3).docx` — `3ea33a8e2986fd9de84927ca112fbcc7c65eba169f96466d69b1d2d5be838ad9` | 23 | 0 | 0 |
| `Ревью-2 (вопрос 2,3).docx` — `479e03f11633565339bc08d6a4330aaacdf948cb31fcdcfec251ceee77a976de` | 29 | 0 | 0 |
| `Article.tex` — `220eb9b265bfc28430857a7cd6318a2ab03c58aa8c0aedbb1fdb1e3b6f27587e` | — | — | — |

Сырые комментарии и все XML/relationships сохранены. Техническая проверка экспорта: 24 снимка, 333 комментария, 11 отслеживаемых изменений, 359 XML/relationship-частей по всем пакетам; байты совпали. Смысл замечаний других пакетов ещё не рассмотрен; C03 не закрывается этим экспортом.

Следующий этап — паспорт постановки/данных/методов; затем численные тезисы, критическое чтение и роль. Графику принимать отдельно в C04.
