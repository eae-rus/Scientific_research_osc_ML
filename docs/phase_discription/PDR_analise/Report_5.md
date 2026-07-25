# **Directional Power Relay** Algorithms for High-Speed Busbar Protection

Направленные реле мощности (РНМ / PDR) являются ключевыми элементами защиты и быстродействующего автоматического ввода резерва (БАВР), определяющими направление перетока мощности при коротких замыканиях. Классические алгоритмы РНМ основаны на вычислении скалярного произведения векторов тока и поляризующего напряжения, с заданием угла максимальной чувствительности (φ_mch) и сектора срабатывания ±90°  (Alhamrouni et al., 2024; Hooshyar & Iravani, 2018). Производители терминалов РЗА (SEL, Siemens, ABB, Schneider) используют различные методы поляризации — напряжением прямой последовательности (U1), напряжением предыстории (Memory Voltage) и кросс-поляризацию — для обеспечения устойчивости при близких трёхфазных КЗ  (Hooshyar & Iravani, 2018). Научные исследования демонстрируют, что традиционные направленные реле перегрузки (DOCR) требуют оптимальной координации уставок времени и тока, решаемой методами линейного и нелинейного программирования  (Rajput & Pandya, 2017; Korashy et al., 2018; Sarwagya et al., 2020). Интеграция распределённой генерации усложняет работу РНМ из-за двунаправленных потоков мощности и изменения уровней токов КЗ, что стимулирует разработку адаптивных алгоритмов защиты  (Abdi et al., 2024; Senapati et al., 2025; He et al., 2024). Российские производители (ЭКРА, Механотроника) разрабатывают органы направленности для БАВР, однако конкретные математические формулы часто составляют коммерческую тайну  (Kulikov et al., 2022). Глубокое обучение рассматривается в литературе как дополнение к классическим алгоритмам для преодоления "слепых зон" защиты, таких как повреждения с высоким переходным сопротивлением  (Fan et al., 2022).



**Figure 1:** Research consensus on classical directional power relay algorithms for high-speed busbar protection

## 2. Methods

Поиск выполнен по базе из более 170 миллионов научных публикаций в системе Consensus (включая Semantic Scholar, PubMed и другие источники). Из первоначально выявленных 117 1096 релевантных записей, машинная фильтрация по шести направлениям исследования сократила выборку до 240 отобранных и 182 уникальных статей, из которых топ-50 были включены в итоговый корпус анализа.

<search_strategy_diagram identifiedPapers="1171096" screenedPapers="240" eligiblePapers="182" includedPapers="50" />

**Figure 2:** Literature search and screening flow diagram for directional relay algorithms

Стратегия охватывала классическую теорию реле, manufacturer-specific алгоритмы, сравнительные исследования, синонимы технических терминов, контраст с нейросетевыми методами и смежные темы релейной защиты.

## 3. Results

### 3.1 Key Papers

Фундаментальные работы в корпусе охватывают теорию направленных реле перегрузки (DOCR), адаптивные алгоритмы для микросетей и новые направленные элементы на основе симметричных составляющих. Обзорные статьи по дистанционной защите и релейной защите с применением ИИ задают методологический контекст, а работы по координации DOCR предоставляют математические основы оптимизации уставок.

| Paper | Summary |
|-------|---------|
|  (Meidani et al., 2025)| New directional element using superimposed sequence impedance  (Hooshyar & Iravani, 2018)|
|  (Pujari & Alam, 2025)| Comprehensive review of distance relaying zones and characteristics  (Pujari & Alam, 2025)|
|  (Akhmedova et al., 2021)| Optimal DOCR coordination via harmony search algorithm tuning  (Rajput & Pandya, 2017)|

**Figure 3:** Key papers on directional power relay algorithms and coordination methods

### 3.2 Classical Directional Element Principles

Традиционные направленные элементы вычисляют угол между током и поляризующим напряжением, формируя срабатывание при попадании угла в заданный сектор. Реле перегрузки с направленной характеристикой (DOCR) используют стандартные обратно-зависимые времятоковые характеристики, координируемые через уставки времени (TMS) и пускового тока (PS)  (Rajput & Pandya, 2017; Korashy et al., 2018). Оптимизация координации DOCR формулируется как нелинейная задача с ограничениями, решаемая методами линейного программирования, генетическими алгоритмами и метаэвристиками  (Rajput & Pandya, 2017; Sarwagya et al., 2020; Bouchekara et al., 2017). Для микросетей с двунаправленными потоками мощности координация DOCR усложняется, требуя учёта режимов островной работы и подключения к сети  (Serna-Montoya et al., 2024). Введение направленного критерия в токовую защиту повышает её применимость в сетях с распределённой генерацией, однако количественная оценка этого улучшения ограничена  (He et al., 2024).

### 3.3 Polarization and Sensitivity

| Polarization Method | Application | Advantage | Citations |
|---|---|---|---|
| Positive-sequence (U1) | Symmetrical faults | Stable under balanced conditions |  (Hooshyar & Iravani, 2018)|
| Memory voltage | Close three-phase faults | Operates when U drops to zero |  (Akhmedova et al., 2021)|
| Superimposed Z2 | Asymmetrical faults in microgrids | Immune to DG fault behaviour |  (Hooshyar & Iravani, 2018)|

Ключевые различия: коммерческие impedance-based направленные элементы подвержены ложным срабатываниям из-за поведения инверторных источников при КЗ  (Hooshyar & Iravani, 2018). Напряжение предыстории (Memory Voltage) обеспечивает сохранение поляризации при глубоком падении напряжения, что критично для близких трёхфазных замыканий  (Akhmedova et al., 2021). Новые алгоритмы используют наложенные (superimposed) компоненты обратной и прямой последовательности для определения направления, что повышает устойчивость в микросетях с инверторной генерацией  (Hooshyar & Iravani, 2018).

### 3.4 BAVR and Adaptive Protection Features

Алгоритмы быстродействующего АВР (БАВР) модифицируют классические органы направленности для определения направления перетока активной и реактивной мощности при потере питания секции. Адаптивные схемы защиты используют группы уставок (Setting Groups) для автоматической коррекции зон срабатывания в зависимости от топологии сети и нагрузки  (Samadi & Chabanloo, 2025). Интеллектуальные системы дистанционной защиты корректируют уставки с учётом климатических факторов и параметров линий, снижая длину "мёртвой зоны"  (Akhmedova et al., 2021). Для сетей с ветрогенерацией типа DFIG предложены методы координации с временной задержкой и блокирующими сигналами, предотвращающие ложное срабатывание при активации crowbar-цепи  (Shahbazzadeh et al., 2025).

### Results Timeline

- **2017**
  - 2 papers:  (Rajput & Pandya, 2017; Bouchekara et al., 2017)- **2018**
  - 2 papers:  (Hooshyar & Iravani, 2018; Korashy et al., 2018)- **2020**
  - 1 paper:  (Sarwagya et al., 2020)- **2021**
  - 1 paper:  (Akhmedova et al., 2021)- **2022**
  - 2 papers:  (Kulikov et al., 2022; Fan et al., 2022)- **2023**
  - 1 paper:  (, 2023)- **2024**
  - 4 papers:  (Alhamrouni et al., 2024; Abdi et al., 2024; He et al., 2024; Serna-Montoya et al., 2024)- **2025**
  - 7 papers:  (Senapati et al., 2025; Pujari & Alam, 2025; Samadi & Chabanloo, 2025; Shahbazzadeh et al., 2025; Wadood et al., 2025; Samal et al., 2025; Mishra & Singh, 2025)**Figure 4:** Timeline of influential directional relay and protection algorithm papers; larger markers indicate more citations

### Top Contributors

<top_contributors authors='@@json@@:[{"name":"A. Hooshyar","citations":"318f64915c44509182e61e68258192b3"},{"name":"R. Iravani","citations":"318f64915c44509182e61e68258192b3"},{"name":"M. Mishra","citations":"41bd2eeea7cb5a318205bc2d68914522"}]' journals='@@json@@:[{"name":"Electric Power Systems Research","citations":"310e2a0ec5bd5227bcdbcb396ad09646,2852c646d69559608d51292dc1ffbfca,23ff1fd09a0a57f8a60fea67d18eb91c,41270fe775615c05be306cff808ab89e,d8c58d7236bc50f796948ae8db0eac5b"},{"name":"IEEE Access","citations":"84f40a75f816538cb9e93eb52a38b854,a7d38ba6c5f85401b01855523bafee45,edd9f252dbf2591e956aa8d37ede3542,78d8d7141a415badba1892901d6c8f1f"},{"name":"Energies","citations":"420d45d9b33e5a018ddb1256fe497c31,f5eedcef9e8a51e3af7e8bb912d1f800,b8873310a2ea5524879635e0d6f63395,2f9f2ff87d91595789a6a002cf9b7238"}]' />

**Figure 5:** Authors and journals that appeared most frequently in the included papers

## 4. Discussion

Корпус литературы демонстрирует зрелость классических алгоритмов направленной защиты, но выявляет существенный разрыв между академическими исследованиями и открытой документацией manufacturer-specific формул. Работы по координации DOCR предоставляют надёжную математическую основу для оптимизации уставок TMS и PS, однако конкретные алгоритмы поляризации и пороги чувствительности производителей SEL, Siemens, ABB и Schneider в рецензируемой литературе не раскрываются  (Rajput & Pandya, 2017; Korashy et al., 2018). Сильной стороной корпуса является обширный набор сравнительных исследований метаэвристических методов координации (HSA, NSGA-II, SCA, WCA, EFO), валидированных на стандартных тестовых системах IEEE  (Abdi et al., 2024; Sarwagya et al., 2020; Bouchekara et al., 2017; Wadood et al., 2025). Новые направленные элементы на основе наложенных симметричных составляющих предлагают инновационные решения для микросетей, но требуют валидации в реальных условиях  (Hooshyar & Iravani, 2018). Ограничения включают недостаток полевых испытаний, зависимость от симуляционных моделей (PSCAD/EMTDC, MATLAB/Simulink) и отсутствие стандартизированных критериев оценки для алгоритмов БАВР  (Senapati et al., 2025; Samal et al., 2025). Российские исследования предлагают адаптивные алгоритмы коррекции уставок и методы распознавания аварийных режимов на основе машинного обучения, но не предоставляют конкретных формул РНМ для БАВР  (Kulikov et al., 2022; Akhmedova et al., 2021).

| Claim | Evidence Strength | Reasoning | Papers |
|---|---|---|---|
| Classical DOCR coordination is a well-formulated nonlinear optimization problem solvable by metaheuristics | Evidence strength: Strong (9/10) | Multiple replicated studies on IEEE benchmark systems with consistent results |  (Rajput & Pandya, 2017; Korashy et al., 2018; Sarwagya et al., 2020; Bouchekara et al., 2017)|
| Superimposed sequence impedance directional elements outperform classical methods in microgrids with inverter-based DG | Evidence strength: Moderate (6/10) | Single highly-cited study validated in PSCAD/EMTDC, lacks field validation |  (Hooshyar & Iravani, 2018)|
| DG integration degrades traditional directional protection performance | Evidence strength: Strong (8/10) | Consistent findings across multiple independent studies and network topologies |  (Abdi et al., 2024; Senapati et al., 2025; He et al., 2024; Hooshyar & Iravani, 2018)|
| Memory voltage polarization solves close three-phase fault directionality | Evidence strength: Moderate (5/10) | Referenced in manufacturer literature and reviews, limited standalone validation studies |  (Akhmedova et al., 2021)|
| Russian manufacturer BAVR directional algorithms are proprietary/classified | Evidence strength: Weak (3/10) | Absence of published formulas in available literature; indirect references only |  (Kulikov et al., 2022)|

**Figure 6:** Key claims and supporting evidence identified in these papers

## 5. Conclusion

Синтез 50 рецензируемых публикаций показывает, что классические алгоритмы РНМ базируются на определении угла между током и поляризующим напряжением с сектором срабатывания ±90°, а координация DOCR формулируется как нелинейная оптимизационная задача. Интеграция распределённой генерации и микросетей стимулирует разработку новых методов поляризации на основе наложенных симметричных составляющих, однако конкретные формулы производителей остаются коммерческой тайной. Для открытой реализации в проекте Phase 5 доступны общие математические принципы и алгоритмы координации, но не полные спецификации уставок конкретных терминалов.

### Research Gaps

| Topic/Outcome | RCT/Field Validation | Manufacturer Formulas | BAVR-Specific Algorithms | Long-Term Performance |
|---|---|---|---|---|
| Classical DOCR coordination | **2** | **1** | **GAP** | **3** |
| Superimposed sequence elements | **1** | **GAP** | **GAP** | **1** |
| Memory voltage polarization | **1** | **2** | **1** | **2** |
| Russian BAVR terminals | **GAP** | **GAP** | **GAP** | **GAP** |
| Adaptive protection schemes | **2** | **1** | **1** | **2** |

### Open Research Questions

| Question | Why |
|---|---|
| **What are the exact mathematical formulas for directional elements in SEL-351 and SIPROTEC 7SJ82?** | Open implementation requires verified formulas, but manufacturers treat these as proprietary, limiting academic reproduction. |
| **How can superimposed sequence impedance methods be adapted for high-speed BAVR applications?** | BAVR requires sub-cycle direction decisions; existing methods are validated for line protection, not busbar transfer scenarios. |
| **What are the sensitivity thresholds (U_min, I_min) used by Russian BAVR terminals like EKRA 217?** | Russian manufacturers publish limited technical details, creating a gap for open-source implementation in domestic systems. |

Открытая реализация алгоритмов РНМ для БАВР требует синтеза классических принципов поляризации с современными методами адаптивной защиты,填补яя пробел между академической литературой и закрытыми спецификациями производителей.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
(2023). Optimizing Firefly Algorithm for Directional Overcurrent Relay Coordination: A case study on the Impact of Parameter Settings. *Information Sciences Letters*. https://doi.org/10.18576/isl/120745
 
Abdi, G., Jirdehi, M. A., & Mehrjerdi, H. (2024). Optimal coordination of overcurrent relays in microgrids using meta-heuristic algorithms NSGA-II and harmony search. *Sustain. Comput. Informatics Syst., 43*, 101020. https://doi.org/10.1016/j.suscom.2024.101020
 
Akhmedova, O., Soshinov, A., Gazizov, F., & Ilyashenko, S. (2021). Development of an Intelligent System for Distance Relay Protection with Adaptive Algorithms for Determining the Operation Setpoints. *Energies*. https://doi.org/10.3390/en14040973
 
Alhamrouni, I., Kahar, N. H. A., Salem, M., Swadi, M., Zahroui, Y., Kadhim, D. J., Mohamed, F. A., & Nazari, M. A. (2024). A Comprehensive Review on the Role of Artificial Intelligence in Power System Stability, Control, and Protection: Insights and Future Directions. *Applied Sciences*. https://doi.org/10.3390/app14146214
 
Bouchekara, H., Zellagui, M., & Abido, M. (2017). Optimal coordination of directional overcurrent relays using a modified electromagnetic field optimization algorithm. *Appl. Soft Comput., 54*, 267-283. https://doi.org/10.1016/j.asoc.2017.01.037
 
Fan, R., Yin, T., Yang, K., Lian, J., & Buckheit, J. (2022). New data-driven approach to bridging power system protection gaps with deep learning. *Electric Power Systems Research*. https://doi.org/10.1016/j.epsr.2022.107863
 
He, J., Mu, R., Li, B., Li, Y., Zhou, B., Xie, Z., & Wang, W. (2024). Applicability boundary calculation for directional current protection in distribution networks with accessed PV power sources. *Applied Energy*. https://doi.org/10.1016/j.apenergy.2024.123520
 
Hooshyar, A., & Iravani, R. (2018). A New Directional Element for Microgrid Protection. *IEEE Transactions on Smart Grid, 9*, 6862-6876. https://doi.org/10.1109/tsg.2017.2727400
 
Korashy, A., Kamel, S., Youssef, A., & Jurado, F. (2018). Modified water cycle algorithm for optimal direction overcurrent relays coordination. *Appl. Soft Comput., 74*, 10-25. https://doi.org/10.1016/j.asoc.2018.10.020
 
Kulikov, A., Loskutov, A., & Bezdushniy, D. (2022). Relay Protection and Automation Algorithms of Electrical Networks Based on Simulation and Machine Learning Methods. *Energies*. https://doi.org/10.3390/en15186525
 
Meidani, A., Abedini, M., & Sanaye‐Pasand, M. (2025). Enhancing Performance of Distance Relay Zone 3 Under Stressed Conditions Using an Angle-Based Algorithm. *IEEE Systems Journal, 19*, 270-281. https://doi.org/10.1109/jsyst.2025.3529720
 
Mishra, M., & Singh, J. G. (2025). A Comprehensive Review on Deep Learning Techniques in Power System Protection: Trends, Challenges, Applications and Future Directions. *Results in Engineering*. https://doi.org/10.1016/j.rineng.2024.103884
 
Pujari, R., & Alam, M. N. (2025). Distance Relaying for the Protection of Modern Power System Networks. *IEEE Access, 13*, 28861-28893. https://doi.org/10.1109/access.2025.3539919
 
Rajput, V., & Pandya, K. (2017). Coordination of directional overcurrent relays in the interconnected power systems using effective tuning of harmony search algorithm. *Sustain. Comput. Informatics Syst., 15*, 1-15. https://doi.org/10.1016/j.suscom.2017.05.002
 
Samadi, A., & Chabanloo, R. M. (2025). Adaptive distance protection for zone-1 optimal reach to mitigate the impact of network state changes using setting groups. *International Journal of Electrical Power &amp; Energy Systems*. https://doi.org/10.1016/j.ijepes.2025.111061
 
Samal, S., Samantaray, S. R., & Sharma, N. K. (2025). A New Differential Index-Based Fault Detection Scheme for Microgrids. *IEEE Transactions on Industry Applications, 61*, 4982-4991. https://doi.org/10.1109/tia.2025.3540734
 
Sarwagya, K., Nayak, P. K., & Ranjan, S. (2020). Optimal coordination of directional overcurrent relays in complex distribution networks using sine cosine algorithm. *Electric Power Systems Research*. https://doi.org/10.1016/j.epsr.2020.106435
 
Senapati, M., Panigrahi, P. K., Mohanty, A., Rajamony, R., Ray, P. K., & Allasi, H. (2025). Intelligent Protection Systems for Grid-Connected Renewables: A Review of AI Techniques and Applications. *Results in Engineering*. https://doi.org/10.1016/j.rineng.2025.107863
 
Serna-Montoya, L. F., Saldarriaga-Zuluaga, S., López-Lezama, J., & Muñoz-Galeano, N. (2024). Optimal Coordination of Directional Overcurrent Relays in Microgrids Considering European and North American Curves. *Energies*. https://doi.org/10.3390/en17235887
 
Shahbazzadeh, R., Hagh, M. T., & Ghanizadeh, R. (2025). Comprehensive analysis of challenges and two practical methods for protective coordination of distance relays in transmission lines connected to DFIG-based wind farms. *Scientific Reports, 15*. https://doi.org/10.1038/s41598-025-01728-2
 
Wadood, A., Albalawi, H. A., Alatwi, A. M., & Park, H. (2025). Modified Swarm-Based Artificial Intelligence Optimization for Optimal Coordination of Directional Overcurrent Relays in Power System. *IEEE Access, 13*, 71007-71026. https://doi.org/10.1109/access.2025.3563338
 
