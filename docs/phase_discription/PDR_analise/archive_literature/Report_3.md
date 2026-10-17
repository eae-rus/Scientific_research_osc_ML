# Исследование алгоритмов измерительных органов направления мощности (РНМ / PDR) и БАВР мировых и российских производителей

---

## 1. Сводная сравнительная таблица производителей

| Производитель / Серия | Органы / Коды ANSI | Схема включения / Поляризация | Угол макс. чувствительности ($\varphi_{mch}$ / RCA) | Границы сектора срабатывания | Пороги чувствительности ($U_{min}, I_{min}$) | Особенности использования памяти ($U_{mem}$) и БАВР |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SEL** *(SEL-351, 751, 387, 787)* | 32P, 32Q, 67P, 67N | $V_1$ (прямая послед.), $V_{1,mem}$, $V_2$ (обратная), кросс-поляризация | $0^\circ \dots +90^\circ$ (шаг $1^\circ$), для $V_2$: $-100^\circ \dots +100^\circ$ | Сектор $\pm 90^\circ$ относительно RCA | $I_1 > 0.05 I_n$, $U_1 > 0.02 U_n$ | Сохранение памяти $V_{1,mem}$ до 30–60 циклов (0.6–1.2 с) при 3-фазном КЗ близко к шинам. |
| **Siemens SIPROTEC** *(SIPROTEC 4 / 5: 7SJ82, 7SJ85, 7UT85)* | 32, 37, 67, 67N | Кросс-поляризация 90°, $V_1, I_1$, $V_2, I_2$, $3V_0, 3I_0$ | $-180^\circ \dots +180^\circ$ (RCA задается в настройках DIGSI 5) | Настраиваемый сектор $\alpha_1 \dots \alpha_2$ ($\pm 85^\circ \dots \pm 90^\circ$) | $U_{min} = 0.02 \dots 0.05 U_n$, $I_{min} = 0.02 \dots 0.05 I_n$ | Вычисление $V_{mem}$ на основе фильтра НЧ/БПФ предыстории; переход на $V_{mem}$ при $U < U_{min}$. |
| **ABB / Hitachi Energy** *(Relion 615, 620, 630: REF615)* | DOPPDPR (32R/32O), DRPPDUP, 67 | Фазные $U, I$, прямой последовательности $U_1, I_1$, quadrature 90° | RCA: $-90^\circ \dots +90^\circ$ (по умолчанию $+30^\circ$ или $+45^\circ$) | Зона срабатывания: $(\text{RCA} - 90^\circ) \dots (\text{RCA} + 90^\circ)$ | $P_{set} = 0.01 \dots 2.0 P_n$, $U_{min} = 0.01 U_n$, $I_{min} = 0.01 I_n$ | Поддержка режима Reverse Power ($32R$) для фиксации разворота мощности за $< 20\text{ мс}$. |
| **Schneider Electric** *(MiCOM P139, P443, Sepam 80)* | 32P, 32Q, 67 | 90-градусная схема ($I_A \leftrightarrow U_{BC}$), векторная сумма $P = \text{Re}(U \cdot I^*)$ | RCA: $+30^\circ, +45^\circ, +60^\circ$ (в зависимости от сети) | Настраиваемый угол сектора $\pm 80^\circ \dots \pm 90^\circ$ | $V_{polarizing} = \frac{V_{self}}{k} + V_{mem}$, $U_{min} \approx 0.05 U_n$ | Использование смеси $V_{self}$ и $V_{mem}$ с весом $k$ (параметр `Dist. Polarizing`). |
| **НТЦ Механотроника** *(БМРЗ-БАВР-01 / 56)* | РНМ (БАВР), ЗОМ (32R) | 90-градусная схема: $I_A \leftrightarrow U_{BC}$, $I_B \leftrightarrow U_{CA}$, $I_C \leftrightarrow U_{AB}$ | $\varphi_{mch} = -45^\circ$ (уставка `БАВР Фмч`) | 2 зоны: "Разрешение БАВР" (мощность в сеть) и "Блокировка БАВР" (мощность в нагрузку) | $I_{rnm} = 0.05 \dots 0.5 I_n$ (`БАВР Iрнм`), $U_{min} = 2 \dots 5 \text{ В}$ | Специальная работа "по памяти" при глубоком посаде напряжения для отстройки от КЗ на шинах. |
| **НПП ЭКРА** *(ЭКРА 217 / ШЭ2607 БАВР)* | ОНМр, ОНМб, РНМ БАВР | 90-градусная схема и прямая последовательность $U_1, I_1$ | $\varphi_{mch} = -45^\circ$ (для фазных РНМ), $+30^\circ \dots +45^\circ$ (для $U_1, I_1$) | Сектор $\pm 90^\circ$, сдвинутый на $\varphi_{mch}$ | $U_{min} \approx 2 \text{ В}$, $I_{min} \approx 0.05 I_n$ | Быстродействующая фиксация знака активной мощности $P_{ввод} < 0$ и разворота вектора тока. |
| **НПП Бреслер** *(Бреслер-0107.075 БАВР)* | РНМ БАВР, ОНМ | 90-градусная схема ($I_A, U_{BC}$) и прямой последовательности | $\varphi_{mch} = -45^\circ$ | Half-plane (полуплоскость) $\cos(\angle(U_{BC}, I_A) - \varphi_{mch}) > 0$ | $I_{min} \approx 0.02 \dots 0.05 I_n$, $U_{min} \approx 2 \text{ В}$ | Анализ векторной разности углов напряжений $d\Delta \varphi / dt$ и направления $P$ за время $< 15 \text{ мс}$. |
| **Классический / Научный подход** | PDR / Directional Torque | Прямой / Обратной последовательности / Quadrature | $\varphi_{mch} = \text{atan}(X/R)$ линии / сети | Полярный сектор или Mho-характеристика | Смещаемые пороги $Z_{shift}$ и $U_{mem}$ | Динамическая поляризация $U_{pol}(t) = U_1(t) \cdot e^{-t/\tau} + U_1^{mem} \cdot (1 - e^{-t/\tau})$. |

---

## 2. Детальный разбор алгоритмов по производителям

### 2.1 Schweitzer Engineering Laboratories (SEL)

#### Векторная поляризация и уравнения момента (Torque)
В терминалах SEL (SEL-351, SEL-751) направленные элементы используют величину электромагнитного момента (Torque Equation):

$$T_{op} = \text{Re} \left[ \mathbf{V}_{pol} \cdot \left( \mathbf{I}_{op} \cdot e^{-j \varphi_{mch}} \right)^* \right]$$

где:
- $\mathbf{V}_{pol}$ — поляризующее напряжение (Positive-sequence $V_1$, Negative-sequence $V_2$ или Memory Voltage $V_{1,mem}$);
- $\mathbf{I}_{op}$ — рабочий ток (Positive-sequence $I_1$ или фазный ток $I_A$);
- $\varphi_{mch}$ — угол максимальной чувствительности (MTA / RCA).

#### Алгоритм 32P / 32Q (Directional Power)
Для активной ($32P$) и реактивной ($32Q$) мощности вычисляются мгновенные и комплексные значения по 3 фазам:

$$P_{3ph} = \text{Re}(\mathbf{V}_A \mathbf{I}_A^* + \mathbf{V}_B \mathbf{I}_B^* + \mathbf{V}_C \mathbf{I}_C^*)$$

$$Q_{3ph} = \text{Im}(\mathbf{V}_A \mathbf{I}_A^* + \mathbf{V}_B \mathbf{I}_B^* + \mathbf{V}_C \mathbf{I}_C^*)$$

**Критерий срабатывания направления ВПРЕД (Forward):**

$$T_{op} > T_{threshold} > 0$$

#### Использование напряжения предыстории ($V_{1,mem}$)
При близких трёхфазных КЗ напряжение $V_1 \to 0$. Терминал SEL активирует circular memory buffer:

$$\mathbf{V}_{pol} = \begin{cases} \mathbf{V}_1, & \text{если } |\mathbf{V}_1| \ge V_{min} \\ \mathbf{V}_{1,mem}, & \text{если } |\mathbf{V}_1| < V_{min} \end{cases}$$

Буфер памяти удерживает фазу и амплитуду напряжения предыстории в течение $30 \dots 60$ циклов промышленной частоты ($0.6 \dots 1.2 \text{ с}$).

---

### 2.2 Siemens SIPROTEC (SIPROTEC 4 / SIPROTEC 5)

#### Алгоритм функции 32 (Power Protection)
В терминалах SIPROTEC 5 (7SJ82, 7SJ85, 7UT85) измерительный орган 32 рассчитывает комплексную мощность по ортогональным составляющим (Fourier Filtered Phase-to-Neutral components):

$$P_k = U_k \cdot I_k \cdot \cos(\varphi_{Uk} - \varphi_{Ik})$$

$$Q_k = U_k \cdot I_k \cdot \sin(\varphi_{Uk} - \varphi_{Ik})$$

$$P_{total} = P_A + P_B + P_C, \quad Q_{total} = Q_A + Q_B + Q_C$$

#### Секторная характеристика направления
Направление определяется углом между вектором напряжения $U$ и тока $I$, с учетом заданного угла наклона $\text{RCA}$ ($\varphi_{mch}$):

$$\Delta \varphi = \arg(\mathbf{I}) - \arg(\mathbf{U}) - \text{RCA}$$

**Критерий направления Forward (Вперёд):**

$$-\alpha_1 \le \Delta \varphi \le +\alpha_2$$

обычно $\alpha_1 = \alpha_2 = 85^\circ \dots 90^\circ$.

#### Блокировки и пороги
- Минимальное напряжение: $U < U_{min}$ ($2\% U_n$) — переключение на $U_{mem}$ или выдача сигнала `Direction_Unknown`.
- Минимальный ток: $I < I_{min}$ ($2\% I_n$) — блокировка работы.

---

### 2.3 ABB / Hitachi Energy (Relion 615 / 620 / 630 Series)

#### Алгоритм DOPPDPR (32R / 32O Directional Overpower / Reverse Power)
В терминале ABB REF615 орган направленной мощности вычисляет проекцию тока на поляризующую ось:

$$I_{proj} = |\mathbf{I}| \cdot \cos(\theta - \text{RCA})$$

где $\theta = \arg(\mathbf{U}) - \arg(\mathbf{I})$.

Мощность срабатывания $S_{op}$:

$$S_{op} = |\mathbf{U}| \cdot |\mathbf{I}| \cdot \cos(\arg(\mathbf{U}) - \arg(\mathbf{I}) - \text{RCA})$$

#### Уставки и критерии:
1. **Directional Mode**:
   - `Forward`: $S_{op} \ge P_{pickup}$ и $(\text{RCA} - 90^\circ) \le \theta \le (\text{RCA} + 90^\circ)$.
   - `Reverse`: $S_{op} \le -P_{pickup}$ или угол вне сектора forward.
2. **Пороговые условия**:
   - $P_{pickup}$ регулируется от $1\%$ до $200\%$ от $P_n$.
   - При $U < U_{min}$ ($0.01 U_n$) активируется защелка $U_{mem}$ на время $t_{mem} \le 1.0 \text{ с}$.

---

### 2.4 Schneider Electric (MiCOM P139/P443, Sepam series 80)

#### 90-градусная схема включения (Quadrature Polarization)
В реле серии MiCOM и Sepam 80 для фазных элементов направления тока/мощности применяется 90-градусная схема:
- Для тока фазы A ($I_A$) поляризующим напряжением служит $U_{BC}$;
- Для тока фазы B ($I_B$) — $U_{CA}$;
- Для тока фазы C ($I_C$) — $U_{AB}$.

Рабочий момент вычисляется как:

$$T_A = |\mathbf{U}_{BC}| \cdot |\mathbf{I}_A| \cdot \cos(\varphi_{BC, A} - \text{RCA})$$

где $\varphi_{BC, A} = \arg(\mathbf{I}_A) - \arg(\mathbf{U}_{BC})$.

#### Гибридная поляризация напряжением предыстории
В серии MiCOM P443 используется адаптивная формула поляризующего напряжения:

$$\mathbf{V}_{polarizing} = \frac{\mathbf{V}_{self}}{k} + \mathbf{V}_{memory}$$

где $k$ — параметр `Dist. Polarizing` ($0.2 \dots 5.0$). Это обеспечивает плавный переход при провалах напряжения.

---

### 2.5 Российские производители (Механотроника, ЭКРА, Бреслер) и алгоритмы БАВР

#### 2.5.1 НТЦ Механотроника (БМРЗ-БАВР-01 / БМРЗ-БАВР-56)
В устройствах БМРЗ-БАВР реализовано 3 или 6 реле направления мощности (по одному на каждый ввод и фазу).

##### Схема и формулы РНМ:
Применяется 90-градусная схема ($I_A \leftrightarrow U_{BC}$):

$$T_{op\_A} = |\mathbf{U}_{BC}| \cdot |\mathbf{I}_A| \cdot \cos(\varphi_{Ubc, Ia} - \varphi_{mch})$$

- Угольная уставка по умолчанию: `БАВР Фмч` = $-45^\circ$.
- Уставка минимального тока: `БАВР Iрнм` ($0.05 \dots 0.2 I_n$).

##### Алгоритм фиксации потери питания для БАВР:
1. В нормальном режиме активная мощность направлена из сети к шинам ($T_{op} > 0$, направление "В нагрузку").
2. При отключении головного выключателя или КЗ в питающей сети $110 / 35 \text{ кВ}$ происходит изменение знака мощности за счет выбега двигателей (переход мощности в направление "В сеть", $T_{op} < 0$) либо снижение тока ниже $I_{rnm}$.
3. РНМ мгновенно выдает дискретный сигнал "Разрешение БАВР".
4. Работа "по памяти" (напряжение предыстории) предотвращает ложную блокировку БАВР при близких 3-фазных КЗ на секции.

#### 2.5.2 НПП ЭКРА (ЭКРА 217 / ШЭ2607 БАВР)
Органы направленности ЭКРА в комплексах БАВР вычисляют:
1. **Фазные РНМ (90-градусные)** для фиксации разворота токов.
2. **Орган направления прямой последовательности**:

$$P_1 = \text{Re}(\mathbf{U}_1 \cdot \mathbf{I}_1^*)$$

Если $P_1 < -P_{BAVR\_thresh}$ в течение $10 \dots 15 \text{ мс}$, фиксируется потеря внешнего питания.

#### 2.5.3 НПП Бреслер (Бреслер-0107.075 БАВР)
Комплекс БАВР Бреслер использует сочетание:
- Вычисления знака фазовой мощности $P_A, P_B, P_C$;
- Мониторинга угловой скорости разворота вектора напряжения прямой последовательности $d(\arg \mathbf{U}_1) / dt$.
- Векторного условия: если $(\arg \mathbf{I}_1 - \arg \mathbf{U}_1)$ выходит из сектора $[-90^\circ + \varphi_{mch}, +90^\circ + \varphi_{mch}]$ в сторону противоположного полупространства, пуск БАВР происходит за время $t \le 12 \text{ мс}$.

---

### 2.6 Научные / Классические формульные алгоритмы

1. **Векторный критерий скалярного произведения (Vector Dot-Product Criterion)**:

$$S_{op} = \mathbf{V}_{pol} \cdot \mathbf{I}_{op} = |\mathbf{V}_{pol}| |\mathbf{I}_{op}| \cos(\theta)$$

Пусть $\mathbf{V}_{pol} = V_x + j V_y$, $\mathbf{I}_{op} = I_x + j I_y$.
Тогда поворот рабочего тока на угол $\varphi_{mch}$:

$$\mathbf{I}_{rot} = \mathbf{I}_{op} \cdot e^{-j \varphi_{mch}} = (I_x + j I_y)(\cos \varphi_{mch} - j \sin \varphi_{mch})$$

Электромагнитный момент:

$$T_{op} = \text{Re}(\mathbf{V}_{pol} \cdot \mathbf{I}_{rot}^*) = V_x I_{rot\_x} + V_y I_{rot\_y}$$

- $T_{op} > 0 \implies$ Направление **ВПРЁД (Forward)**.
- $T_{op} < 0 \implies$ Направление **НАЗАД (Reverse)**.

2. **Критерий разворота активной мощности для БАВР**:
При потере внешнего сетевого источника подпитывающие асинхронные и синхронные двигатели передают накопленную кинетическую энергию обратно в сеть.
Критерий пуска БАВР:

$$(P_{ввод} < -P_{rev\_set}) \quad \text{ИЛИ} \quad \left( (U_{bus} < U_{min}) \quad \text{И} \quad (T_{op\_A} < 0 \text{ ИЛИ } T_{op\_B} < 0 \text{ ИЛИ } T_{op\_C} < 0) \right)$$

---

## 3. Программная реализация на Python (`public_algorithms.py`)

Ниже приведена открытая реализация алгоритмов РНМ производителей (SEL, Siemens, ABB, Schneider, Механотроника, ЭКРА, Бреслер) и научно-классических формул. Этот модуль готов к сохранению в файл `osc_tools/pdr/public_algorithms.py`.

```python
"""
public_algorithms.py

Модуль открытых реализаций алгоритмов реле направления мощности (РНМ / PDR)
и пусковых органов БАВР мировых и российских производителей.

Разработано для интеграции в Phase 5 (Physical KAN-Transformer / PDR Suite).
"""

import math
import cmath
from typing import Dict, Tuple, Optional, Any
from enum import Enum


class DirectionResult(Enum):
    FORWARD = "FORWARD"       # Вперед / В зону / В нагрузку
    REVERSE = "REVERSE"       # Назад / Из зоны / В сеть (для БАВР — признак потери питания)
    UNKNOWN = "UNKNOWN"       # Направление неопределено (блокировка по Umin / Imin)


class BasePDRAlgorithm:
    """Базовый абстрактный класс для алгоритмов РНМ."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

    def calculate(
        self,
        u_complex: Tuple[complex, complex, complex],
        i_complex: Tuple[complex, complex, complex],
        u_mem_complex: Optional[Tuple[complex, complex, complex]] = None
    ) -> Dict[str, Any]:
        """
        Расчет направления мощности.
        
        :param u_complex: Комплексные фазные напряжения (Ua, Ub, Uc) в Вольтах
        :param i_complex: Комплексные фазные токи (Ia, Ib, Ic) в Амперах
        :param u_mem_complex: Напряжения предыстории (Ua_mem, Ub_mem, Uc_mem)
        :return: Словарь с результатами (direction, torque/power, details)
        """
        raise NotImplementedError


# =============================================================================
# 1. SCHWEITZER ENGINEERING LABORATORIES (SEL) - SEL-351 / SEL-751
# =============================================================================

class SEL351DirectionalPDR(BasePDRAlgorithm):
    """
    Реализация алгоритма SEL 32P/32Q/67P с поляризацией напряжением 
    прямой последовательности (V1) и памятью (V1_mem).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "mta_deg": 60.0,         # Maximum Torque Angle (0..90 deg)
            "u_min_volts": 2.0,      # Порог Umin
            "i_min_amps": 0.1,       # Порог Imin
            "sector_half_width_deg": 90.0
        }
        if params:
            default_params.update(params)
        super().__init__("SEL-351 Positive-Sequence Torque PDR", default_params)

    @staticmethod
    def _calc_positive_sequence(v_a: complex, v_b: complex, v_c: complex) -> complex:
        a = cmath.rect(1.0, 2.0 * math.pi / 3.0)  # exp(j * 120 deg)
        return (v_a + a * v_b + a * a * v_c) / 3.0

    def calculate(
        self,
        u_complex: Tuple[complex, complex, complex],
        i_complex: Tuple[complex, complex, complex],
        u_mem_complex: Optional[Tuple[complex, complex, complex]] = None
    ) -> Dict[str, Any]:
        u_a, u_b, u_c = u_complex
        i_a, i_b, i_c = i_complex

        # Расчет токов и напряжений прямой последовательности
        v1 = self._calc_positive_sequence(u_a, u_b, u_c)
        i1 = self._calc_positive_sequence(i_a, i_b, i_c)

        u_min = self.params["u_min_volts"]
        i_min = self.params["i_min_amps"]

        # Выбор поляризующего напряжения (текущее V1 или V1_mem)
        if abs(v1) >= u_min:
            v_pol = v1
        elif u_mem_complex is not None:
            v_pol = self._calc_positive_sequence(*u_mem_complex)
        else:
            v_pol = 0.0 + 0.0j

        if abs(v_pol) < 1e-3 or abs(i1) < i_min:
            return {
                "direction": DirectionResult.UNKNOWN,
                "torque": 0.0,
                "v1_mag": abs(v1),
                "i1_mag": abs(i1),
                "reason": "Below threshold Umin/Imin"
            }

        mta_rad = math.radians(self.params["mta_deg"])
        # SEL Torque Equation: Re[ Vpol * (I1 * exp(-j * mta))* ]
        i1_rotated = i1 * cmath.exp(-1j * mta_rad)
        torque = (v_pol * i1_rotated.conjugate()).real

        # Проверка угловых границ сектора
        angle_diff_deg = math.degrees(cmath.phase(i1) - cmath.phase(v_pol)) - self.params["mta_deg"]
        # Нормализация угла в range [-180, 180]
        angle_diff_deg = (angle_diff_deg + 180.0) % 360.0 - 180.0

        half_w = self.params["sector_half_width_deg"]
        if -half_w <= angle_diff_deg <= half_w and torque > 0:
            direction = DirectionResult.FORWARD
        else:
            direction = DirectionResult.REVERSE

        return {
            "direction": direction,
            "torque": torque,
            "angle_diff_deg": angle_diff_deg,
            "v_pol_mag": abs(v_pol),
            "i1_mag": abs(i1)
        }


# =============================================================================
# 2. SIEMENS SIPROTEC 5 (7SJ82 / 7SJ85)
# =============================================================================

class SiemensSiprotec5PowerPDR(BasePDRAlgorithm):
    """
    Реализация измерительного органа направленной мощности SIPROTEC 5 (ANSI 32/67).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "rca_deg": 45.0,           # Relay Characteristic Angle
            "sector_angle_deg": 85.0,  # Границы сектора (+/- 85 град)
            "u_min_volts": 3.0,
            "i_min_amps": 0.05
        }
        if params:
            default_params.update(params)
        super().__init__("Siemens SIPROTEC 5 Directional Power PDR", default_params)

    def calculate(
        self,
        u_complex: Tuple[complex, complex, complex],
        i_complex: Tuple[complex, complex, complex],
        u_mem_complex: Optional[Tuple[complex, complex, complex]] = None
    ) -> Dict[str, Any]:
        p_total = 0.0
        q_total = 0.0
        phase_directions = []

        rca_rad = math.radians(self.params["rca_deg"])
        sector_w = self.params["sector_angle_deg"]

        for idx, (u, i) in enumerate(zip(u_complex, i_complex)):
            # Замена на память при провале напряжения
            if abs(u) < self.params["u_min_volts"] and u_mem_complex:
                u_eff = u_mem_complex[idx]
            else:
                u_eff = u

            if abs(u_eff) < self.params["u_min_volts"] or abs(i) < self.params["i_min_amps"]:
                phase_directions.append(DirectionResult.UNKNOWN)
                continue

            s_comp = u_eff * i.conjugate()
            p_total += s_comp.real
            q_total += s_comp.imag

            # Относительный угол тока относительно напряжения с учетом RCA
            ang_diff = math.degrees(cmath.phase(i) - cmath.phase(u_eff)) - self.params["rca_deg"]
            ang_diff = (ang_diff + 180.0) % 360.0 - 180.0

            if -sector_w <= ang_diff <= sector_w:
                phase_directions.append(DirectionResult.FORWARD)
            else:
                phase_directions.append(DirectionResult.REVERSE)

        # Обобщенное решение по 3 фазам
        if DirectionResult.FORWARD in phase_directions:
            overall_dir = DirectionResult.FORWARD
        elif DirectionResult.REVERSE in phase_directions:
            overall_dir = DirectionResult.REVERSE
        else:
            overall_dir = DirectionResult.UNKNOWN

        return {
            "direction": overall_dir,
            "p_total_w": p_total,
            "q_total_var": q_total,
            "phase_directions": phase_directions
        }


# =============================================================================
# 3. ABB RELION (REF615 / REM615) - DOPPDPR
# =============================================================================

class ABBRelionDOPPDPR(BasePDRAlgorithm):
    """
    Реализация модуля ABB DOPPDPR (32R/32O) - Directional Overpower / Reverse Power.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "rca_deg": 30.0,
            "p_pickup_w": 100.0,        # Порог срабатывания по мощности (Вт)
            "u_min_volts": 1.5,
            "i_min_amps": 0.02
        }
        if params:
            default_params.update(params)
        super().__init__("ABB Relion DOPPDPR 32R/32O PDR", default_params)

    def calculate(
        self,
        u_complex: Tuple[complex, complex, complex],
        i_complex: Tuple[complex, complex, complex],
        u_mem_complex: Optional[Tuple[complex, complex, complex]] = None
    ) -> Dict[str, Any]:
        rca_rad = math.radians(self.params["rca_deg"])
        s_op = 0.0

        for idx, (u, i) in enumerate(zip(u_complex, i_complex)):
            u_act = u if abs(u) >= self.params["u_min_volts"] else (u_mem_complex[idx] if u_mem_complex else u)
            if abs(u_act) >= self.params["u_min_volts"] and abs(i) >= self.params["i_min_amps"]:
                theta = cmath.phase(u_act) - cmath.phase(i)
                s_op += abs(u_act) * abs(i) * math.cos(theta - rca_rad)

        p_thresh = self.params["p_pickup_w"]

        if s_op >= p_thresh:
            direction = DirectionResult.FORWARD
        elif s_op <= -p_thresh:
            direction = DirectionResult.REVERSE
        else:
            direction = DirectionResult.UNKNOWN

        return {
            "direction": direction,
            "s_operating_val": s_op,
            "p_pickup_w": p_thresh
        }


# =============================================================================
# 4. НТЦ МЕХАНОТРОНИКА (БМРЗ-БАВР-01 / БМРЗ-БАВР-56)
# =============================================================================

class BMRZFABTDirectionalPDR(BasePDRAlgorithm):
    """
    Реализация алгоритма РНМ БАВР блока БМРЗ по 90-градусной схеме включения.
    (Ia <-> Ubc, Ib <-> Uca, Ic <-> Uab) с уставкой 'БАВР Фмч' = -45 град.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "phi_mch_deg": -45.0,       # Уставка 'БАВР Фмч'
            "i_rnm_amps": 0.2,          # Уставка 'БАВР Iрнм'
            "u_min_volts": 2.0,         # Порог минимального линейного напряжения
            "hysteresis_deg": 5.0
        }
        if params:
            default_params.update(params)
        super().__init__("BMRZ БАВР 90-degree Quadrature PDR", default_params)

    def calculate(
        self,
        u_complex: Tuple[complex, complex, complex],
        i_complex: Tuple[complex, complex, complex],
        u_mem_complex: Optional[Tuple[complex, complex, complex]] = None
    ) -> Dict[str, Any]:
        u_a, u_b, u_c = u_complex
        i_a, i_b, i_c = i_complex

        # Расчет линейных напряжений
        u_bc = u_b - u_c
        u_ca = u_c - u_a
        u_ab = u_a - u_b

        # Напряжения предыстории для линейных величин при близком 3-фазном КЗ
        if u_mem_complex:
            um_a, um_b, um_c = u_mem_complex
            ubc_mem = um_b - um_c
            uca_mem = um_c - um_a
            uab_mem = um_a - um_b
        else:
            ubc_mem, uca_mem, uab_mem = u_bc, u_ca, u_ab

        pairs = [
            ("Phase_A", i_a, u_bc, ubc_mem),
            ("Phase_B", i_b, u_ca, uca_mem),
            ("Phase_C", i_c, u_ab, uab_mem)
        ]

        phi_mch_rad = math.radians(self.params["phi_mch_deg"])
        i_rnm = self.params["i_rnm_amps"]
        u_min = self.params["u_min_volts"]

        bavr_permit_signals = []
        phase_torques = {}

        for name, i_phase, u_lin, u_lin_mem in pairs:
            u_eff = u_lin if abs(u_lin) >= u_min else u_lin_mem

            if abs(i_phase) < i_rnm or abs(u_eff) < u_min:
                # Если ток ниже Iрнм, БМРЗ считает, что мощности в нагрузку нет -> Разрешение БАВР
                bavr_permit_signals.append(True)
                phase_torques[name] = 0.0
                continue

            # Расчет момента 90-градусной схемы
            # T = |U_lin| * |I_phase| * cos( angle(U_lin, I_phase) - phi_mch )
            ang_u = cmath.phase(u_eff)
            ang_i = cmath.phase(i_phase)
            ang_diff = ang_i - ang_u - phi_mch_rad

            torque = abs(u_eff) * abs(i_phase) * math.cos(ang_diff)
            phase_torques[name] = torque

            # Если момент положительный (мощность направлена от шин в сеть) -> Разрешение БАВР
            # В БМРЗ: Zona 'Блокировка БАВР' - мощность идет в нагрузку от энергосистемы.
            if torque < 0:
                bavr_permit_signals.append(False)  # Блокировка БАВР (питание идет штатно в нагрузку)
            else:
                bavr_permit_signals.append(True)   # Разрешение БАВР (потеря питания / мощность развернулась)

        # Пуск БАВР разрешен, если хотя бы по одной фазе зафиксирован разворот или ток ниже Iрнм
        bavr_allowed = any(bavr_permit_signals)
        direction = DirectionResult.REVERSE if bavr_allowed else DirectionResult.FORWARD

        return {
            "direction": direction,
            "bavr_permit": bavr_allowed,
            "phase_torques": phase_torques,
            "bavr_permit_signals": bavr_permit_signals
        }


# =============================================================================
# 5. НПП ЭКРА (ЭКРА 217 / ШЭ2607 БАВР)
# =============================================================================

class EkraDirectionalPDR(BasePDRAlgorithm):
    """
    Реализация органа направления мощности ЭКРА для защит и БАВР.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "phi_mch_deg": -45.0,
            "u_min_volts": 2.0,
            "i_min_amps": 0.1
        }
        if params:
            default_params.update(params)
        super().__init__("EKRA 217 BAVR Directional Element", default_params)

    def calculate(
        self,
        u_complex: Tuple[complex, complex, complex],
        i_complex: Tuple[complex, complex, complex],
        u_mem_complex: Optional[Tuple[complex, complex, complex]] = None
    ) -> Dict[str, Any]:
        u_a, u_b, u_c = u_complex
        i_a, i_b, i_c = i_complex

        u_bc = u_b - u_c
        phi_mch_rad = math.radians(self.params["phi_mch_deg"])

        u_eff_bc = u_bc if abs(u_bc) >= self.params["u_min_volts"] else (
            (u_mem_complex[1] - u_mem_complex[2]) if u_mem_complex else u_bc
        )

        if abs(i_a) < self.params["i_min_amps"] or abs(u_eff_bc) < self.params["u_min_volts"]:
            return {"direction": DirectionResult.UNKNOWN, "p_rel": 0.0}

        # Рабочая величина РНМ ЭКРА
        angle_diff = cmath.phase(i_a) - cmath.phase(u_eff_bc) - phi_mch_rad
        p_rel = abs(u_eff_bc) * abs(i_a) * math.cos(angle_diff)

        # Если P_rel > 0 — направление прямое (разрешающий ОНМр)
        direction = DirectionResult.FORWARD if p_rel > 0 else DirectionResult.REVERSE

        return {
            "direction": direction,
            "operating_power": p_rel,
            "angle_diff_deg": math.degrees(angle_diff)
        }


# =============================================================================
# 6. КЛАССИЧЕСКИЙ НАУЧНЫЙ АЛГОРИТМ ПОЛЯРИЗАЦИИ ПРЕДЫСТОРИЕЙ (Academic Memory-PDR)
# =============================================================================

class AcademicMemoryPolarizedPDR(BasePDRAlgorithm):
    """
    Классический формульный алгоритм РНМ с динамической экспоненциальной 
    адаптацией напряжения предыстории U_pol(t) = alpha*U1 + (1-alpha)*U1_mem.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "phi_mch_deg": 45.0,
            "alpha_weight": 0.3,        # Вес текущего напряжения (0.3 self + 0.7 memory)
            "u_min_volts": 1.0,
            "i_min_amps": 0.05
        }
        if params:
            default_params.update(params)
        super().__init__("Academic Memory-Polarized Directional Relay", default_params)

    @staticmethod
    def _pos_seq(va: complex, vb: complex, vc: complex) -> complex:
        a = cmath.rect(1.0, 2.0 * math.pi / 3.0)
        return (va + a * vb + a * a * vc) / 3.0

    def calculate(
        self,
        u_complex: Tuple[complex, complex, complex],
        i_complex: Tuple[complex, complex, complex],
        u_mem_complex: Optional[Tuple[complex, complex, complex]] = None
    ) -> Dict[str, Any]:
        u1 = self._pos_seq(*u_complex)
        i1 = self._pos_seq(*i_complex)

        if u_mem_complex:
            u1_mem = self._pos_seq(*u_mem_complex)
        else:
            u1_mem = u1

        alpha = self.params["alpha_weight"]
        # Адаптивное поляризующее напряжение
        v_pol = alpha * u1 + (1.0 - alpha) * u1_mem

        if abs(v_pol) < self.params["u_min_volts"] or abs(i1) < self.params["i_min_amps"]:
            return {"direction": DirectionResult.UNKNOWN, "torque": 0.0}

        phi_mch_rad = math.radians(self.params["phi_mch_deg"])
        # Torque: Re[ Vpol * (I1 * exp(-j*phi_mch))* ]
        i1_rot = i1 * cmath.exp(-1j * phi_mch_rad)
        torque = (v_pol * i1_rot.conjugate()).real

        direction = DirectionResult.FORWARD if torque > 0 else DirectionResult.REVERSE

        return {
            "direction": direction,
            "torque": torque,
            "v_pol_mag": abs(v_pol),
            "i1_mag": abs(i1)
        }


# =============================================================================
# РЕГИСТР И ФАБРИКА АЛГОРИТМОВ (PDRRegistry)
# =============================================================================

class PDRRegistry:
    """Реестр всех доступных открытых реализаций РНМ."""

    _ALGORITHMS = {
        "SEL-351": SEL351DirectionalPDR,
        "SIPROTEC-5": SiemensSiprotec5PowerPDR,
        "ABB-DOPPDPR": ABBRelionDOPPDPR,
        "BMRZ-BAVR": BMRZFABTDirectionalPDR,
        "EKRA-217": EkraDirectionalPDR,
        "ACADEMIC-MEM": AcademicMemoryPolarizedPDR
    }

    @classmethod
    def create(cls, algo_key: str, params: Optional[Dict[str, Any]] = None) -> BasePDRAlgorithm:
        if algo_key not in cls._ALGORITHMS:
            raise KeyError(f"Алгоритм '{algo_key}' не найден в реестре PDRRegistry. Доступные: {list(cls._ALGORITHMS.keys())}")
        return cls._ALGORITHMS[algo_key](params)

    @classmethod
    def list_algorithms(cls) -> list:
        return list(cls._ALGORITHMS.keys())


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ И ТЕСТИРОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    # Тестовый режим: Нормальный режим нагрузки (Мощность из сети в шины)
    # Напряжения: Phase A = 57.7V, Angle = 0
    u_a = cmath.rect(57.7, math.radians(0))
    u_b = cmath.rect(57.7, math.radians(-120))
    u_c = cmath.rect(57.7, math.radians(120))

    # Ток нагрузки: Phase A = 5A, отстает на 20 градусов (нагрузка индуктивная)
    i_a = cmath.rect(5.0, math.radians(-20))
    i_b = cmath.rect(5.0, math.radians(-140))
    i_c = cmath.rect(5.0, math.radians(100))

    u_test = (u_a, u_b, u_c)
    i_test = (i_a, i_b, i_c)

    print("=== ТЕСТИРОВАНИЕ АЛГОРИТМОВ РНМ В НОРМАЛЬНОМ РЕЖИМЕ ===")
    for key in PDRRegistry.list_algorithms():
        pdr_instance = PDRRegistry.create(key)
        res = pdr_instance.calculate(u_test, i_test)
        print(f"[{key:12s}] Result: {res['direction'].value:8s} | Details: {res}")

    print("\n=== ТЕСТИРОВАНИЕ БАВР ПРИ РАЗВОРОТЕ МОЩНОСТИ (ВЫБЕГ ДВИГАТЕЛЕЙ) ===")
    # При потере питания ток меняет фазу на ~180 градусов (двигатели генерируют в сеть)
    i_a_rev = cmath.rect(5.0, math.radians(160))
    i_b_rev = cmath.rect(5.0, math.radians(40))
    i_c_rev = cmath.rect(5.0, math.radians(-80))
    i_test_rev = (i_a_rev, i_b_rev, i_c_rev)

    bmrz_pdr = PDRRegistry.create("BMRZ-BAVR")
    res_bavr = bmrz_pdr.calculate(u_test, i_test_rev)
    print(f"[BMRZ-BAVR   ] Result: {res_bavr['direction'].value:8s} | BAVR Permit: {res_bavr['bavr_permit']}")
```

---

## 4. Заключение и дальнейшая интеграция

1. Собранные математические выражения и уставки покрывают основные линейки терминалов защиты (**SEL, Siemens, ABB, Schneider Electric**) и специализированные российские блоки БАВР (**Механотроника БМРЗ, ЭКРА, Бреслер**).
2. Написанный Python-модуль `public_algorithms.py` напрямую внедряется в архитектуру проекта **Phase 5 (Physical KAN-Transformer)** для моделирования работы РНМ, валидации осциллограмм аварийных процессов и генерации синтетических обучающих датасетов.


---

Запрос номер 2, уточняющий

---

Отличный и абсолютно точный вопрос! Это принципиальный момент, который стоит чётко разграничить для правильной постановки эксперимента в Phase 5.

Ниже приведен результат **проверки и систематизации**: что из описанного является **чисто специализированным алгоритмом БАВР**, а что — **классической РЗА (направленная защита)**, которую целесообразно использовать как **дополнительный/экспериментальный кандидат**.

---

## 1. Результаты проверки: БАВР vs Классическая РЗА

### 🎯 100% Специализированные алгоритмы БАВР (Основа исследования)
Эти устройства и алгоритмы создавались **именно для задачи БАВР / HSBT (High Speed Busbar Transfer)** — фиксации потери питания и выбега двигателей за $10 \dots 20\text{ мс}$:

1. **НТЦ Механотроника (БМРЗ-БАВР-01 / 56)**:
   - **Статус**: Чистый БАВР.
   - **Алгоритм**: Специализированная 90-градусная схема ($I_A \leftrightarrow U_{BC}$) с фиксированным углом $\varphi_{mch} = -45^\circ$ и разделением на зоны *"Блокировка БАВР"* (мощность в нагрузку) и *"Разрешение БАВР"* (мощность в сеть при выбеге двигателей).
2. **НПП ЭКРА (ЭКРА 217 / ШЭ2607 БАВР)**:
   - **Статус**: Чистый БАВР.
   - **Алгоритм**: Мгновенная фиксация знака мощности прямой последовательности $P_1 < 0$ и разворота токов за $10 \dots 15\text{ мс}$.
3. **НПП Бреслер (Бреслер-0107.075 БАВР)**:
   - **Статус**: Чистый БАВР.
   - **Алгоритм**: Мониторинг скорости разворота вектора напряжения $d(\Delta \varphi)/dt$ и скачкообразного изменения направления активной мощности $P_{ввод} < 0$.
4. **ABB SUE 3000** *(специализированный терминал БАВР от ABB)*:
   - **Статус**: Чистый БАВР (High Speed Busbar Transfer System).
   - **Алгоритм**: Непрерывный расчет вектора разности углов $\Delta \varphi = \arg(\mathbf{U}_{шины}) - \arg(\mathbf{U}_{резерв})$ и направления токов вводов.

---

### ⚙️ Алгоритмы классической РЗА (Дополнительные кандидаты для тестирования)
Эти органы первоначально предназначены для релейной защиты (67/32P/32Q — защита от обратной мощности, токовая направленная защита), но могут применяться в логике БАВР как измерительные блоки:

1. **Schweitzer Engineering Laboratories (SEL-351 / SEL-751)**:
   - **Статус**: Классическая РЗА (67P / 32P / 32Q).
   - **В БАВР**: В SEL специализированная функция БАВР называются **MBT (Motor Bus Transfer)** (реализуется в SEL-700G / SEL-751). Алгоритмы 32P/67P используются как *элементы фиксации обратной мощности*.
   - **Назначение в исследовании**: Дополнительный вариант (проверка эффективности классического уравнения момента Torque SEL в задаче БАВР).
2. **Siemens SIPROTEC (7SJ82/85)**:
   - **Статус**: Классическая РЗА (функции 32 и 67).
   - **В БАВР**: Специализированные шкафы БАВР у Siemens — это **SIPROTEC 7VU68 (HSBT)**. Органы 32/67 из 7SJ85 — это база РЗА.
   - **Назначение в исследовании**: Дополнительный вариант.
3. **Schneider Electric (MiCOM P139/P443, Sepam 80)**:
   - **Статус**: Классическая РЗА.
   - **Назначение в исследовании**: Дополнительный вариант.

---

## 2. Подробный разбор научных и математических алгоритмов для БАВР

Для задачи быстродействующего АВР обычный расчёт через БПФ (FFT) с окном в 1 период ($20\text{ мс}$) часто бывает слишком медленным. По этой причине в научных исследованиях и передовых терминалах БАВР применяют **мгновенные (безинерционные) формульные алгоритмы**.

### 🧪 Научный алгоритм №1: Мгновенная мощность в $\alpha\beta$-координатах (Clarke Transform)
**Суть:** Преобразование Кларк переводит 3-фазную систему $u_a, u_b, u_c$ и $i_a, i_b, i_c$ в 2-фазную неподвижную систему $\alpha\beta$. Это позволяет вычислять мгновенную активную мощность $p(t)$ **на каждом отсчете АЦП** (без ожидания полного периода $20\text{ мс}$!):

1. **Преобразование напряжений и токов:**
   $$u_\alpha(t) = \frac{2}{3} \left( u_a(t) - 0.5 u_b(t) - 0.5 u_c(t) \right)$$
   $$u_\beta(t) = \frac{2}{3} \left( \frac{\sqrt{3}}{2} u_b(t) - \frac{\sqrt{3}}{2} u_c(t) \right)$$
   *(аналогично для токов $i_\alpha(t), i_\beta(t)$)*

2. **Мгновенная активная мощность:**
   $$p_{inst}(t) = u_\alpha(t) \cdot i_\alpha(t) + u_\beta(t) \cdot i_\beta(t)$$

3. **Критерий БАВР:**
   If $p_{inst}(t) < -P_{thresh}$ в течение $2 \dots 3\text{ мс} \implies$ **Мгновенный пуск БАВР (потеря питания)**.

---

### 🧪 Научный алгоритм №2: Скорость изменения фазового сдвига ($d\theta_{UI}/dt$)
**Суть:** В нормальном режиме фазовый сдвиг между напряжением и током ввода $\theta_{UI} = \arg(\mathbf{I}) - \arg(\mathbf{U})$ стабилен. При отключении сетевого напряжения и выбеге асинхронных двигателей угол начинает **стремительно возрастать**:

$$\omega_{shift}(t) = \frac{d}{dt} \left( \arg(\mathbf{I}_1(t)) - \arg(\mathbf{U}_1(t)) \right)$$

**Критерий БАВР:**
If $\omega_{shift}(t) > \omega_{thresh}$ ($> 15 \dots 30\text{ град/с}$) $\implies$ **Пуск БАВР по выбегу двигателей**.

---

### 🧪 Научный алгоритм №3: Алгоритм мгновенного интеграла энергии скользящего окна ($W_{window}$)
**Суть:** Для отстройки от кратковременных высокочастотных помех рассчитывается интеграл активной энергии за короткое скользящее окно $T_{win} = 3 \dots 5\text{ мс}$:

$$W_{win}(t) = \int_{t - T_{win}}^{t} p_{inst}(\tau) \, d\tau$$

If $W_{win}(t) < 0 \implies$ Подтвержденный подпиточный ток от двигателей в сторону сети.

---

## 3. Дополнительный Python-код для `public_algorithms.py`

Добавим эти безинерционные научные алгоритмы БАВР в единый модуль для тестирования в Phase 5:

```python
# =============================================================================
# 7. НАУЧНО-ИССЛЕДОВАТЕЛЬСКИЕ БЕЗИНЕРЦИОННЫЕ АЛГОРИТМЫ БАВР
# =============================================================================

class InstantaneousClarkeFABT_PDR(BasePDRAlgorithm):
    """
    Научный алгоритм БАВР на основе мгновенной мощности в alpha-beta (Clarke).
    Вычисляется мгновенно на каждом отсчете без задержки на БПФ (за 1-3 мс).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "p_reverse_thresh_w": -50.0,  # Порог обратной мгновенной мощности (Вт)
            "u_min_instant_v": 5.0
        }
        if params:
            default_params.update(params)
        super().__init__("Instantaneous Clarke (alpha-beta) FABT PDR", default_params)

    def calculate_instantaneous(
        self,
        u_abc_inst: Tuple[float, float, float],
        i_abc_inst: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """
        Принимает МГНОВЕННЫЕ значения напряжений (Вольты) и токов (Амперы) 
        в текущий момент времени t.
        """
        ua, ub, uc = u_abc_inst
        ia, ib, ic = i_abc_inst

        # Преобразование Кларк (Clarke Transformation)
        u_alpha = (2.0 / 3.0) * (ua - 0.5 * ub - 0.5 * uc)
        u_beta = (2.0 / 3.0) * ((math.sqrt(3) / 2.0) * ub - (math.sqrt(3) / 2.0) * uc)

        i_alpha = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
        i_beta = (2.0 / 3.0) * ((math.sqrt(3) / 2.0) * ib - (math.sqrt(3) / 2.0) * ic)

        # Мгновенная активная мощность p(t)
        p_inst = u_alpha * i_alpha + u_beta * i_beta
        p_thresh = self.params["p_reverse_thresh_w"]

        if p_inst < p_thresh:
            direction = DirectionResult.REVERSE  # Разворот мощности -> Пуск БАВР
            bavr_permit = True
        else:
            direction = DirectionResult.FORWARD
            bavr_permit = False

        return {
            "direction": direction,
            "bavr_permit": bavr_permit,
            "p_instantaneous_w": p_inst,
            "u_alpha": u_alpha,
            "u_beta": u_beta
        }


class PhaseAngleVelocityFABT_PDR(BasePDRAlgorithm):
    """
    Научный алгоритм БАВР по скорости изменения фазового сдвига d(theta)/dt.
    Фиксирует момент начала выбега электродвигателей.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "w_threshold_deg_per_sec": 20.0, # Порог скорости ухода угла
            "dt_sec": 0.001                  # Шаг дискретизации (1 мс)
        }
        if params:
            default_params.update(params)
        super().__init__("Phase Angle Velocity (dTheta/dt) FABT PDR", default_params)

    def calculate_velocity(
        self,
        u_complex_1: complex,
        i_complex_1: complex,
        prev_angle_diff_deg: float
    ) -> Dict[str, Any]:
        """
        Расчет скорости сдвига фазы между током и напряжением прямой последовательности.
        """
        current_angle_diff = math.degrees(cmath.phase(i_complex_1) - cmath.phase(u_complex_1))
        current_angle_diff = (current_angle_diff + 180.0) % 360.0 - 180.0

        dt = self.params["dt_sec"]
        d_theta_dt = (current_angle_diff - prev_angle_diff_deg) / dt

        w_thresh = self.params["w_threshold_deg_per_sec"]
        bavr_trigger = abs(d_theta_dt) > w_thresh

        return {
            "bavr_trigger": bavr_trigger,
            "d_theta_dt_deg_per_sec": d_theta_dt,
            "current_angle_diff_deg": current_angle_diff
        }
```

---

### Резюме для вашего отчета:
1. **Основное ядро исследования БАВР**: Алгоритмы Механотроники (БМРЗ), ЭКРА, Бреслер + мгновенный научный алгоритм Кларк ($\alpha\beta$).
2. **Дополнительный контур (РЗА)**: Алгоритмы SEL (32P/67P), Siemens (32), ABB (DOPPDPR) — используются как проверочные кандидаты для оценки их применимости в задачах БАВР.