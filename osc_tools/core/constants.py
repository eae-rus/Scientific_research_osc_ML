from enum import Enum

# This file will contain shared constants for the project.

class TYPE_OSC(Enum):
        # TODO: написать полноценные описание типов
        COMTRADE_CFG_DAT = "Файлы типа Comtrade, состоящие из двух файлов .cfg и .dat"
        COMTRADE_CFF = "Файлы типа Comtrade, состоящие из одного файла .cff"
        BRESLER = "Тип файла (.brs), характерный для производителя терминалов ООО НПП 'Бреслер', официальная ссылка: https://www.bresler.ru/"
        BLACK_BOX = "черный ящик (.bb)" #??
        RES_3 = "Тип файла (.sg2), характерный для производителя терминалов ООО 'Прософт-Системы', официальная ссылка: https://prosoftsystems.ru/ и ссылка на устройства https://prosoftsystems.ru/catalog/show/cifrovoj--registrator-jelektricheskih-sobytij-pjes3"
        EKRA = "Тип файла (.dfr), характерный для производителя терминалов ООО НПП 'ЭКРА', официальная ссылка: https://ekra.ru/"
        PARMA = "Тип файла (.do), характерный для производителя терминалов ООО 'ПАРМА', официальная ссылка: https://parma.spb.ru/"
        PARMA_TO = "Тип файла (.to), характерный для производителя терминалов ООО 'ПАРМА', этот тип предназначен для регистрации длительных процессов, официальная ссылка: https://parma.spb.ru/"
        NEVA = "Тип файла (.os1 и аналогичные), характерный для производителя терминалов НПФ 'ЭНЕРГОСОЮЗ', официальная ссылка: https://www.energosoyuz.spb.ru/ru и ссылка на устройства https://www.energosoyuz.spb.ru/ru/content/registrator-avariynyh-sobytiy-neva-ras"
        OSC = "Формат файла (.osc) - контактная привязка еще не до конца выяснена"


class Features:
    # TODO: подумать о том, чтобы сделать отдельными параметрами для более удобного отслеживания, так как некоторые параметры были временны. 
    
    # Индексные сопоставления из model.py
    CURRENT_INDICES = {"IA": 0, "IB": 1, "IC": 2, "IN": -1}
    VOLTAGE_PHAZE_BB_INDICES = {"UA BB" : 3, "UB BB" : 4, "UC BB" : 5, "UN BB" : 6}
    VOLTAGE_PHAZE_CL_INDICES = {"UA CL" : 7, "UB CL" : 8, "UC CL" : 9, "UN CL" : 10}
    VOLTAGE_LINE_BB_INDICES = {"UAB BB" : -1,"UBC BB" : -1,"UCA BB" : -1}
    VOLTAGE_LINE_CL_INDICES = {"UAB CL" : 11,"UBC CL": 12,"UCA CL": 13}

    # Списки имен из train.py и marking_up_oscillograms.py
    # TODO: Подумать на будущее, надо будет как-то аккуратнее это использовать. Данное иземенение вроде и хорошее, но
    # могут быть несостыковки по требуемому перечню
    CURRENT = ["IA", "IB", "IC"]
    VOLTAGE_PHAZE_BB = ["UA BB", "UB BB", "UC BB"]
    VOLTAGE_PHAZE_CL = ["UA CL", "UB CL", "UC CL"]
    VOLTAGE_ZERO_SEQ = ["UN BB", "UN CL"]
    VOLTAGE_LINE_BB = ["UAB BB", "UBC BB", "UCA BB"] # пока не используется
    VOLTAGE_LINE_CL = ["UAB CL", "UBC CL", "UCA CL"]

    VOLTAGE = VOLTAGE_PHAZE_BB + VOLTAGE_PHAZE_CL + VOLTAGE_ZERO_SEQ + VOLTAGE_LINE_CL # пока без VOLTAGE_LINE_BB, так как не использовался.

    ALL = CURRENT + VOLTAGE

    # Целевые признаки
    TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]

    # Признаки для графиков
    CURRENT_FOR_PLOT = ["IA", "IB", "IC"]
    VOLTAGE_FOR_PLOT = ['UA BB', 'UB BB', 'UC BB']
    ANALOG_SIGNALS_FOR_PLOT = CURRENT_FOR_PLOT + VOLTAGE_FOR_PLOT


class PDRFeatures:
    CURRENT_1 = ["I_pos_seq_mag", "I_pos_seq_angle"]
    CURRENT_2 = ["I_neg_seq_mag", "I_neg_seq_angle"]
    VOLTAGE_1 = ["V_pos_seq_mag"]
    VOLTAGE_2 = ["V_neg_seq_mag", "V_neg_seq_angle"]
    POWER_1 = ["P_pos_seq", "Q_pos_seq"]
    POWER_2 = ["P_neg_seq", "Q_neg_seq"]
    IMPEDANCE = ["Z_pos_seq_mag", "Z_pos_seq_angle"]

    # Для модели 1
    ALL_MODEL_1 = CURRENT_1 + VOLTAGE_1

    # Для модели 2 (закомментировано в оригинале)
    # ALL_MODEL_2 = CURRENT_1 + CURRENT_2 + VOLTAGE_1 + VOLTAGE_2 + POWER_1 + POWER_2 + IMPEDANCE

    TARGET_TRAIN = ["rPDR PS"]
    TARGET_TEST = ["iPDR PS"]
