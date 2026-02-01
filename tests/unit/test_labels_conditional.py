import polars as pl

from osc_tools.ml.labels import prepare_labels_for_experiment, get_target_columns


def test_base_sequential_labels_mapping():
    """Проверка формирования базовых меток с ограничением ML_2=0 при ML_3=1."""
    df = pl.DataFrame(
        {
            "ML_1_1": [0, 1, 0, 0, 0],
            "ML_2_1": [0, 0, 1, 0, 1],
            "ML_3_2": [0, 0, 0, 1, 1],
        }
    )

    df = prepare_labels_for_experiment(df, target_level="base_sequential")
    target_cols = get_target_columns("base_sequential")

    assert target_cols == ["Target_Normal", "Target_ML_1", "Target_ML_2", "Target_ML_3"]

    normal = df["Target_Normal"].to_list()
    ml1 = df["Target_ML_1"].to_list()
    ml2 = df["Target_ML_2"].to_list()
    ml3 = df["Target_ML_3"].to_list()

    # 0: нет событий
    assert normal[0] == 1
    assert ml1[0] == 0
    assert ml2[0] == 0
    assert ml3[0] == 0

    # 1: ML_1
    assert normal[1] == 0
    assert ml1[1] == 1
    assert ml2[1] == 0
    assert ml3[1] == 0

    # 2: ML_2
    assert normal[2] == 0
    assert ml1[2] == 0
    assert ml2[2] == 1
    assert ml3[2] == 0

    # 3: ML_3
    assert normal[3] == 0
    assert ml1[3] == 0
    assert ml2[3] == 0
    assert ml3[3] == 1

    # 4: ML_2 и ML_3 одновременно -> ML_2 должен быть 0
    assert normal[4] == 0
    assert ml1[4] == 0
    assert ml2[4] == 0
    assert ml3[4] == 1
