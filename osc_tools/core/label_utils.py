from typing import List, Optional

def map_to_parent_class(label: str, trained_labels: List[str]) -> Optional[str]:
    """
    Отображает метку на ближайший родительский класс, который присутствует в списке обученных меток.
    
    Пример:
    trained_labels = ["ML_1", "ML_2", "ML_3_1", "ML_3_2"]
    label = "ML_3_2_1" -> "ML_3_2"
    label = "ML_3_3" -> None (или "ML_3", если он есть в списке)
    
    Args:
        label: Исходная метка (например, "ML_3_2_1").
        trained_labels: Список меток, на которых обучалась модель.
        
    Returns:
        Родительская метка или None, если совпадений не найдено.
    """
    if label in trained_labels:
        return label
        
    parts = label.split('_')
    # Пытаемся сокращать метку с конца
    for i in range(len(parts) - 1, 0, -1):
        parent = "_".join(parts[:i])
        if parent in trained_labels:
            return parent
            
    return None

def get_hierarchical_accuracy(y_true: List[str], y_pred: List[str], trained_labels: List[str]) -> float:
    """
    Рассчитывает точность с учетом иерархии. 
    Предсказание считается верным, если оно совпадает с истинной меткой 
    или ее родителем из списка обученных меток.
    """
    if not y_true:
        return 0.0
        
    correct = 0
    for true, pred in zip(y_true, y_pred):
        mapped_true = map_to_parent_class(true, trained_labels)
        if mapped_true == pred:
            correct += 1
            
    return correct / len(y_true)
