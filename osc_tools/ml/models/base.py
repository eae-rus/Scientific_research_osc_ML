import torch
import torch.nn as nn
import os

class BaseModel(nn.Module):
    """
    Базовый класс для всех моделей в проекте.
    Обеспечивает единый интерфейс для сохранения/загрузки и предсказания.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        """
        Метод для инференса (без градиентов).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def save(self, path):
        """
        Сохранение весов модели.
        """
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        """
        Загрузка весов модели.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
