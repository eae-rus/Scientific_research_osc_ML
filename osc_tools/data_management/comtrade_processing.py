from osc_tools.core.comtrade_custom import Comtrade
import os
import traceback # Добавляем импорт для подробной трассировки

class ReadComtrade():
    def __init__(self):
        """Инициализатор класса ReadComtrade."""
        # Пока не требуется дополнительная инициализация.
        return None
      
      
    def read_comtrade(self, file_name):
        """Загружает и читает содержимое файлов comtrade.

        Возвращает кортеж (Comtrade, DataFrame) или (None, None) при ошибке.
        """
        try:
            rec = Comtrade()
            # rec.load загружает данные в сам объект rec и не возвращает значения
            rec.load(file_name)

            # После успешной загрузки возвращаем сам объект и DataFrame
            raw_df = rec.to_dataframe()
            return rec, raw_df
        except Exception as ex:
            # Диагностика ошибки
            print("-" * 60)
            print(f"[ERROR] Произошла критическая ошибка при обработке файла: {os.path.basename(file_name)}")
            print(f"Тип ошибки: {type(ex).__name__}")
            print(f"Сообщение: {ex}")
            print("Полная трассировка ошибки:")
            traceback.print_exc()
            print("-" * 60)
            return None, None
