from osc_tools.core.comtrade_custom import Comtrade

class ReadComtrade():
    def init(self):
        """
        Инициализирует класс.
        
          Возвращает:
              None
        """
        pass
      
      
    def read_comtrade(self, file_name):
        """
        Загружает и читает содержимое файлов comtrade.

        Аргументы:
            file_name (str): Имя файла comtrade.

        Возвращает:
            tuple:
            - raw_date (Comtrade): Необработанные данные comtrade
            - raw_df (pandas.DataFrame): DataFrame необработанного файла comtrade.
        """
        raw_df = None
        try:
            rec = Comtrade()
            raw_date = rec.load(file_name)
            raw_df = raw_date.to_dataframe()
            return raw_date, raw_df
        except Exception as ex:
            # TODO: Добавить "self.unread_files.add((file_name, ex))"
            return None, None