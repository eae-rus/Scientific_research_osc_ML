import comtrade # comtrade 0.1.2

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
            raw_date = comtrade.load(file_name)
            raw_df = raw_date.to_dataframe()
            return raw_date, raw_df
        except Exception as ex:
            # TODO: Добавить "self.unread_files.add((file_name, ex))"
            return None, None