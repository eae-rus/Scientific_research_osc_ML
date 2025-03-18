import comtrade # comtrade 0.1.2

class ReadComtrade():
    def init(self):
        """
        Initialize the class.
        
          Returns:
              None
        """
        pass
      
      
    def read_comtrade(self, file_name):
        """
        Load and read comtrade files contents.

        Args:
            file_name (str): The name of comtrade file.

        Returns:
            tuple:
            - raw_date (Comtrade): The raw comtrade
            - raw_df (pandas.DataFrame): DataFrame of raw comtrade file.
        """
        raw_df = None
        try:
            raw_date = comtrade.load(file_name)
            raw_df = raw_date.to_dataframe()
            return raw_date, raw_df
        except Exception as ex:
            # TODO: Add "self.unread_files.add((file_name, ex))"
            return None, None