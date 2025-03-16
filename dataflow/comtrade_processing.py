class ReadComtrade
  def init(self, file_name):
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
        except Exception as ex:
            self.unread_files.add((file_name, ex))
        return raw_date, raw_df
    
   def get_bus_names(self, analog=True, discrete=False):

   def get_all_names(self,):

   def get_ml_signals
