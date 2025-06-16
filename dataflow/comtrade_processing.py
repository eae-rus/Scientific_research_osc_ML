from comtrade_APS import Comtrade # comtrade 0.1.2

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
        raw_data_obj = None
        try:
            raw_data_obj = Comtrade()
            raw_data_obj.load(cfg_file=file_name)

            raw_df = raw_data_obj.to_dataframe()
            return raw_data_obj, raw_df
        except Exception as ex:
            print(f"Error reading COMTRADE file {file_name}: {ex}")
            # TODO: Add "self.unread_files.add((file_name, ex))"
            return None, None