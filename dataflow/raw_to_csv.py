from dataflow.read_comtrade import ReadComtrade


class RawToCSV():
    """
    This class implemented to convert raw comtrade files to csv file.
    """
    def __init__(self, raw_path='data/raw/', csv_path='data/csv', norm_coef_file_path='norm_coef.csv', uses_buses = ['1', '2', '12']):
      
    def create_csv(self, csv_name='datset.csv', is_cut_out_area = False):

    def _create_one_df(self, file_path, file_name) -> pd.DataFrame:

    def _split_buses(self, raw_df, file_name):

     def cut_out_area(self, buses_df: pd.DataFrame, samples_before: int, samples_after: int) -> pd.DataFrame:
