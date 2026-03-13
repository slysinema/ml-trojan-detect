import pandas as pd

class ReadDataset:
    """
    Class for read parquet datasets
    Args:
        path - path to parquet dataset
    """
    def __init__(self, path: str):
        self.path = path

    def read_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        return df
