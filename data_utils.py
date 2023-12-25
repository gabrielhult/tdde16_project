import os
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data
