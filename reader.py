import pandas as pd

class DataReader:
    def load(filename, delimitier=','):
        return pd.read_csv(filename, delimiter=delimitier)
    
    def load_data_with_labels(filename, label_column, delimiter=','):
        data = pd.read_csv(filename, delimiter=delimiter)
        return data.drop(label_column, axis=1), data[label_column]
