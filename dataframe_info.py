import pandas as pd

class InfoDataframe:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    # this method bring us the column names to pass trought gpt-3    
    def get_column_names(self):
        return list(self.dataframe.columns)
    

    def remove_categorical_columns(self):
        # Seleccionar todas las columnas que contienen valores categóricos
        cat_cols = self.dataframe.select_dtypes(include=['object']).columns

        # Eliminar las columnas seleccionadas del DataFrame
        df = self.dataframe.drop(columns=cat_cols)
        
        return df
