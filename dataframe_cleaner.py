import pandas as pd

class CleanDataframe:
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    #this method can clean a dataframe and drop the bad and missing data
    def clean_dataframe(self):
        
        # clean duplicated values
        self.dataframe = self.dataframe.drop_duplicates()
        
        # clean null values
        self.dataframe = self.dataframe.dropna()

        # clean nan values
        self.dataframe = self.dataframe.dropna(how = 'all')

        return self.dataframe
    
    #this method brings to the df columns name a standard configuration and structure
    def name_formater(self):

        #clean the names and get the best one's
        self.dataframe.columns = [x.lower().replace(" ","_").replace("?","")\
        .replace("-","_").replace(r"/","_").replace("\\","_").replace("%",'')\
        .replace(")","").replace(r"(","").replace("$","") for x in self.dataframe.columns]

    