import pandas as pd
from pycatch22 import catch22_all

class MFE:
    def __init__(self, series: pd.DataFrame) -> None:
        self.series = series
    
    @property
    def catch22(self) -> pd.DataFrame:
        metafeatures = list()
        for i in range(self.series.shape[1]):
            if i == 0:
                metafeatures.append(catch22_all(self.series.iloc[:, i],catch24=False, short_names=True)["short_names"])
                metafeatures.append(catch22_all(self.series.iloc[:, i],catch24=False, short_names=True)["values"])
            else:
                metafeatures.append(catch22_all(self.series.iloc[:, i],catch24=False, short_names=True)["values"])
        return pd.DataFrame(metafeatures[1:], columns=metafeatures[0])


    
    
    

