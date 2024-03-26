import pandas as pd
import tsfel
from pycatch22 import catch22_all

class MFE:
    def __init__(self, series: pd.DataFrame) -> None:
        self.series = series
    
    @property
    def tsfel(self) -> pd.DataFrame:
        metafeatures = list()
        cfg = tsfel.get_features_by_domain()
        for i in range(self.series.shape[1]):
            if i == 0:
                mf = tsfel.time_series_features_extractor(cfg, self.series.iloc[:, i], header_names = None, verbose = 0)
                mf.columns = mf.columns.str.lstrip("0_")
                metafeatures.append(mf.columns.to_list())
                metafeatures.append(mf.values.reshape(-1).tolist())
            else:
                mf = tsfel.time_series_features_extractor(cfg, self.series.iloc[:, i], header_names = None, verbose = 0)
                metafeatures.append(mf.values.reshape(-1).tolist())
        return pd.DataFrame(metafeatures[1:], columns=metafeatures[0])
    
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


    
    
    

