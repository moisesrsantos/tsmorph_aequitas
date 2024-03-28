import os
from extract_perf import Extract_Performance
import pandas as pd
from sklearn.impute import SimpleImputer
from mfe import MFE
import numpy as np
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

nn5 = pd.read_csv("./data/NN5.csv", delimiter=";", decimal=",")
nn5 = nn5.iloc[:,:]
nn5 = nn5.drop(["Time Series #ID"], axis=1)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
nn5_imp = imp_mean.fit_transform(nn5)
nn5 = pd.DataFrame(nn5_imp, index=nn5.index, columns=nn5.columns)
nn5.to_csv("./data/NN5_preproc.csv", index=None)

os.system('python base_models_main_ni.py')


Extract_Performance(filename="NN5_ni.csv", morphing=False, train=nn5).fit_transform()
performance = pd.read_csv("./data/performance/mase_nn5_ni.csv", index_col="id")

