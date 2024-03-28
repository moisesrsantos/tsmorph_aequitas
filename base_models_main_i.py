import numpy as np
import pandas as pd
from base_models import Base_Models
from multiprocessing import Pool
from sklearn.impute import SimpleImputer
import psutil
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


nn5 = pd.read_csv("./data/NN5_preproc.csv")
nn5_train = nn5.iloc[:735,:]
nn5_test = nn5.iloc[735:,:]


def forecast(i):
    train = nn5_train.iloc[:,i]
    test = nn5_test.iloc[:,i].to_list()
    name = train.name
    h = 56 
    b = Base_Models(train=train, h=h)
    return [[name, "informer"]+b.informer(),[name, "test"]+test]
    #return [[name, "lstm"]+b.lstm(),[name, "test"]+test]


if __name__ == '__main__':
    cpus = psutil.cpu_count()
    with Pool(cpus-1) as p:
        forecasting = p.map(forecast, range(nn5_train.shape[1]))
    
    col = ["id", "model", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
           "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37",
           "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56"]
    flat_list = [item for sublist in forecasting for item in sublist]
    df_for = pd.DataFrame(flat_list, columns=col).set_index('id')
    df_for.to_csv("./data/forecast/NN5_i.csv")


