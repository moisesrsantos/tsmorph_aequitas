import numpy as np
import pandas as pd
from base_models import Base_Models
from multiprocessing import Pool
import psutil
import sys


filename = sys.argv[1]
alg = sys.argv[2]
nn5 = pd.read_csv(f"./results/{alg}/morph/{filename}")
nn5_train = nn5.iloc[:735,:]
nn5_test = nn5.iloc[735:,:]


def forecast(i):
    train = nn5_train.iloc[:,i]
    test = nn5_test.iloc[:,i].to_list()
    h = 56 
    b = Base_Models(train=train, h=h)
    if alg == "LSTM":
        return [[i, "lstm"]+b.lstm(),[i, "test"]+test]
    if alg == "Informer":
        return [[i, "informer"]+b.informer(),[i, "test"]+test]
    if alg == "NHITS":
        return [[i, "nhits"]+b.nhits(),[i, "test"]+test]

if __name__ == '__main__':
    
    cpus = psutil.cpu_count()
    with Pool(cpus-1) as p:
        forecasting = p.map(forecast, range(nn5_train.shape[1]))
    
    col = ["id", "model", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
           "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37",
           "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56"]
    flat_list = [item for sublist in forecasting for item in sublist]
    df_for = pd.DataFrame(flat_list, columns=col).set_index('id')
    df_for.to_csv(f"./results/{alg}/forecast/{filename}")
