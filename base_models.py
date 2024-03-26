import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.models import LSTM, NHITS, Informer


class Base_Models:

    def __init__(self, train, h) -> None:
        self.train = train
        self.h = h

    def lstm(self):
        Y_df = self.train.reset_index()
        Y_df['unique_id'] = 1.
        Y_df = Y_df.rename(columns={'index': 'ds', Y_df.iloc[:,1].name: 'y'})
        Y_df = Y_df[['unique_id', 'ds', 'y']]
        model = LSTM(h=self.h, max_steps=500)
        nf = NeuralForecast(models=[model], freq='D')
        nf.fit(df=Y_df)
        return nf.predict()["LSTM"].to_list()
    
    def nhits(self):
        Y_df = self.train.reset_index()
        Y_df['unique_id'] = 1.
        Y_df = Y_df.rename(columns={'index': 'ds', Y_df.iloc[:,1].name: 'y'})
        Y_df = Y_df[['unique_id', 'ds', 'y']]
        model = NHITS(h=self.h, max_steps=500, input_size=2 * self.h)
        nf = NeuralForecast(models=[model], freq='D')
        nf.fit(df=Y_df)
        return nf.predict()["NHITS"].to_list()
    
    def informer(self):
        Y_df = self.train.reset_index()
        Y_df['unique_id'] = 1.
        Y_df = Y_df.rename(columns={'index': 'ds',  Y_df.iloc[:,1].name: 'y'})
        Y_df = Y_df[['unique_id', 'ds', 'y']]
        model = Informer(h=self.h, max_steps=500,input_size=2 * self.h)
        nf = NeuralForecast(models=[model], freq='D')
        nf.fit(df=Y_df)
        return nf.predict()["Informer"].to_list()
    
