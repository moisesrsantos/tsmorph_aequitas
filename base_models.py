import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.models import LSTM, DeepAR

class Base_Models:

    def __init__(self, train, h) -> None:
        self.train = train
        self.h = h

    def lstm(self):
        Y_df = self.train.reset_index()
        Y_df['unique_id'] = 1.
        Y_df = Y_df.rename(columns={'index': 'ds', Y_df.iloc[:,1].name: 'y'})
        Y_df = Y_df[['unique_id', 'ds', 'y']]
        model = LSTM(h=self.h, max_steps=100, accelerator='cpu', enable_checkpointing=False, logger=False, random_seed=14)
        nf = NeuralForecast(models=[model], freq=1)
        nf.fit(df=Y_df)
        return nf.predict()["LSTM"].to_list()
    
    def deepar(self):
        Y_df = self.train.reset_index()
        Y_df['unique_id'] = 1.
        Y_df = Y_df.rename(columns={'index': 'ds', Y_df.iloc[:,1].name: 'y'})
        Y_df = Y_df[['unique_id', 'ds', 'y']]
        model = DeepAR(h=self.h, max_steps=100, input_size=2 * self.h, scaler_type='robust', accelerator='cpu', enable_checkpointing=False, logger=False, random_seed=14)
        nf = NeuralForecast(models=[model], freq=1)
        nf.fit(df=Y_df)
        return nf.predict()["DeepAR"].to_list()
    
