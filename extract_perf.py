import pandas as pd
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError


class Extract_Performance:
    def __init__(self, filename : str, morphing : bool, train : pd.DataFrame, alg : str = "") -> None:
        self.filename = filename
        self.morphing = morphing
        self.alg = alg
        self.train = train
    
    def fit_transform(self) -> None:
        
        if self.morphing:
            data = pd.read_csv(f"./results/{self.alg}/forecast/{self.filename}")
            error = {"id": list(),
            self.alg: list(),}
        else:
            data = pd.read_csv("./data/forecast/"+self.filename)
            error = {"id": list(),
            "LSTM": list(),
            "Informer": list(),
            "NHITS": list()}
        groups = data.groupby("id")

        if self.morphing:
            for i in data["id"].unique():
                i_train=f"S2T_{i}"
                source = groups.get_group(i).loc[groups.get_group(i)["model"] == self.alg.lower(),:].iloc[:,2:].values[0]
                test = groups.get_group(i).loc[groups.get_group(i)["model"] == "test",:].iloc[:,2:].values[0]
                y_train = self.train.iloc[:735,:]
                y_train = y_train.loc[:,i_train].values
                error["id"].append(i)
                mase = MeanAbsoluteScaledError()
                error[self.alg].append(mase(test, source, y_train=y_train))
            error_perf = pd.DataFrame(error)
            error_perf.to_csv(f"./results/{self.alg}/performance/mase_{self.filename}", index=False)
        else:
            for i in data["id"].unique():
                lstm = groups.get_group(i).loc[groups.get_group(i)["model"] == "lstm",:].iloc[:,2:].values[0]
                nhits = groups.get_group(i).loc[groups.get_group(i)["model"] == "nhits",:].iloc[:,2:].values[0]
                informer = groups.get_group(i).loc[groups.get_group(i)["model"] == "informer",:].iloc[:,2:].values[0]
                test = groups.get_group(i).loc[groups.get_group(i)["model"] == "test",:].iloc[:,2:].values[0]
                y_train = self.train.iloc[:735,:]
                y_train = y_train.loc[:,i].values
                mase = MeanAbsoluteScaledError()
                error["id"].append(i)
                error["LSTM"].append(mase(test,lstm, y_train=y_train))
                error["Informer"].append(mase(test,informer, y_train=y_train))
                error["NHITS"].append(mase(test,nhits, y_train=y_train))
            error_perf = pd.DataFrame(error)
            
            error_perf.to_csv("./data/performance/mase_"+self.filename, index=False)
            
        