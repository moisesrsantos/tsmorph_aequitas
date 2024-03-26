import numpy as np
import pandas as pd


class TSmorphing:
    """_summary_
    """
    def __init__(self, S : np.array, T : np.array, granularity : int) -> None:
        """_summary_

        Args:
            S (np.array): _description_
            T (np.array): _description_
            granularity (int): _description_
        """
        self.T = T
        self.S = S
        self.granularity = granularity
    
    def transform(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """

        alpha = np.linspace(0, 1, self.granularity)
        y_morph = dict()
        index = 0
        for i in alpha:
            for j in range(len(self.S)):
                try:
                    y_morph["S2T_{}".format(index)].append(i*self.T[j]+(1-i)*self.S[j])
                except KeyError:
                    y_morph["S2T_{}".format(index)] = list()
                    y_morph["S2T_{}".format(index)].append(i*self.T[j]+(1-i)*self.S[j])
            index+=1
        return pd.DataFrame(y_morph)        

 