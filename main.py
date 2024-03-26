
import os
from tsmorph import TSmorphing
from extract_perf import Extract_Performance
from mfe import MFE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from matplotlib.colors import Normalize

#plt.rcParams.update({'font.size': 40})

# NN5 data
nn5 = pd.read_csv("./data/NN5_preproc.csv")
nn5 = nn5.iloc[:,:]


def plot_results(performances, mf_name, alg):
    l = list_filenames(f"./results/{alg}/mf")
    fe = pd.DataFrame()
    for i in l:
        fe[i.split(".")[0]] = pd.read_csv(f"./results/{alg}/mf/"+i)[mf_name]
    plt.figure()
    min = performances.min().min()
    max = performances.max().max()
    normalizer = Normalize(vmin=min, vmax=max)
    for j in range(fe.shape[1]): 
        scatter = plt.scatter(fe.index, fe.iloc[:,j], c = performances.iloc[:,j], cmap='viridis', norm=normalizer,)
    plt.colorbar(scatter, label='Relative Performance')
    plt.ylabel(mf_name.upper())
    plt.xticks(ticks= np.arange(20))
    plt.xlabel("Morphing Process")
    plt.savefig("./results/{}/img/{}.png".format(alg,mf_name))
    plt.close()
        

def list_filenames(path):
    try:
        files = os.listdir(path)
        list_of_filenames = list()
        for file in files:
            list_of_filenames.append(file)
        return list_of_filenames
    except OSError as e:
        print(f"Error to access the directory: {e}")

def morph_evaluate(alg):
    performance = pd.read_csv("./data/performance/mase_nn5.csv", index_col="id")
    performance = performance[[alg]]  
    perform_source_top = performance.sort_values(by=[alg], ascending=True).iloc[:10,:]
    perform_target_top = performance.sort_values(by=[alg], ascending=False).iloc[:10,:]

    # Create data morphing
    source_list = perform_source_top.index.to_list()
    target_list = perform_target_top.index.to_list()[0]
    print(target_list)


    path = './results/{}'.format(alg)
    if not(os.path.isdir(path)):
            os.mkdir(path)
    
    path = './results/{}/morph'.format(alg)
    if not(os.path.isdir(path)):
            os.mkdir(path)
    
    path = './results/{}/cor'.format(alg)
    if not(os.path.isdir(path)):
            os.mkdir(path)
    
    path = './results/{}/forecast'.format(alg)
    if not(os.path.isdir(path)):
            os.mkdir(path)
    
    path = './results/{}/img'.format(alg)
    if not(os.path.isdir(path)):
            os.mkdir(path)
    
    path = './results/{}/mf'.format(alg)
    if not(os.path.isdir(path)):
            os.mkdir(path)

    path = './results/{}/performance'.format(alg)
    if not(os.path.isdir(path)):
            os.mkdir(path)
    
    for i in range(10):
        morph = TSmorphing(T = nn5.loc[:, target_list], S = nn5.loc[:, source_list[i]], granularity = 20).transform()
        morph.to_csv("./results/{}/morph/nn5_morph_{}_{}.csv".format(alg, source_list[i], target_list), index=False)

    # Base models for morphing
    filenames = os.listdir(f"./results/{alg}/morph/")
    for i in filenames:
        os.system('python base_models_morph.py '+i+' '+alg)
        
    df_cor = pd.DataFrame(columns=["pair", "mf", "correlation"])

    for i in range(10):

        # Extract meta-features 
        morph_n = pd.read_csv("./results/{}/morph/nn5_morph_{}_{}.csv".format(alg,source_list[i], target_list))
        mfe = MFE(series = morph_n.iloc[:735,:]).catch22
        #mfe.columns = mfe.columns.str.lstrip("S2T_0__")
        mfe.to_csv("./results/{}/mf/nn5_morph_{}_{}.csv".format(alg,source_list[i], target_list), index=False)

        # Extract performance for morphing
        Extract_Performance(filename="nn5_morph_{}_{}.csv".format(source_list[i], target_list), morphing=True, alg=alg, train=morph_n).fit_transform()

        # Plot Figures
        perf = pd.read_csv("./results/{}/performance/mase_nn5_morph_{}_{}.csv".format(alg,source_list[i], target_list), index_col="id")
        
    # Calcular correlação entre meta-feature com a diferença de performance
        for j in mfe.columns.to_list():       
            df_cor.loc[df_cor.shape[0]] = ["{}".format(source_list[i], target_list), j, pearsonr(perf[alg],mfe[j])[0]]

    csv_cor = df_cor.drop(["pair"], axis=1).groupby('mf').agg([np.std,np.mean]).sort_values(by=[('correlation','std')], ascending=[True])
    csv_cor.columns = ['_'.join(col) for col in csv_cor.columns.values]
    csv_cor.to_csv("./results/{}/cor/cor.csv".format(alg))
    list_cor = csv_cor.index.to_list()

    l = list_filenames(f"./results/{alg}/performance")
    performances = pd.DataFrame()
    for i in l:
        performances[i.split(".")[0]] = pd.read_csv(f"./results/{alg}/performance/"+i)["LSTM"]

    for j in list_cor:
        plot_results(performances, j, alg)


#algs = ["LSTM", "Informer", "NHITS"]
algs = ["LSTM"]

for alg in algs:
    morph_evaluate(alg=alg)