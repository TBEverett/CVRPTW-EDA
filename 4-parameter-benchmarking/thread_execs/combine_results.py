import os
import pandas as pd
import numpy as np

files = os.listdir(".")
files = [f for f in files if "output" in f]
seeds = [1,2,3,4,5]
times = [10,30]
threads = list(range(40))
#Obtenemos la eval promedio entre seeds
for thread in threads:
    for time in times:
        filtered_files = [f for f in files if (f"thread_{thread}_time_{time}" in f)]
        evals = list()
        for f in filtered_files:
            df = pd.read_csv(f,names=["set","instance","gs","nc","ne","ps","xi","eval"])
            evals.append(np.array(df["eval"]))
        mean_evals = np.mean(evals,axis=0)
        mean_df = pd.read_csv(f,names=["set","instance","gs","nc","ne","ps","xi","eval"])
        mean_df["eval"] = mean_evals
        mean_df.to_csv(f"results/thread_{thread}_time_{time}.csv",header=False,index=False)

#Juntamos todos los resultados de cada time execution
files = os.listdir("results")
results = {time:pd.DataFrame(columns=["set","instance","gs","nc","ne","ps","xi","eval"]) for time in times}
for time in times:
    filtered_files = [f for f in files if (f"time_{time}" in f)]
    for f in filtered_files:
        df = pd.read_csv(f"results/{f}",names=["set","instance","gs","nc","ne","ps","xi","eval"])
        results[time] = pd.concat([results[time],df], ignore_index=True)
    results[time] = results[time].sort_values(by="set").reset_index(drop=True)
    results[time].to_csv(f"results/results_time_{time}.csv",index=False,header=False)
for f in files:
    os.remove(f"results/{f}")

#Aggregamos los resultados de cada tiempo según categoría de instancia (tabla final de memoria)
files = [f for f in os.listdir("results") if ".csv" in f]
instance_groups = ["C1","C2","R1","R2","RC1","RC2"]
for f in files:
    results_per_group = pd.DataFrame(columns=["set","type","mean_eval"])
    df = pd.read_csv(f"results/{f}",names=["set","instance","gs","nc","ne","ps","xi","eval"])
    sets = list(df["set"].unique())
    for s in sets:
        set_df = df.loc[df["set"] == s]
        for group in instance_groups:
            group_df = set_df.loc[set_df["instance"].str.contains(group)]
            group_mean = group_df["eval"].mean()
            results_per_group.loc[len(results_per_group.index)] = [s,group,int(group_mean)]
    os.remove(f"results/{f}")
    results_per_group.to_csv(f"results/{f}",index=False,header=False)
exit()
