import subprocess
import random
import os
import pandas as pd
import numpy as np
import sys
import threading
import math
import queue

# Ejecutar { time python3 parameterBenchmarking.py > output.txt; } 2> time.txt &
#
# Para tener calculo de tiempo de cómputo y tambien almacenar algún error en output.txt en caso de haber

# Nota: Este codigo crea muchos .csv para almacenar cómputos intermedios, pero los elimina al final


def get_parameter_sets(time):
    parameter_sets = dict()
    parameter_sets["KNN1"] = pd.read_csv(f"predicted_params/KNN1_predicted_parameters_T{time}.csv")
    parameter_sets["KNN3"] = pd.read_csv(f"predicted_params/KNN3_predicted_parameters_T{time}.csv")
    parameter_sets["KNN5"] = pd.read_csv(f"predicted_params/KNN5_predicted_parameters_T{time}.csv")
    parameter_sets["KNN7"] = pd.read_csv(f"predicted_params/KNN7_predicted_parameters_T{time}.csv")
    parameter_sets["KNN9"] = pd.read_csv(f"predicted_params/KNN9_predicted_parameters_T{time}.csv")
    parameter_sets["KNN11"] = pd.read_csv(f"predicted_params/KNN11_predicted_parameters_T{time}.csv")
    parameter_sets["NeuralNetwork"] = pd.read_csv(f"predicted_params/NeuralNetwork_predicted_parameters_T{time}.csv")

    parameter_table = [[10,20,40,70,100],
                       [0.1,0.2,0.4,0.6,0.8],
                       [0.1,0.2,0.4,0.6,0.8],
                       [5,15,25,35,45],
                       [0.1,0.2,0.4,0.6,0.8]]

    #Construimos parametros random
    parameter_sets["Random"] = list() 
    for instance in parameter_sets["KNN1"]["instance"]:
        random_vector = (np.random.rand(5) * 5).astype(int)
        random_parameters = [parameter_table[i][random_vector[i]] for i in range(5)]
        parameter_sets["Random"].append(random_parameters)
    parameter_sets["Random"] = pd.concat([parameter_sets["KNN1"]["instance"],pd.DataFrame(parameter_sets["Random"],columns=["gs","nc","ne","ps","xi"])],axis=1)

    #Construimos parametros base
    parameter_sets["Base"] = [[i,40,0.2,0.4,25,0.2] for i in parameter_sets["KNN1"]["instance"]]
    parameter_sets["Base"] = pd.DataFrame(parameter_sets["Base"],columns=["instance","gs","nc","ne","ps","xi"])
    return parameter_sets

#Esta función es llamada con multiples hebras
def run_thread(task_group, core,time,seed):
    results = list()
    for task in task_group:
        command = f"python3 SolveVRP.py -i Homberger/{task[1]} -ps {int(task[5])} -gs {int(task[2])} -ne {task[4]} -nc {task[3]} -xi {task[6]} -t {time} -s {seed}"
        result = subprocess.check_output(command, shell=True, stderr=subprocess.PIPE, text=True)
        if result == "inf\n":
            result = "99999\n"
        results.append(task + [result])
    f = open(f"thread_execs/output_thread_{core}_time_{time}_seed_{seed}.csv","w")
    for result in results:
        f.write(",".join(map(str,result)))
    f.close()

# 1 seed, 1 sec toma 3.5 minutos
# 1 seed, 10 sec debería tomar 35 min
# 1 seed, 30 sec debería tomar 105 min
time_limits = [10,30,60]
seeds = [1,2,3,4,5]
NTHREADS = 40

for time in time_limits:
    parameter_sets = get_parameter_sets(time)
    # Creamos pool de ejecuciones para que los hilos vayan tomando de aqui
    tasks = list()
    for parameter_set,df in parameter_sets.items():
        for i in range(len(df)):
            tasks.append([parameter_set] + list(df.iloc[i]))
    for seed in seeds:
        print(f"Running seed {seed}, time {time}")
        threads = list()
        for core in range(NTHREADS):
            task_group_size = math.ceil(len(tasks)/NTHREADS) #Dividimos las instancias en grupos de igual tamaño (salvo por el último probablemente)
            task_group = tasks[core * task_group_size : min(core * task_group_size + task_group_size,len(tasks))] #Aquí min para no pasarnos del índice
            x = threading.Thread(target=run_thread, args=(task_group,core,time,seed))
            threads.append(x)
            x.start()
        for index, thread in enumerate(threads):
            thread.join() 


# Combinamos los resultados obtenidos por todos los threads, seeds y tiempos
files = os.listdir("thread_execs")
files = [f for f in files if ".csv" in f]
threads = [i for i in range(NTHREADS)]
#Obtenemos la eval promedio entre seeds
for thread in threads:
    for time in time_limits:
        filtered_files = [f for f in files if (f"thread_{thread}_time_{time}" in f)]
        evals = list()
        for f in filtered_files:
            df = pd.read_csv(f"thread_execs/{f}",names=["set","instance","gs","nc","ne","ps","xi","eval"])
            evals.append(np.array(df["eval"]))
        mean_evals = np.mean(evals,axis=0)
        mean_df = pd.read_csv(f"thread_execs/{f}",names=["set","instance","gs","nc","ne","ps","xi","eval"])
        mean_df["eval"] = mean_evals
        mean_df.to_csv(f"thread_execs/results/thread_{thread}_time_{time}.csv",header=False,index=False)

#Juntamos todos los resultados de cada time execution
files = os.listdir("thread_execs/results")
results = {time:pd.DataFrame(columns=["set","instance","gs","nc","ne","ps","xi","eval"]) for time in time_limits}
for time in time_limits:
    filtered_files = [f for f in files if (f"time_{time}" in f)]
    for f in filtered_files:
        df = pd.read_csv(f"thread_execs/results/{f}",names=["set","instance","gs","nc","ne","ps","xi","eval"])
        results[time] = pd.concat([results[time],df], ignore_index=True)
    results[time] = results[time].sort_values(by="set").reset_index(drop=True)
    results[time].to_csv(f"thread_execs/results/results_time_{time}.csv",index=False,header=False)

#Limpiamos la carpeta results de archivos intermedios
files = os.listdir("thread_execs/results")
files = [f for f in files if "results" not in f]  
for f in files:
    os.remove(f"thread_execs/results/{f}")

#Aggregamos los resultados de cada tiempo según categoría de instancia (tabla final de memoria)
files = [f for f in os.listdir("thread_execs/results") if ".csv" in f]
instance_groups = ["C1","C2","R1","R2","RC1","RC2"]
for f in files:
    results_per_group = pd.DataFrame(columns=["set","type","mean_eval"])
    df = pd.read_csv(f"thread_execs/results/{f}",names=["set","instance","gs","nc","ne","ps","xi","eval"])
    sets = list(df["set"].unique())
    for s in sets:
        set_df = df.loc[df["set"] == s]
        for group in instance_groups:
            group_df = set_df.loc[set_df["instance"].str.contains(group)]
            group_mean = group_df["eval"].mean()
            results_per_group.loc[len(results_per_group.index)] = [s,group,int(group_mean)]
    results_per_group.to_csv(f"thread_execs/results/{f}",index=False,header=False)

#Finalmente, eliminamos todos los outputs individuales de threads
files = os.listdir("thread_execs")
files = [f for f in files if ".csv" in f]
for f in files:
    os.remove(f"thread_execs/{f}")






