import subprocess
import random
import os
import pandas as pd
import numpy as np
import sys
import threading
import math
import queue

parameter_sets = dict()
parameter_sets["KNN1"] = pd.read_csv("KNN1_predicted_parameters.csv")
parameter_sets["KNN3"] = pd.read_csv("KNN3_predicted_parameters.csv")
parameter_sets["KNN5"] = pd.read_csv("KNN5_predicted_parameters.csv")
parameter_sets["KNN7"] = pd.read_csv("KNN7_predicted_parameters.csv")
parameter_sets["KNN9"] = pd.read_csv("KNN9_predicted_parameters.csv")
parameter_sets["KNN11"] = pd.read_csv("KNN11_predicted_parameters.csv")
parameter_sets["NeuralNetwork"] = pd.read_csv("NeuralNetwork_predicted_parameters.csv")

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

# Creamos pool de ejecuciones para que los hilos vayan tomando de aqui
tasks = list()
for parameter_set,df in parameter_sets.items():
    for i in range(len(df)):
        tasks.append([parameter_set] + list(df.iloc[i]))

#Realizamos benchmarking
def benchmark(task_group, core,time,seed):
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
time_limits = [10,30]
seeds = [2,3,4,5]
NTHREADS = 40
for seed in seeds:
    for time in time_limits:
        print(f"Running seed {seed}, time {time}")
        threads = list()
        for core in range(NTHREADS):
            task_group_size = math.ceil(len(tasks)/NTHREADS)
            task_group = tasks[core * task_group_size : min(core * task_group_size + task_group_size,len(tasks))]
            x = threading.Thread(target=benchmark, args=(task_group,core,time,seed))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join() 

