import subprocess
import random
import os
import pandas as pd
import numpy as np
import sys
import threading
import math

NTHREADS = 10

solomon_instances = ["solomon/" + i for i in os.listdir("solomon") if ".txt" in i]
homberger_instances = ["Homberger/" + i for i in os.listdir("Homberger") if ".txt" in i]
instances = solomon_instances + homberger_instances

parameter_sets = dict()
#parameter_sets["KNN1"] = pd.read_csv("KNN1_predicted_parameters.csv")
#parameter_sets["KNN3"] = pd.read_csv("KNN3_predicted_parameters.csv")
#parameter_sets["KNN5"] = pd.read_csv("KNN5_predicted_parameters.csv")
#parameter_sets["NeuralNetwork"] = pd.read_csv("NeuralNetwork_predicted_parameters.csv")
parameter_sets["KNN7"] = pd.read_csv("KNN7_predicted_parameters.csv")
parameter_sets["KNN9"] = pd.read_csv("KNN9_predicted_parameters.csv")
parameter_sets["KNN11"] = pd.read_csv("KNN11_predicted_parameters.csv")

parameter_table = [[10,20,40,70,100],
                   [0.1,0.2,0.4,0.6,0.8],
                   [0.1,0.2,0.4,0.6,0.8],
                   [5,15,25,35,45],
                   [0.1,0.2,0.4,0.6,0.8]]

#parameter_sets["Random"] = list()
#for instance in parameter_sets["KNN1"]["instance"]:
#    random_vector = (np.random.rand(5) * 5).astype(int)
#    random_parameters = [parameter_table[i][random_vector[i]] for i in range(5)]
#    parameter_sets["Random"].append(random_parameters)
#parameter_sets["Random"] = pd.concat([parameter_sets["KNN1"]["instance"],pd.DataFrame(parameter_sets["Random"],columns=["gs","nc","ne","ps","xi"])],axis=1)

#del(parameter_sets["KNN1"])

#Construimos parametros base
#parameter_sets["Base"] = [[i,40,0.2,0.4,25,0.2] for i in parameter_sets["KNN1"]["instance"]]
#parameter_sets["Base"] = pd.DataFrame(parameter_sets["Base"],columns=["instance","gs","nc","ne","ps","xi"])


def benchmark(parameter_sets, instances, core):
    evals_per_set = dict()
    for name, parameter_set in parameter_sets.items():
        eval_per_instance = dict()
        for i in instances:
            params = parameter_set[parameter_set["instance"] == i.split("/")[1]].to_numpy()[0]
            command = f"python3 SolveVRP.py -i {i} -ps {int(params[4])} -gs {int(params[1])} -ne {params[3]} -nc {params[2]} -xi {params[5]} -t {time_limit} -s {seed}"
            result = subprocess.check_output(command, shell=True, stderr=subprocess.PIPE, text=True)
            if result == "inf\n":
                result = 99999
            eval_per_instance[i] = int(result)
        evals_per_set[name] = eval_per_instance

    for name,evals in evals_per_set.items():
        pairs = [(i,eval) for i,eval in evals.items()]
        evals = pd.DataFrame(pairs,columns=["instance","eval"])
        evals.to_csv("preds_final3/" + name + "_preds_" + str(core) + ".csv", index=False)


#Obtenemos instancias y obtenemos un subconjunto dependiente del core
solomon_instances = ["solomon/" + i for i in os.listdir("solomon") if ".txt" in i]
homberger_instances = ["Homberger/" + i for i in os.listdir("Homberger") if ".txt" in i]
instances = solomon_instances + homberger_instances

time_limit = 10
seed = 36

threads = list()
for core in range(NTHREADS):
    instance_group_size = math.ceil(len(instances)/NTHREADS) #Para 10 cores
    instance_group = instances[core * instance_group_size : min(core * instance_group_size + instance_group_size,len(instances))]
    x = threading.Thread(target=benchmark, args=(parameter_sets,instance_group,core))
    threads.append(x)
    x.start()

for index, thread in enumerate(threads):
    thread.join()
    print("Thread Done: ", index) 

