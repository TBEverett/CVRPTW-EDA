import os
import pandas as pd
import numpy as np
import math
import csv

files = dict()
files["base"] = [i for i in os.listdir(".") if "Base" in i]
files["KNN1"] = [i for i in os.listdir(".") if "KNN1" in i]
files["KNN3"] = [i for i in os.listdir(".") if "KNN3" in i]
files["KNN5"] = [i for i in os.listdir(".") if "KNN5" in i]
files["KNN7"] = [i for i in os.listdir(".") if "KNN7" in i]
files["KNN9"] = [i for i in os.listdir(".") if "KNN9" in i]
files["KNN11"] = [i for i in os.listdir(".") if "KNN11" in i]
files["Random1"] = [i for i in os.listdir(".") if ("Random" in i and len(i) == 18)]
files["NeuralNetwork"] = [i for i in os.listdir(".") if "NeuralNetwork" in i]

sets = list(files.keys())
for s in sets:
    if files[s] == False:
        del(files[s])

instances = dict()
#Para cada parameter set, obtenemos sus instancias
for parameter_set, parameter_files in files.items():
    for file in parameter_files:
        if parameter_set not in instances:
            instances[parameter_set] = list()
        instances[parameter_set] += [l.strip() for l in open(file).readlines() if "eval" not in l]

#Obtenemos instancias ordenadas por cada set y type
parameter_evals = dict()
for parameter_set, insts in instances.items():
    solomon_evals = dict()
    homberger_evals = dict()
    for i in insts:
        if not i: #Revisamos linea vacía
            continue
        name = i.split(",")[0]
        eval = i.split(",")[1]
        types = ["RC1","RC2","R1","R2","C1","C2"]
        for t in types:
            if t in name:
                type = t
                break
        if "solomon" in name:
            if type not in solomon_evals:
                solomon_evals[type] = list()
            solomon_evals[type].append(int(eval))
        elif "Homberger" in name:
            if type not in homberger_evals:
                homberger_evals[type] = list()
            homberger_evals[type].append(int(eval))
    for t,evals in solomon_evals.items():
        solomon_evals[t] = str(int(np.mean(evals)))
    for t,evals in homberger_evals.items():
        homberger_evals[t] = str(int(np.mean(evals)))
    parameter_evals[parameter_set] = {"solomon":solomon_evals,"homberger":homberger_evals}

output = []
for parameter_set, authors in parameter_evals.items():
    for author, types in authors.items():
        output.append([parameter_set,author,"C1",types["C1"]])
        output.append([parameter_set,author,"C2",types["C2"]])
        output.append([parameter_set,author,"R1",types["R1"]])
        output.append([parameter_set,author,"R2",types["R2"]])
        output.append([parameter_set,author,"RC1",types["RC1"]])
        output.append([parameter_set,author,"RC2",types["RC2"]])

solomon_output = open("solomon_results.csv","w")
homberger_output = open("homberger_results.csv","w")
for l in output:
    if "solomon" in l:
        solomon_output.write(",".join(l) + "\n")
    else:
        homberger_output.write(",".join(l) + "\n")

solomon_output.close()
homberger_output.close()




