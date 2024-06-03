import os
import pandas as pd

#Conjunto de resultados con el que trabajaremos
time="10"
os.chdir("/home/elizabeth/Tomas/CVRPTW-EDA/1-parameter_tuning")

#Creamos un diccionario con todos los resultados de cada instancia y para cada seed.
all_dirs = os.listdir("execution")
all_dirs = [d for d in all_dirs if (d[0] == "_" and "time:"+time in d)]
parameters = dict()
seeds = list()
for seed_dir in all_dirs:
    seed = seed_dir.strip().split(":")[-1]
    seeds.append(seed)
    
    instance_dirs = os.listdir(f'execution/{seed_dir}/')
    seed_parameters = {}
    for dir in instance_dirs:
        results_file = open(f'execution/{seed_dir}/{dir}/ParamILS_ASolveVRP_FSolveVRP_S{seed}.out',"r")
        results_lines = list()
        for line in results_file:
            if "Final best" in line:
                result_line = line
            if "Training quality of this final best" in line:
                eval_line = line

        l = result_line.strip().split(" ")
        seed_parameters[dir] = [dir.replace("baseTuning","").strip("_"), #nombre instancia
                                round(float(eval_line.split(" ")[9].strip(",")),2), #evaluacion
                                l[4].strip("gs=,"), #parametros
                                l[5].strip("nc=,"),
                                l[6].strip("ne=,"),
                                l[7].strip("ps=,"),
                                l[8].strip("xi=,")]
        results_file.close()
    parameters[seed] = seed_parameters

#Nos quedamos solo con la mejor config de parametros por instancia entre seeds de paramILS
min_eval = {instance:99999999999 for instance in instance_dirs}
best_parameters = {instance:[] for instance in instance_dirs}
for instance in instance_dirs:
    for seed in seeds:
        params = parameters[seed][instance]
        if params[1] < min_eval[instance]:
            min_eval[instance] = params[1]
            best_parameters[instance] = params

#Almacenamos los mejores parametros en un .csv
seed_params_df = pd.DataFrame.from_dict(best_parameters,columns=["instance","eval","gs","nc","ne","ps","xi"],orient="index")
seed_params_df.to_csv(f"best_params_T{time}.csv",index=False)
