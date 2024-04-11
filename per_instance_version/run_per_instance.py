import os
import subprocess
import threading

from dotenv import load_dotenv
load_dotenv()

time = os.getenv("SOLVEVRP_TIME_LIMIT")
max_evals = os.getenv("PARAMILS_MAXEVALS")

folder_name = "_"+time+"_"+max_evals
subprocess.run(["mkdir",folder_name])

NTHREADS = 10

instance_group = "solomon" #o Homberger

#Obtenemos lista de instancias
instances_with_sols = os.listdir(instance_group)
instances = [file.replace(".txt","") for file in instances_with_sols if ".sol" not in file]
instance_amount_per_thread = len(instances)//NTHREADS

def threaded_paramILS(offset):
    for i in range(instance_amount_per_thread):
        #Ejecutamos paramILS, almacenando todo en un directorio para esa instancia
        if i + offset >= len(instances): #Caso para instancias finales
            break
        subprocess.run(["bash","execute.sh",instances[i + offset],instance_group,folder_name])

threads = list()
for thread_id in range(NTHREADS + 2):
    offset = thread_id*instance_amount_per_thread
    x = threading.Thread(target=threaded_paramILS, args=(offset,))
    threads.append(x)
    x.start()

for index, thread in enumerate(threads):
    thread.join() 

print("Todos los paramILS finalizaron su ejecución")

#Escribimos resultados a best_parameters.csv
all_dirs = os.listdir(folder_name)
dirs = [d for d in all_dirs if d[0] == "_"]
best_parameters = list()
for dir in dirs:
    results_file = open(folder_name + "/" + dir + "/ParamILS_ASolveVRP_FSolveVRP_S0.out","r")
    results_lines = list()
    for line in results_file:
        if "Final best" in line:
            results_lines.append(line)
    result_line = results_lines[-1]
    l = result_line.strip().split(" ")
    best_parameters.append({"dir":dir,"params":[l[4].strip("gs=,"),
                                                l[5].strip("nc=,"),
                                                l[6].strip("ne=,"),
                                                l[7].strip("ps=,"),
                                                l[8].strip("xi=,")]})
    results_file.close()

output_file_name = "best_params_" + time + "_" + max_evals + ".csv"
output_file = open(output_file_name,"a")
if os.path.getsize(output_file_name) == 0: #Si archivo esta vacio agregamos enunciado
    output_file.write("time,maxevals,instance,gs,nc,ne,ps,xi\n")
for element in best_parameters:
    output_file.write(time + "," + max_evals + "," + element["dir"].split("_")[1])
    for p in element["params"]:
        output_file.write(","+p)
    output_file.write("\n")
output_file.close()