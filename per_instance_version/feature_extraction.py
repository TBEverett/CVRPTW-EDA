# Feature extractor for CVRPTW problem instances
# Input: A directory containing CVRPTW problem instances as .txt files.

import pyvrp as p
import numpy as np
import argparse
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS


parser = argparse.ArgumentParser(prog="VRPFeatureExtractor",
                                 description="Produces metrics for VRP instance description")
parser.add_argument("-d", "--dir", required=True)
args = parser.parse_args()

def client_dist(A,B):
    return np.sqrt((A.y - B.y)**2 + (A.x - B.x)**2)
def dist(tuple1, tuple2):
    return np.sqrt((tuple1[0] - tuple2[0])**2 + (tuple1[1] - tuple2[1])**2)

# Funcion para obtener la maxima y minima cantidad de overlaps de ventanas de tiempo
# Este metodo de time sweep es extremadamente ineficiente, sería mejor sortear primero
# los comienzos y fines de las time windows y recorrerlos en orden, sumando al pasar por
# un comienzo y restando al pasar por un fin
def get_time_window_features(clients_list):
    time_windows_early = dict()
    time_windows_late = dict()
    overlaps = 0
    time_overlaps_dict = dict()
    window_lengths = list()
    for client in clients_list:
        if client.tw_late not in time_windows_late:
            time_windows_late[client.tw_late] = 0
        if client.tw_early not in time_windows_early:
            time_windows_early[client.tw_early] = 0
        time_windows_late[client.tw_late] += 1
        time_windows_early[client.tw_early] += 1
        window_lengths.append(client.tw_late - client.tw_early)
    max_time = max(time_windows_late) 
    for t in range(max_time):
        if t in time_windows_early:
            overlaps += time_windows_early[t]
        if t in time_windows_late:
            overlaps -= time_windows_late[t]
        time_overlaps_dict[t] = overlaps
    max_overlaps = max(list(time_overlaps_dict.values()))
    avg_overlaps = np.mean(list(time_overlaps_dict.values()))
    mean_window_length = np.mean(window_lengths)
    max_window_length = max(window_lengths)
    return {"max_overlaps": np.round(max_overlaps/len(clients),2), 
            "avg_overlaps": np.round(avg_overlaps/len(clients),2),
            "avg_window_length":np.round(mean_window_length/max_window_length,2),
            "cv_window_length":np.round((np.std(window_lengths)/mean_window_length)*100/max_window_length,2)}

def clusteringQuality(clients, labels):
    #Almacenamos los clientes en listas separadas para cada cluster
    client_clusters = dict()
    for i in range(len(clients)):
        if labels[i] not in client_clusters.keys():
            client_clusters[labels[i]] = []
        client_clusters[labels[i]].append(clients[i])

    #Obtenemos centroides
    centroids = dict()
    for label, cluster in client_clusters.items():
        centroid = np.array([0,0])
        for client in cluster:
            centroid += np.array([client.x,client.y])
        centroid = np.divide(centroid,len(cluster))
        centroids[label] = centroid

    #Obtenemos average intra cluster distance
    intra_cluster_distances = dict()
    for label,cluster in client_clusters.items():
        distance_to_centroid = 0
        for client in cluster:
            distance_to_centroid += np.sqrt((client.x - centroids[label][0])**2 + (client.y - centroids[label][1])**2)
        distance_to_centroid /= len(cluster)
        
        intra_cluster_distances[label] = distance_to_centroid
    average_intra_cluster_distance = np.mean(list(intra_cluster_distances.values()))

    #Obtenemos distancia de cada cluster al cluster vecino más cercano
    centroid_locations = list(centroids.values())
    if len(centroid_locations) > 1:
        neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(centroid_locations)
        distances, indices = neighbors.kneighbors(centroid_locations)
        average_inter_cluster_distance = np.mean(distances)
    else:
        average_inter_cluster_distance = 0
    
    #Normalizamos los valores para que escala de instancia no afecte
    max_x = 0
    max_y = 0
    for client in clients:
        if (client.x > max_x):
            max_x = client.x
        if (client.y > max_y):
            max_y = client.y

    max_possible_distance = np.sqrt(max_x**2 + max_y**2)
    average_intra_cluster_distance /= max_possible_distance
    average_inter_cluster_distance /= max_possible_distance

    #Queremos minimizar la distancia intra cluster mientras maximizamos al distancia inter cluster
    #Penalizamos tener demasiados outliers
    outlier_ratio = (sum([i for i in labels if i == -1])*-1) / len(labels)
    quality = -2*average_intra_cluster_distance + 1*average_inter_cluster_distance + -1*outlier_ratio
    return quality
        

features = dict()

#Leemos instancia y generamos modelo
dir = args.dir
instances = os.listdir(dir)
instances = [i for i in instances if ".txt" in i]
first_pass = True
for instance_name in instances:
    print("Extracting features for: ", instance_name)
    instance = p.read(dir+ "/" + instance_name, round_func="round", instance_format="solomon")
    model = p.Model.from_data(instance)

    #Construimos matriz de distancias (pyvrp ya trabaja con una interna pero no es facilmente accesible)
    clients = model._clients
    distance_matrix = [[] for client in clients]
    for i,client in enumerate(clients):
        for j,_client in enumerate(clients):
            distance_matrix[i].append(client_dist(client, _client))
            
    #Obtenemos distancia diagonal de la boundbox para normalizar todas las features espaciales
    max_x_value = max([client.x for client in clients])
    max_y_value = max([client.y for client in clients])
    max_possible_distance = np.sqrt(max_x_value**2 + max_y_value**2)

    # DC1: Number of clients
    features["client_number"] = float(len(clients))

    #Depot Location (siempre hay 1 en las instancias)
    depot = (model._depots[0].x, model._depots[0].y)

    #Centroid of the nodes
    centroid = model.data().centroid()

    # DC3: Distance between centroid and depot
    features["distance_centroid_depot"] = dist(depot,centroid) / max_possible_distance

    # DC4: Average client distance to depot
    client_distances_to_depo = [dist((client.x,client.y),depot) for client in clients]
    features["average_distance_to_depot"] = np.mean(client_distances_to_depo) / max_possible_distance
    features["cv_distance_to_depot"] = (np.std(client_distances_to_depo)/np.mean(client_distances_to_depo))*100 / max_possible_distance

    # ND4: Distance to centroid (Interpretado como Average Distance to Centroid y CV of Distance to Centroid)
    # ND4A: Average Distance to Centroid
    distances_to_centroid = [dist(centroid,(c.x,c.y)) for c in clients]
    mean_distance_to_centroid = np.mean(distances_to_centroid)
    features["average_distance_to_centroid"] = mean_distance_to_centroid / max_possible_distance
    # ND4B: CV of Distance to Centroid
    features["cv_distance_to_centroid"] = (np.std(distances_to_centroid)/mean_distance_to_centroid)*100 / max_possible_distance

    # DC5: Client Demands (Aqui no está claro a que se refiere el paper, se incluyeron 2 métricas)
    # DC5B: Ratio of Mean of Client Demands to capacity
    capacity = model._vehicle_types[0].capacity
    all_demands = [client.demand for client in clients]
    mean_demand = np.mean(all_demands)
    features["ratio_mean_client_demand_capacity"] = mean_demand/capacity

    # DC5A: Ratio of CV of Client Demands to capacity
    features["ratio_cv_client_demand_capacity"] = (np.std(all_demands)/mean_demand)*100/capacity

    # DC10: Average number of clients per vehicle
    features["average_clients_per_vehicle"] = features["client_number"]/model.data().num_vehicles

    #the average of the normalized nearest neighbour distances (nNNd’s)  
    X = np.array([[client.x,client.y] for client in clients])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.array([d[1] for d in distances])
    mean_distance =  np.mean(distances)
    features["avg_NN_distances"] = mean_distance / max_possible_distance
    #the cv of the normalized nearest neighbour distances (nNNd’s)
    features["cv_NN_distances"] = (np.std(distances)/mean_distance) / max_possible_distance    

    # Extra: Features de Time Windows
    time_window_features = get_time_window_features(clients)
    features["tw_ratio_max_overlaps_to_total"] = time_window_features["max_overlaps"]
    features["tw_ratio_avg_overlaps_to_total"] = time_window_features["avg_overlaps"]
    features["tw_ratio_avg_window_length_to_longest"] = time_window_features["avg_window_length"]
    features["tw_ratio_cv_window_length_to_longest"] = time_window_features["cv_window_length"]

    #CLUSTERING FEATURES
    #Iteramos sobre valores de min_samples y usamos el clustering de mayor calidad.
    best_quality = -10000
    best_min_samples = 0
    for min_samples in range(2,50):
        clustering = OPTICS(min_samples=min_samples).fit(X)
        quality = clusteringQuality(clients,clustering.labels_)
        if (quality > best_quality):
            best_quality = quality
            best_clustering = clustering
            best_min_samples = min_samples
            
    # CLS1 - Optimal min_samples value
    features["optimal_min_samples"] = best_min_samples
    # CLS1 - the cluster ratio (the ratio of the number of clusters to the number of clients with clusters generated using the GDBSCAN algorithm [29])
    cluster_amount = len(set(best_clustering.labels_))
    features["cluster_ratio"] = cluster_amount / len(clients)
    
    # CLS2 - the outlier ratio (ratio of number of outliers to clients)
    outlier_amount = len([label for label in best_clustering.labels_ if label == -1])
    features["outlier_ratio"] = outlier_amount / len(clients)

    # CLS3 - the average of the number of clients per cluster relative to total client amount
    clients_per_cluster = dict()
    for label in best_clustering.labels_:
        if label not in clients_per_cluster.keys():
            clients_per_cluster[label] = 0
        clients_per_cluster[label] += 1
    mean_clients_per_cluster = np.mean(list(clients_per_cluster.values()))
    features["avg_clients_per_cluster"] = mean_clients_per_cluster / len(clients)

    # CLS4 - the CV to the number of clients per cluster relative to total client amount
    features["cv_clients_per_cluster"] = (np.std(list(clients_per_cluster.values()))/mean_clients_per_cluster) / len(clients)

    # CLS5 - cluster density (normalized intra cluster distance)
    # CLS6 - cluster spread (normalized distance to nearest cluster)
    # Este es un bloque enorme de codigo sacado de la funcion clusteringQuality.
    client_clusters = dict()
    for i in range(len(clients)):
        if best_clustering.labels_[i] not in client_clusters.keys():
            client_clusters[best_clustering.labels_[i]] = []
        client_clusters[best_clustering.labels_[i]].append(clients[i])

    #Obtenemos centroides
    centroids = dict()
    for label, cluster in client_clusters.items():
        centroid = np.array([0,0])
        for client in cluster:
            centroid += np.array([client.x,client.y])
        centroid = np.divide(centroid,len(cluster))
        centroids[label] = centroid

    intra_cluster_distances = dict()
    for label,cluster in client_clusters.items():
        distance_to_centroid = 0
        for client in cluster:
            distance_to_centroid += np.sqrt((client.x - centroids[label][0])**2 + (client.y - centroids[label][1])**2)
        distance_to_centroid /= len(cluster)
        
        intra_cluster_distances[label] = distance_to_centroid
    average_intra_cluster_distance = np.mean(list(intra_cluster_distances.values()))

    #Obtenemos distancia de cada cluster al cluster vecino más cercano
    centroid_locations = list(centroids.values())
    if len(centroid_locations) > 1:
        neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(centroid_locations)
        distances, indices = neighbors.kneighbors(centroid_locations)
        average_inter_cluster_distance = np.mean(distances)
    else:
        average_inter_cluster_distance = 0

    average_intra_cluster_distance /= max_possible_distance
    average_inter_cluster_distance /= max_possible_distance
    #Finalmente registramos las features CLS5 y CLS6
    features["intra_cluster_distance"] = average_intra_cluster_distance
    features["inter_cluster_distance"] = average_inter_cluster_distance

    # Escritura final de features
    f = open(dir + "_features.csv","a")
    if first_pass:
        f.write("instance")
        for feature in features.keys():
            f.write("," + feature)
        f.write("\n")
        first_pass = False
    f.write(instance_name)
    for _,value in features.items():
        rounded_value = round(value,4)
        f.write("," + str(rounded_value))
    f.write("\n") 
    f.close()






