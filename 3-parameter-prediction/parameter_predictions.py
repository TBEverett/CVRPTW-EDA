import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import plot_model
from sklearn.neighbors import KNeighborsRegressor


# Hacemos el match entre las caracteristicas de cada instancia y sus parametros obtenidos con paramILS
df_features = pd.read_csv("Homberger_features.csv")
df_params = pd.read_csv("best_params_T10.csv")

df_params["instance"] = df_params["instance"] + ".txt"
df_params = df_params.drop("eval",axis=1)

#Combinamos features y parametros en un solo df
df = pd.merge(df_features,df_params,on="instance")


#Empezamos codigo para ML
instances = df["instance"]
X = df.drop(["instance"],axis=1)
Y = X[X.columns[-5:]]
X = X[X.columns[:-5]]

#Normalizamos X
scaler_X = preprocessing.MinMaxScaler(feature_range=(0,1))
X_normalized = scaler_X.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized,columns=X.columns)

#Normalizamos Y
scaler_Y = preprocessing.MinMaxScaler(feature_range=(0,1)) 
Y_normalized = scaler_Y.fit_transform(Y)
Y_normalized = pd.DataFrame(Y_normalized,columns=Y.columns)

#Building a simple neural network using k-fold cross validation

# Convert X_normalized and Y_normalized to numpy arrays if they are DataFrames
X_normalized = X_normalized.values if isinstance(X_normalized, pd.DataFrame) else X_normalized
Y_normalized = Y_normalized.values if isinstance(Y_normalized, pd.DataFrame) else Y_normalized

"""
# Define the number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Initialize early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)

# Loop over the folds
i = 0
for train_index, val_index in kf.split(X_normalized):
    print("Training model number " + str(i))
    X_train, X_val = X_normalized[train_index], X_normalized[val_index]
    Y_train, Y_val = Y_normalized[train_index], Y_normalized[val_index]

    model = Sequential()
    model.add(Dense(1000, input_shape=(22,), activation='tanh'))
    model.add(Dense(800, activation='tanh'))
    model.add(Dense(600, activation='tanh'))
    model.add(Dense(400, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(5, activation='linear'))

    optimizer = AdamW(learning_rate=0.002)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

    # Train the model with early stopping
    history = model.fit(X_train, Y_train, epochs=1000, batch_size=100, verbose=False, 
                        validation_data=(X_val, Y_val), callbacks=[early_stopping])

    # Evaluate the model on the training set
    train_loss, train_accuracy = model.evaluate(X_train, Y_train)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, Y_val)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print("Train Loss: " + str(train_loss) + " Validation Loss: " + str(val_loss))
    i += 1

# Calculate and print the average results over all folds
avg_train_loss = np.mean(train_losses)
avg_train_accuracy = np.mean(train_accuracies)
avg_val_loss = np.mean(val_losses)
avg_val_accuracy = np.mean(val_accuracies)

print(f"Avg Training Loss: {avg_train_loss}")
print(f"Avg Validation Loss: {avg_val_loss}")

print(f"Validation Losses: {train_losses}")
print(f"Avg Validation Loss: {val_losses}")

preds = model.predict(X_normalized)
predicted_params = pd.DataFrame(scaler_Y.inverse_transform(preds),columns=Y.columns)
predicted_params = pd.concat([instances,predicted_params],axis=1)
predicted_params["gs"] = predicted_params["gs"].astype(int)
predicted_params["ps"] = predicted_params["ps"].astype(int)
predicted_params = predicted_params.round(2)

predicted_params.to_csv("NeuralNetwork_predicted_parameters.csv",index=False)
"""

#Predicción usando KNN
#Aqui K es el numero de vecinos que queremos considerar
Ks = [1,3,5,7,9,11]
for K in Ks:
    df_no_instances = df.drop("instance",axis=1)
    X = np.array(df_no_instances)
    KNN = KNeighborsRegressor(n_neighbors=K+1)
    KNN.fit(X, X)

    instances = df["instance"]

    #Para cada instancia, obtenemos las instancias más cercanas
    nearest_instances = dict()
    for instance in instances:
        instance_features = df[df["instance"] == instance].drop("instance",axis=1)
        feature_vector = np.array(instance_features).reshape(1, -1)

        nearest_neighbors = KNN.kneighbors(feature_vector, return_distance=False)
        nearest_neighbor_indexes = [nearest_neighbors[0][i] for i in range(1,K+1)]

        nearest_instances[instance] = list(df.loc[nearest_neighbor_indexes]["instance"])

    # Obtenemos vector de parametros predichos usando KNN para cada instancia
    # Promediamos los parametros de los 3 vecinos
    predicted_parameters = dict()
    for instance, neighbours in nearest_instances.items():
        predicted_parameters[instance] = np.array([0,0,0,0,0],dtype=np.float64)
        for n in neighbours:
            pred = np.array(df[df["instance"] == n][["gs","nc","ne","ps","xi"]])[0]
            predicted_parameters[instance] += pred
        predicted_parameters[instance] = np.round(np.divide(predicted_parameters[instance],K),2)
        #Pasamos gs y ps a int para evitar tamaños de poblacion fraccionales
        predicted_parameters[instance][0] = int(predicted_parameters[instance][0])
        predicted_parameters[instance][3] = int(predicted_parameters[instance][3])

    # Almacenamos resultados
    list_for_csv = list()
    for inst, ns in predicted_parameters.items():
        list_for_csv.append([inst] + list(ns))
    
    df_out = pd.DataFrame(list_for_csv,columns=["instance","gs","nc","ne","ps","xi"])
    csv_file_path = "KNN" + str(K) + "_predicted_parameters.csv"
    df_out.to_csv(csv_file_path, index=False)
