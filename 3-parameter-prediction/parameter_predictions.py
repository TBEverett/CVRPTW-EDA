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

time = "60"

# Hacemos el match entre las caracteristicas de cada instancia y sus parametros obtenidos con paramILS
df_features = pd.read_csv("Homberger_features.csv")
df_params = pd.read_csv(f"best_params_T{time}.csv")

df_params["instance"] = df_params["instance"] + ".txt"
df_params = df_params.drop("eval",axis=1)

#Combinamos features y parametros en un solo df
df = pd.merge(df_features,df_params,on="instance")

#Ahora separaremos las instancias (300) en dos conjuntos, (225 y 75) solo se entrenará con el primero (usando 80/20 train validation)
#Luego predecimos para las 75 instancias hold-out, y ejecutaremos sobre ellas para ver la calidad de los resultados.

C1_instances = df.loc[df["instance"].str.contains("C1")]
C1_instances = C1_instances.loc[C1_instances["instance"].str.contains("RC1") == False] 
C2_instances = df.loc[df["instance"].str.contains("C2")]
C2_instances = C2_instances.loc[C2_instances["instance"].str.contains("RC2") == False]
R1_instances = df.loc[df["instance"].str.contains("R1")]
R2_instances = df.loc[df["instance"].str.contains("R2")]
RC1_instances = df.loc[df["instance"].str.contains("RC1")]
RC2_instances = df.loc[df["instance"].str.contains("RC2")]

C1_train, C1_test = C1_instances.iloc[10:], C1_instances.iloc[:10]
C2_train, C2_test = C2_instances.iloc[10:], C2_instances.iloc[:10]
R1_train, R1_test = R1_instances.iloc[10:], R1_instances.iloc[:10]
R2_train, R2_test = R2_instances.iloc[10:], R2_instances.iloc[:10]
RC1_train, RC1_test = RC1_instances.iloc[10:], RC1_instances.iloc[:10]
RC2_train, RC2_test = RC2_instances.iloc[10:], RC2_instances.iloc[:10]

df_train = pd.concat([C1_train,C2_train,R1_train,R2_train,RC1_train,RC2_train],axis=0).reset_index(drop=True)
df_test = pd.concat([C1_test,C2_test,R1_test,R2_test,RC1_test,RC2_test],axis=0).reset_index(drop=True)

#Almacenamos instancias usadas para train y test como .csv. Luego usaremos el test para los gráficos de evaluación
df_train.to_csv(f"T{time}_train.csv",header=False,index=False)
df_test.to_csv(f"T{time}_test.csv",header=False,index=False)

#Empezamos codigo para ML
instances = df_train["instance"]
test_instances = df_test["instance"]
X = df_train.drop(["instance"],axis=1)
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

# Define the number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=30)

# Initialize lists to store results
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
models = []

# Initialize early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

print("Training Neural Network...")
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
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(5, activation='linear'))

    optimizer = AdamW(learning_rate=0.002)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

    # Train the model with early stopping
    history = model.fit(X_train, Y_train, epochs=1000, batch_size=30, verbose=False, 
                        validation_data=(X_val, Y_val), callbacks=[early_stopping])

    # Evaluate the model on the training set
    train_loss, train_accuracy = model.evaluate(X_train, Y_train)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, Y_val)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    models.append(model)
    print("Train Loss: " + str(train_loss) + " Validation Loss: " + str(val_loss))
    i += 1

#Sacamos mejor modelo
best_model_index = np.argmin(val_losses)
best_model = models[best_model_index]


#Aqui predecimos sobre el conjunto de test y así veremos despues la comparación de parametros
#sobre instancias que los modelos nunca han visto

X_test = df_test.drop(["instance"],axis=1)
X_test = X_test[X_test.columns[:-5]]
X_test = scaler_X.fit_transform(X_test)

preds = best_model.predict(X_test)

print(preds)
predicted_params = pd.DataFrame(scaler_Y.inverse_transform(preds),columns=Y.columns)
predicted_params = pd.concat([test_instances,predicted_params],axis=1)

print(predicted_params)
predicted_params["gs"] = predicted_params["gs"].astype(int)
predicted_params["ps"] = predicted_params["ps"].astype(int)
predicted_params = predicted_params.round(2)

predicted_params.to_csv(f"predicted_params/NeuralNetwork_predicted_parameters_T{time}.csv",index=False)

print("Training KNN models...")
#Predicción usando KNN
#Aqui K es el numero de vecinos que queremos considerar
Ks = [1,3,5,7,9,11]
for K in Ks:
    #Se entrena KNN
    df_no_instances = df_train.drop("instance",axis=1)
    X = np.array(df_no_instances)
    KNN = KNeighborsRegressor(n_neighbors=K+1)
    KNN.fit(X, X)

    #Para cada instancia de test, obtenemos las instancias más cercanas que conoce KNN
    nearest_instances = dict()
    for instance in test_instances:
        instance_features = df_test[df_test["instance"] == instance].drop("instance",axis=1)
        feature_vector = np.array(instance_features).reshape(1, -1)

        nearest_neighbors = KNN.kneighbors(feature_vector, return_distance=False)
        nearest_neighbor_indexes = [nearest_neighbors[0][i] for i in range(1,K+1)]

        nearest_instances[instance] = list(df_train.loc[nearest_neighbor_indexes]["instance"])

    # Obtenemos vector de parametros predichos usando KNN para cada instancia
    # Promediamos los parametros de los vecinos
    predicted_parameters = dict()
    for instance, neighbours in nearest_instances.items():
        predicted_parameters[instance] = np.array([0,0,0,0,0],dtype=np.float64)
        for n in neighbours:
            pred = np.array(df_train[df_train["instance"] == n][["gs","nc","ne","ps","xi"]])[0]
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
    csv_file_path = f"predicted_params/KNN{K}_predicted_parameters_T{time}.csv"
    df_out.to_csv(csv_file_path, index=False)
print("KNN models DONE")
