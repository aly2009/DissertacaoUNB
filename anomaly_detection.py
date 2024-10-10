import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from evaluation_metrics import *

def detect_lof_anomalies(train_data, test_data, column_name='SemanticVecSeq', metric_params=None, label=False, plot_tsne=False, plot_metrics=True, gridSearch=False):
    train_data_array, test_data_array = np.vstack(train_data[column_name]), np.vstack(test_data[column_name])
    try:
        if gridSearch:
            param_grid = {
                'n_neighbors': [5, 10, 20],
                'leaf_size': [10, 20, 40],
                'metric': ['minkowski', 'euclidean'],
                #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algoritmo LOF
                'p': [1, 2],  # Parâmetro p para métrica de distância (1 para distância de Manhattan, 2 para distância Euclidiana)
                'contamination': [0.05, 0.1, 0.2],  # Proporção esperada de anomalias
            }
            model = LocalOutlierFactor(novelty=True, n_jobs=-1)
            grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='f1')
            start_time = time.time()
            grid_search.fit(train_data_array)
            training_time = round(time.time() - start_time,3)
            
            best_model = grid_search.best_estimator_
            print("Melhores parâmetros encontrados:")
            print(grid_search.best_params_)

            start_time = time.time()
            anomaly_prediction = best_model.predict(test_data_array)
            anomaly_scores = best_model.decision_function(test_data_array)
            
        else:
            model = LocalOutlierFactor(**metric_params)
            
            start_time = time.time()
            model.fit(train_data_array)
            training_time = round(time.time() - start_time,3)

            start_time = time.time()
            anomaly_prediction = model.predict(test_data_array)

            anomaly_scores = model.decision_function(test_data_array)
       
        anomalies = np.where(anomaly_prediction == -1)[0]
        prediction_time = round(time.time() - start_time,3)
        total_time = (training_time, prediction_time)
       
        # Cria um dataframe ordenado com os resultados
        data = {'Anomaly_Prediction': anomaly_prediction, 'Anomaly_Score': anomaly_scores, 'Anomaly_Sequence_Messages': test_data['sequenceMessages'].values}
        anomaly_data_frame = pd.DataFrame(data)
        anomaly_data_frame.index = test_data.index # Mantém o índice original
        anomaly_data_frame_sorted = anomaly_data_frame.sort_values(by='Anomaly_Score', ascending=True)

        if label:
            #print('Total anomaly sequences: ',(test_data['Label'] == 1).sum())
            labels = test_data['Label'].replace({0: 1, 1: -1}) #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
            #calculate_false_positives_and_negatives(labels, anomaly_prediction)
     
            precision, recall, f1 = evaluate_model(labels.values, anomaly_prediction, plot=plot_metrics) 
            fpr, tpr, roc_auc = plot_roc_curve(labels.values, anomaly_scores, plot=plot_metrics)
            metricas = (precision, recall, f1, fpr, tpr, roc_auc)
        else:
            metricas = None
   
        if plot_tsne:
            plot_anomalies_tsne(test_data_array, anomalies)
        
        return anomalies, anomaly_data_frame_sorted, anomaly_prediction, metricas, total_time

    except Exception as e:
        print("An error occurred during anomaly detection:", str(e))
		
		
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from keras.regularizers import l1, l2
from keras.layers import Dropout  
from keras.optimizers import Adam


'''
# Arquitetura 1 do Autoencoder
def create_autoencoder(input_dim):
    encoding_dim = 32
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="elu", activity_regularizer=l1(10e-5))(input_layer)    
    encoder = Dense(int(encoding_dim / 2), activation="elu")(encoder)
    encoder = Dense(int(encoding_dim / 4), activation="elu")(encoder)
   
    decoder = Dense(int(encoding_dim / 2), activation='elu')(encoder)
    decoder = Dense(int(encoding_dim / 4), activation='elu')(encoder)
    
    decoder = Dense(input_dim, activation='elu')(decoder)  
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
'''
'''
# Arquitetura 2 do Autoencoder - melhor old
def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    hidden1 = Dense(32, activation="elu", activity_regularizer=l1(10e-5))(input_layer)
    hidden1 = Dropout(0.2)(hidden1)
    
    hidden2 = Dense(16, activation="elu", activity_regularizer=l1(10e-5))(hidden1)
    hidden2 = Dropout(0.2)(hidden2)
    
    hidden3 = Dense(8, activation="elu", activity_regularizer=l1(10e-5))(hidden2)
    
    output_layer = Dense(input_dim, activation="elu", activity_regularizer=l1(10e-5))(hidden3)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    
    optimizer = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    
    return autoencoder
'''

# Arquitetura 3 do Autoencoder - melhor
def create_autoencoder(input_dim):
    # Encode
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='elu', activity_regularizer=l1(10e-5))(input_layer)
    encoder2 = Dense(32, activation='elu', activity_regularizer=l1(10e-5))(encoder)
    encoder2 = Dropout(0.2)(encoder2)
    encoder3 = Dense(16, activation='elu', activity_regularizer=l1(10e-5))(encoder2)

    # Decode
    decoder = Dense(16, activation='elu', activity_regularizer=l1(10e-5))(encoder3)
    decoder2 = Dense(32, activation='elu', activity_regularizer=l1(10e-5))(decoder)
    decoder2 = Dropout(0.2)(decoder2)
    decoder3 = Dense(64, activation='elu', activity_regularizer=l1(10e-5))(decoder2)
    decoder4 = Dense(input_dim, activation='elu', activity_regularizer=l1(10e-5))(decoder3)

    autoencoder = Model(inputs=input_layer, outputs=decoder4)
    
    optimizer = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder


# Train the autoencoder
def train_autoencoder(train_data, epochs=10, batch_size=64):
    input_dim = train_data.shape[1]
    autoencoder = create_autoencoder(input_dim)
    
    # Exiba uma barra de progresso
    for _ in tqdm(range(epochs), desc="Treinando o Autoencoder", unit="epoch"):
        np.random.shuffle(train_data)  # Embaralhe os dados de treinamento
        autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
        
    return autoencoder


def detect_autoencoder_anomalies(train_data, test_data, column_name='SemanticVecSeq', metric_params=None, label=False, plot_tsne=False, plot_metrics=True, plot_histogram=False):
    train_data_array, test_data_array = np.vstack(train_data[column_name]), np.vstack(test_data[column_name])
    try:
        # Treinamento com train_data
        start_time = time.time()
        model = train_autoencoder(train_data_array)
        training_time = round(time.time() - start_time,3)
 
        # Reconstrução com test_data
        start_time = time.time()
        reconstructed_data = model.predict(test_data_array)

        # Cálculo do erro de reconstrução(MSE)
        mse = np.mean(np.square(test_data_array - reconstructed_data), axis=1)
        #mse = apply_transformation(np.mean(np.square(test_data_array - reconstructed_data), axis=1), transformation_type='log') # Cálculo do MSE entre saídas reconstruídas e entradas originais
       
        # Identificação das anomalias com base no threshold 
        #threshold = np.percentile(mse, 95) 
        threshold = np.percentile(mse, 100 - (metric_params['contamination'] * 100)) # Atribua o threshold 
        anomaly_indices = np.where(mse > threshold)[0]
        
        # Chamada da função para plotar o histograma com o threshold
        if plot_histogram:
            plot_histogram_with_threshold(mse, threshold)
        
        anomaly_prediction = np.ones(len(mse))

        # Atribua -1 para os índices em 'anomaly_indices'
        anomaly_prediction[anomaly_indices] = -1
        prediction_time = round(time.time() - start_time,3)
        total_time = (training_time, prediction_time)
        
        #print('Total sequences: ' ,len(test_data))
    
        # Cria um dataframe ordenado com os resultados
        data = {'Anomaly_Prediction': anomaly_prediction, 'Anomaly_Score': mse, 'Anomaly_Sequence_Messages': test_data['sequenceMessages']}
        anomaly_data_frame = pd.DataFrame(data)
  
        anomaly_data_frame_sorted = anomaly_data_frame.sort_values(by='Anomaly_Score', ascending=False)
        
        if label:
            #print('Total anomaly sequences: ',(test_data['Label'] == 1).sum())
            labels = test_data['Label'].replace({0: 1, 1: -1}) #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 

            #calculate_false_positives_and_negatives(labels, anomaly_prediction)
            precision, recall, f1 = evaluate_model(labels.values, anomaly_prediction, plot=plot_metrics) 
            fpr, tpr, roc_auc = plot_roc_curve(labels.values, -mse, plot=plot_metrics)
            metricas = (precision, recall, f1, fpr, tpr, roc_auc)
        else:
            metricas = None

        if plot_tsne:
            plot_anomalies_tsne(test_data_array, anomaly_indices)
        #print(anomaly_indices)

        return anomaly_indices, anomaly_data_frame_sorted, anomaly_prediction, metricas, total_time

    except Exception as e:
        print("An error occurred during anomaly detection:", str(e))


from minisom import MiniSom

def detect_som_anomalies(train_data, test_data, column_name='SemanticVecSeq', metric_params=None, label=False, plot_tsne=False, plot_metrics=True):
    train_data_array, test_data_array = np.vstack(train_data[column_name]), np.vstack(test_data[column_name])

    try:
        contamination, grid_size, sigma, learning_rate = metric_params['contamination'], metric_params['grid_size'], metric_params['sigma'], metric_params['learning_rate']

        # Inicializa o modelo SOM
        model = MiniSom(grid_size[0], grid_size[1], train_data_array.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=123)

        # Inicializa os pesos randomicamente e treina o SOM
        model.random_weights_init(train_data_array)
        start_time = time.time()
        model.train(train_data_array, num_iteration=len(train_data_array) * 10)
        training_time = round(time.time() - start_time,3)

        start_time = time.time()
        # Encontra os neurônios BMU (Best Matching Unit) para os dados de teste
        bmu = np.array([model.winner(x) for x in test_data_array])

        # Calcula as distâncias entre os pontos de teste e os neurônios BMU
        distances = np.linalg.norm(test_data_array - model.get_weights()[bmu[:, 0], bmu[:, 1]], axis=1)

        # Calcula um limiar com base nas distâncias para identificar anomalias
        threshold = np.percentile(distances, 100 * (1 - contamination))
        
        # Encontra os índices das instâncias que são identificadas como anomalias
        anomaly_indices = np.where(distances > threshold)[0]

        prediction_time = round(time.time() - start_time,3)
        
        total_time = (training_time, prediction_time)
        
        #print('Total sequences: ' ,len(test_data_array))
    
        # Cria um dataframe ordenado com os resultados
        data = {
            'Distance': distances,
            'Anomaly_Flag': [-1 if i in anomaly_indices else 1 for i in range(len(test_data_array))],
            'Anomaly_Sequence_Messages': test_data['sequenceMessages'].values
        }
        anomaly_data_frame = pd.DataFrame(data)
        anomaly_data_frame.index = test_data.index  # Keep the original index
        anomaly_data_frame_sorted = anomaly_data_frame.sort_values(by='Distance', ascending=False)

        if label:
            #print('Total anomaly sequences: ',(test_data['Label'] == 1).sum())
            labels = test_data['Label'].replace({0: 1, 1: -1}) #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
            #calculate_false_positives_and_negatives(labels, anomaly_data_frame['Anomaly_Flag'].values)
            precision, recall, f1 = evaluate_model(labels.values, anomaly_data_frame['Anomaly_Flag'].values, plot=plot_metrics) 
            fpr, tpr, roc_auc = plot_roc_curve(labels.values, -distances, plot=plot_metrics)
            metricas = (precision, recall, f1, fpr, tpr, roc_auc)
        else:
            metricas = None

        if plot_tsne:
            plot_anomalies_tsne(test_data_array, anomaly_indices)

        return anomaly_indices, anomaly_data_frame_sorted, anomaly_data_frame['Anomaly_Flag'].values, metricas, total_time

    except Exception as e:
        print("An error occurred during anomaly detection:", str(e))


from sklearn.ensemble import IsolationForest
def detect_isolation_forest_anomalies(train_data, test_data, column_name='SemanticVecSeq', metric_params=None, 
                                      label=False, plot_tsne=False, plot_metrics=True, gridSearch=False):
    train_data_array, test_data_array = np.vstack(train_data[column_name]), np.vstack(test_data[column_name])
    try:
        if gridSearch:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_samples': [0.5, 0.6, 0.7],
                'contamination': [0.05, 0.1, 0.2],
                'max_features': [1.0, 0.9, 0.8],
            }
            model = IsolationForest()
            grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='f1')
            start_time = time.time()
            grid_search.fit(train_data_array)
            training_time = round(time.time() - start_time,3)
            
            best_model = grid_search.best_estimator_
            print("Melhores parâmetros encontrados:")
            print(grid_search.best_params_)

            start_time = time.time()
            anomaly_prediction = best_model.predict(test_data_array)
            anomaly_scores = best_model.decision_function(test_data_array)
        else:
            model = IsolationForest(**metric_params)
            start_time = time.time()
            model.fit(train_data_array)
            training_time = round(time.time() - start_time,3)

            start_time = time.time()
            anomaly_prediction = model.predict(test_data_array)

            anomaly_scores = model.decision_function(test_data_array)
 
        anomalies = np.where(anomaly_prediction == -1)[0]
        prediction_time = round(time.time() - start_time,3)
        total_time = (training_time, prediction_time)
   
        #print('Total sequences: ' ,len(test_data_array))

        # Cria um dataframe ordenado com os resultados
        data = {'Anomaly_Prediction': anomaly_prediction, 'Anomaly_Score': anomaly_scores, 'Anomaly_Sequence_Messages': test_data['sequenceMessages'].values}
        anomaly_data_frame = pd.DataFrame(data)
        anomaly_data_frame.index = test_data.index # Mantém o índice original
        anomaly_data_frame_sorted = anomaly_data_frame.sort_values(by='Anomaly_Score', ascending=True)

        if label:
            labels = test_data['Label'].replace({0: 1, 1: -1}) #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
            #calculate_false_positives_and_negatives(labels, anomaly_prediction)
            precision, recall, f1 = evaluate_model(labels.values, anomaly_prediction, plot=plot_metrics) 
            fpr, tpr, roc_auc = plot_roc_curve(labels.values, anomaly_scores, plot=plot_metrics)
            metricas = (precision, recall, f1, fpr, tpr, roc_auc)
            
        else:
            metricas = None
            
        if plot_tsne:
            plot_anomalies_tsne(test_data_array, anomalies)
    
        return anomalies, anomaly_data_frame_sorted, anomaly_prediction, metricas, total_time

    except Exception as e:
        print("An error occurred during anomaly detection:", str(e))


from pyod.models.hbos import HBOS  
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

def detect_hbos_anomalies(train_data, test_data, column_name='SemanticVecSeq', metric_params=None, label=False, plot_tsne=False, plot_metrics=True):
    train_data_array, test_data_array = np.vstack(train_data[column_name]), np.vstack(test_data[column_name])
    try:
        model = HBOS(**metric_params)  

        start_time = time.time()
        model.fit(train_data_array)
        training_time = round(time.time() - start_time, 3)

        start_time = time.time()
        anomaly_scores = model.decision_function(test_data_array)  # Calcula os scores de anomalia
        anomaly_prediction = model.predict(test_data_array)  # Realiza as previsões (no pyod, classe 0: normal, classe 1: anomalia)
        
        anomaly_prediction = [1 if value == 0 else -1 for value in anomaly_prediction] #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
        anomaly_prediction = np.array(anomaly_prediction)
        
        anomalies = np.where(anomaly_prediction == -1)[0]  # Encontra as anomalias (classe -1)
        prediction_time = round(time.time() - start_time, 3)
        total_time = (training_time, prediction_time)

        #print('Total sequences: ', len(test_data_array))

        # Cria um dataframe ordenado com os resultados
        data = {'Anomaly_Prediction': anomaly_prediction, 'Anomaly_Score': anomaly_scores,
                'Anomaly_Sequence_Messages': test_data['sequenceMessages'].values}
        anomaly_data_frame = pd.DataFrame(data)
        anomaly_data_frame.index = test_data.index  # Mantém o índice original
        anomaly_data_frame_sorted = anomaly_data_frame.sort_values(by='Anomaly_Score', ascending=True)

        if label:
            labels = test_data['Label'].replace({0: 1, 1: -1})  #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
            #calculate_false_positives_and_negatives(labels, anomaly_prediction)
            precision, recall, f1 = evaluate_model(labels.values, anomaly_prediction, plot=plot_metrics)
            fpr, tpr, roc_auc = plot_roc_curve(labels.values, -anomaly_scores, plot=plot_metrics)
            metricas = (precision, recall, f1, fpr, tpr, roc_auc)
        else:
            metricas = None

        if plot_tsne:
            plot_anomalies_tsne(test_data_array, anomalies)

        return anomalies, anomaly_data_frame_sorted, anomaly_prediction, metricas, total_time

    except Exception as e:
        print("An error occurred during anomaly detection:", str(e))



from pyod.models.cblof import CBLOF  

def detect_cblof_anomalies(train_data, test_data, column_name='SemanticVecSeq', metric_params=None, label=False, plot_tsne=False, plot_metrics=True):
    train_data_array, test_data_array = np.vstack(train_data[column_name]), np.vstack(test_data[column_name])
    try:
        model = CBLOF(**metric_params)  

        start_time = time.time()
        model.fit(train_data_array)
        training_time = round(time.time() - start_time, 3)

        start_time = time.time()
        anomaly_scores = model.decision_function(test_data_array)  # Calcula os scores de anomalia
        anomaly_prediction = model.predict(test_data_array)  # Realiza as previsões (no pyod, classe 0: normal, classe 1: anomalia)

        anomaly_prediction = [1 if value == 0 else -1 for value in anomaly_prediction] #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
        anomaly_prediction = np.array(anomaly_prediction)
        
        anomalies = np.where(anomaly_prediction == -1)[0]  # Encontra as anomalias (classe -1)
        prediction_time = round(time.time() - start_time, 3)
        total_time = (training_time, prediction_time)

        #print('Total sequences: ', len(test_data_array))

        # Cria um dataframe ordenado com os resultados
        data = {'Anomaly_Prediction': anomaly_prediction, 'Anomaly_Score': anomaly_scores,
                'Anomaly_Sequence_Messages': test_data['sequenceMessages'].values}
        anomaly_data_frame = pd.DataFrame(data)
        anomaly_data_frame.index = test_data.index  # Mantém o índice original
        anomaly_data_frame_sorted = anomaly_data_frame.sort_values(by='Anomaly_Score', ascending=True)

        if label:
            labels = test_data['Label'].replace({0: 1, 1: -1})  #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
            #calculate_false_positives_and_negatives(labels, anomaly_prediction)
            precision, recall, f1 = evaluate_model(labels.values, anomaly_prediction, plot=plot_metrics)
            fpr, tpr, roc_auc = plot_roc_curve(labels.values, -anomaly_scores, plot=plot_metrics)
            metricas = (precision, recall, f1, fpr, tpr, roc_auc)
            
         
        else:
            metricas = None

        if plot_tsne:
            plot_anomalies_tsne(test_data_array, anomalies)

        return anomalies, anomaly_data_frame_sorted, anomaly_prediction, metricas, total_time

    except Exception as e:
        print("An error occurred during anomaly detection:", str(e))



import time
from pyod.models.gmm import GMM  

def detect_gmm_anomalies(train_data, test_data, column_name='SemanticVecSeq', metric_params=None, label=False, plot_tsne=False, plot_metrics=True):
    train_data_array, test_data_array = np.vstack(train_data[column_name]), np.vstack(test_data[column_name])
    try:
        model = GMM(**metric_params)  

        start_time = time.time()
        model.fit(train_data_array)
        training_time = round(time.time() - start_time, 3)

        start_time = time.time()
        anomaly_scores = model.decision_function(test_data_array)  # Calcula os scores de anomalia
        anomaly_prediction = model.predict(test_data_array)  # Realiza as previsões (no pyod, classe 0: normal, classe 1: anomalia)
        
        anomaly_prediction = [1 if value == 0 else -1 for value in anomaly_prediction] #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
        anomaly_prediction = np.array(anomaly_prediction)
        
        anomalies = np.where(anomaly_prediction == -1)[0]  # Encontra as anomalias (classe -1)
        prediction_time = round(time.time() - start_time, 3)
        total_time = (training_time, prediction_time)

        #print('Total sequences: ', len(test_data_array))

        # Cria um dataframe ordenado com os resultados
        data = {'Anomaly_Prediction': anomaly_prediction, 'Anomaly_Score': anomaly_scores,
                'Anomaly_Sequence_Messages': test_data['sequenceMessages'].values}
        anomaly_data_frame = pd.DataFrame(data)
        anomaly_data_frame.index = test_data.index  # Mantém o índice original
        anomaly_data_frame_sorted = anomaly_data_frame.sort_values(by='Anomaly_Score', ascending=True)

        if label:
            labels = test_data['Label'].replace({0: 1, 1: -1})  #Substitui a label dos dados normais de 0 para 1 e anomalias de 1 para -1 
            #calculate_false_positives_and_negatives(labels, anomaly_prediction)
            precision, recall, f1 = evaluate_model(labels.values, anomaly_prediction, plot=plot_metrics)
            fpr, tpr, roc_auc = plot_roc_curve(labels.values, -anomaly_scores, plot=plot_metrics)
            metricas = (precision, recall, f1, fpr, tpr, roc_auc)
        else:
            metricas = None

        if plot_tsne:
            plot_anomalies_tsne(test_data_array, anomalies)

        return anomalies, anomaly_data_frame_sorted, anomaly_prediction, metricas, total_time

    except Exception as e:
        print("An error occurred during anomaly detection:", str(e))


from pyod.models.mcd import MCD  

def detect_mcd_anomalies(train_data, test_data, column_name='SemanticVecSeq', metric_params=None, label=False, plot_tsne=False, plot_metrics=True):
    train_data_array, test_data_array = np.vstack(train_data[column_name]), np.vstack(test_data[column_name])
    try:
        model = MCD(**metric_params)  # Inicializa o modelo MCD

        start_time = time.time()
        model.fit(train_data_array)  # Ajusta o modelo aos dados de treinamento
        training_time = round(time.time() - start_time, 3)

        start_time = time.time()
        anomaly_scores = model.decision_function(test_data_array)  # Calcula os scores de anomalia
        anomaly_prediction = model.predict(test_data_array)  # Realiza as previsões
        
        anomaly_prediction = [1 if value == 0 else -1 for value in anomaly_prediction] 
        anomaly_prediction = np.array(anomaly_prediction)
        
        anomalies = np.where(anomaly_prediction == -1)[0]  # Encontra as anomalias
        prediction_time = round(time.time() - start_time, 3)
        total_time = (training_time, prediction_time)

        # Cria um dataframe ordenado com os resultados
        data = {'Anomaly_Prediction': anomaly_prediction, 'Anomaly_Score': anomaly_scores,
                'Anomaly_Sequence_Messages': test_data['sequenceMessages'].values}
        anomaly_data_frame = pd.DataFrame(data)
        anomaly_data_frame.index = test_data.index
        anomaly_data_frame_sorted = anomaly_data_frame.sort_values(by='Anomaly_Score', ascending=True)

        if label:
            labels = test_data['Label'].replace({0: 1, 1: -1})  
            precision, recall, f1 = evaluate_model(labels.values, anomaly_prediction, plot=plot_metrics)
            fpr, tpr, roc_auc = plot_roc_curve(labels.values, -anomaly_scores, plot=plot_metrics)
            metricas = (precision, recall, f1, fpr, tpr, roc_auc)
        else:
            metricas = None

        if plot_tsne:
            plot_anomalies_tsne(test_data_array, anomalies)

        return anomalies, anomaly_data_frame_sorted, anomaly_prediction, metricas, total_time

    except Exception as e:
        print("An error occurred during anomaly detection:", str(e))
        