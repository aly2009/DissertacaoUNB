import ast
import pandas as pd

from data_preprocessing import preprocess_log_event

def create_train_test_dataset(templatesTrain, SequenceVectorTrain, onlyNormalDataTrain=False, shuffleTrain=True, removeEmptySeq=False, 
                              splitData=True, SequenceVectorTest=None, templatesTest=None, preProcessParam=None):
 
    def replace_numbers_with_template(sequence):
        sequence = ast.literal_eval(sequence)
        replaced_sequence = [eventid_to_template[item] for item in sequence]
        return replaced_sequence
    
    templatesTrain['PreProcessedTemplate'] = templatesTrain['EventTemplate'].apply(preprocess_log_event, stemming=preProcessParam['stemming'], 
                                                                                   lemmatization=preProcessParam['lemmatization'])
    eventid_to_template = dict(zip(templatesTrain['EventId'], templatesTrain['PreProcessedTemplate']))
    
    SequenceVectorTrain['ReplacedSequence'] = SequenceVectorTrain['sequence'].apply(replace_numbers_with_template)
    
    train_data = SequenceVectorTrain
    print('Total sequences: ', train_data.shape[0])
    
    label_flag = 'Label' in train_data  # Verifica se a coluna 'Label' está presente
    print('Label column present:', label_flag)
    
    if splitData:
        # Divisão dos dados em treinamento e teste
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(train_data, test_size=0.25, random_state=42, shuffle=True) #teste_size original era 0.3 (usado na qualificação)
        print('Total sequences after splitting: ', train_data.shape[0], test_data.shape[0])
    else:
        if SequenceVectorTest is not None:
            templatesTest['PreProcessedTemplate'] = templatesTest['EventTemplate'].apply(preprocess_log_event, stemming=preProcessParam['stemming'], 
                                                                                       lemmatization=preProcessParam['lemmatization'])
            eventid_to_template = dict(zip(templatesTest['EventId'], templatesTest['PreProcessedTemplate']))
            SequenceVectorTest['ReplacedSequence'] = SequenceVectorTest['sequence'].apply(replace_numbers_with_template)
            test_data = SequenceVectorTest
        else:
            test_data = None
        
    if onlyNormalDataTrain:
        if label_flag:
            total_registros_train = (train_data['Label'] == 1).sum()
        else:
            total_registros_train = 0
        if test_data is not None:
            if label_flag:
                total_registros_test = (test_data['Label'] == 1).sum()
            else:
                total_registros_test = 0
        else:
            total_registros_test = None
        print('Total anomaly sequences - Train:', total_registros_train, '\nTotal anomaly sequences - Test:', total_registros_test)
        if label_flag:
            train_data = train_data[train_data["Label"] == 0]  # Obtain only the normal data

    if shuffleTrain: # Embaralha os dados de treinamento (shuffle)
        train_data = train_data.sample(frac=1, random_state=42)

    if removeEmptySeq: # Remove sequências vazias de teste e treinamento
        mascara = train_data['sequence'].str.contains(r'\[\]')
        train_data = train_data[~mascara]
        if test_data is not None:
            mascara = test_data['sequence'].str.contains(r'\[\]')
            test_data = test_data[~mascara]   
        
    return train_data, test_data, label_flag
