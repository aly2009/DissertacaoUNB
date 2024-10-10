import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess_log_event(log_event, stemming=False, lemmatization=False):
    log_event = re.sub(r'[^a-zA-Z]+', ' ', log_event)
    stop_words = set(stopwords.words('english'))
    tokens = log_event.lower().split()
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 3]
    log_event = ' '.join(filtered_tokens)
    
    log_event = re.sub('([a-z])([A-Z])', r'\1 \2', log_event)
    
    if stemming:
        stemmer = PorterStemmer()
        tokens = log_event.split()
        normalized_tokens = [stemmer.stem(token) for token in tokens]
        log_event = ' '.join(normalized_tokens)
        
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = log_event.split()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        log_event = ' '.join(lemmatized_tokens)
    
    return log_event if log_event != '' else 'oov'

def sliding_window_indices(time_data, window_size, step_size):
    start_time, start_index, end_index = time_data[0], 0, 0
    start_end_index_list = []
    
    for cur_time in time_data:
        if cur_time < start_time + window_size * 3600:
            end_index += 1
            end_time = cur_time
        else:
            start_end_pair = (start_index, end_index)
            start_end_index_list.append(start_end_pair)
            break

    while end_index < len(time_data):
        start_time += step_size * 3600
        end_time += step_size * 3600
        i, j = start_index, end_index

        while i < j and time_data[i] < start_time:
            i += 1
        while j < len(time_data) and time_data[j] < end_time:
            j += 1

        start_index, end_index = i, j
        start_end_pair = (start_index, end_index)
        start_end_index_list.append(start_end_pair)

    return start_end_index_list

def sample_data(bgl_structured, window_size, step_size, output_path, Label=None, window_type = 'time'):

    if window_type == 'time':
        time_data, event_mapping_data, content_data = bgl_structured['seconds_since'].values, bgl_structured['EventId'].values, bgl_structured['Content'].values
        start_end_index_list = sliding_window_indices(time_data, window_size, step_size)
        inst_number = len(start_end_index_list)
        
        print(f'Total de log sequences: {inst_number}')

        expanded_indexes_list = [[] for _ in range(inst_number)]
        expanded_event_list = [[] for _ in range(inst_number)]
        expanded_messages_list = [[] for _ in range(inst_number)]

        for i in range(inst_number):
            start_index, end_index = start_end_index_list[i]
            expanded_indexes_list[i] = list(range(start_index, end_index))
            expanded_event_list[i] = list(event_mapping_data[start_index:end_index])
            expanded_messages_list[i] = list(content_data[start_index:end_index])

        if Label:
            label_data = bgl_structured['Label'].values
            labels = [1 if any(label_data[k] for k in indexes) else 0 for indexes in expanded_indexes_list]
        else:
            labels = None

        print(f"Total de log sequences anômalas: {sum(labels)}" if labels else "A flag 'Label' está desativada, não há rótulos a serem gerados.")

        if labels:
            BGL_sequence = pd.DataFrame({'sequence': expanded_event_list, 'Label': labels, 'sequenceMessages': expanded_messages_list})
        else:
            BGL_sequence = pd.DataFrame({'sequence': expanded_event_list, 'sequenceMessages': expanded_messages_list})

        BGL_sequence.to_csv(output_path, index=None)
        return BGL_sequence
        
    elif window_type == 'sequential':
        message_sequences = []
        id_sequences = []
        line_id_sequences = []
        label_sequences = [] if Label else None  # Crie a lista de labels apenas se Label for True
        result = []

        for _, row in bgl_structured.iterrows():
            message = row['Content']
            message_sequences.append(message)
            
            ids = row['LineId']
            id_sequences.append(ids)

            line_id = row['EventId']
            line_id_sequences.append(line_id)

            if Label:
                label = row['Label']
                label_sequences.append(label)

            if len(message_sequences) == window_size:
                sequence_label = 1 if Label and any(label_sequences) else 0

                if Label:
                    result.append((message_sequences[:], id_sequences[:], line_id_sequences[:], sequence_label))
                else:
                    result.append((message_sequences[:], id_sequences[:], line_id_sequences[:], None))  # Adiciona uma coluna vazia

                message_sequences = message_sequences[step_size:]
                id_sequences = id_sequences[step_size:]
                line_id_sequences = line_id_sequences[step_size:]
                if Label:
                    label_sequences = label_sequences[step_size:]

        if Label:
            df_result = pd.DataFrame(result, columns=['sequenceMessages', 'sequenceIds', 'sequenceEventId', 'sequenceLabel'])
            df_result = df_result.rename(columns={'sequenceMessages': 'sequenceMessages', 'sequenceEventId': 'sequence', 'sequenceLabel': 'Label'}) # Renomeie as colunas
            df_result = df_result[['sequence', 'Label', 'sequenceMessages', 'sequenceIds']] # Reorganize as colunas
            df_result.drop(columns=['sequenceIds'], inplace=True)
        
        else:
            df_result = pd.DataFrame(result, columns=['sequenceMessages', 'sequenceIds', 'sequenceEventId', 'Label'])
            df_result = df_result.rename(columns={'sequenceMessages': 'sequenceMessages', 'sequenceEventId': 'sequence'}) # Renomeie as colunas
            df_result = df_result[['sequence', 'Label', 'sequenceMessages', 'sequenceIds']] # Reorganize as colunas
            df_result.drop(columns=['sequenceIds', 'Label'], inplace=True)
        
        print(f'Total de log sequences: {df_result.shape[0]}')
        print(f"Total de log sequences anômalas: {sum(df_result['Label'] == 1)} Total de log sequences anômalas: {sum(df_result['Label'] == 1)}" if Label else "A flag 'Label' está desativada, não há rótulos a serem gerados.")
        df_result.to_csv(output_path, index=None)
        
        return df_result  