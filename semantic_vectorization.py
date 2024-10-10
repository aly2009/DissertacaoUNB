########## WORD EMBEDDING - VETOR SEMÂNTICO ########## 
from gensim.models import KeyedVectors, fasttext
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def generate_word_vectors_dict(data, model_type='word2vec', window=5, min_count=1, sg=0):

    # Função para obter o embedding de uma palavra do GloVe
    def get_word_embedding(word, dimension):
        try:
            return model[word]
        except KeyError:
            # Se a palavra não estiver no modelo, retorne um vetor de zeros
            return np.zeros(dimension)  # n dimensões do vetor no modelo
        
        
    # Aplicar a função get_word_embedding para cada palavra em cada sequência em 'ReplacedSequence'
    def add_word_to_dict(word, word_vector_dict, dim):
        word = word.strip(',')
        if word not in word_vector_dict:
            word_vector_dict[word] = get_word_embedding(word, dim)
            
    word_vector_dict = {} # dicionário que mapeia palavras para vetores
    model = None
            
    if model_type == 'word2vec':
        
        vector_size = 50
        
        # Combine todas as listas de strings em um único texto
        concatenatedSequencesArray = []
        for logSequence in data['ReplacedSequence']:
            sentenca_concatenada = ' '.join(logSequence)
            concatenatedSequencesArray.append(sentenca_concatenada)
        
        # Tokenização
        Tokenized = [sentence.split() for sentence in concatenatedSequencesArray]
    
        # Configurar e treinar o modelo Word2Vec
        model_word2vec = Word2Vec(Tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    
        # Itere sobre o vocabulário do modelo
        for word in model_word2vec.wv.index_to_key:
            vector = model_word2vec.wv[word] # Obtenha o vetor correspondente à palavra

            word_vector_dict[word] = vector # Armazene a palavra e o vetor no dicionário

    elif model_type == 'glove50d':
        model_path = './glove.6B.50d.txt'  # Caminho para o modelo GloVe
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, no_header=True) # Carregar o modelo GloVe pré-treinado

        vector_size = model.vector_size # Tamanho do vetor
        
        for sequence_list in data['ReplacedSequence']:
            [add_word_to_dict(word, word_vector_dict, vector_size) for sequence in sequence_list for word in sequence.split()]
            
    elif model_type == 'glove100d':
        model_path = './glove.6B.100d.txt'  # Caminho para o modelo GloVe
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, no_header=True) # Carregar o modelo GloVe pré-treinado

        vector_size = model.vector_size # Tamanho do vetor
        
        for sequence_list in data['ReplacedSequence']:
            [add_word_to_dict(word, word_vector_dict, vector_size) for sequence in sequence_list for word in sequence.split()]
            
    elif model_type == 'glove200d':
        model_path = './glove.6B.200d.txt'  # Caminho para o modelo GloVe
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, no_header=True) # Carregar o modelo GloVe pré-treinado

        vector_size = model.vector_size # Tamanho do vetor
        
        for sequence_list in data['ReplacedSequence']:
            [add_word_to_dict(word, word_vector_dict, vector_size) for sequence in sequence_list for word in sequence.split()]
            
    elif model_type == 'glove300d':
        model_path = './glove.6B.300d.txt'  # Caminho para o modelo GloVe
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, no_header=True) # Carregar o modelo GloVe pré-treinado

        vector_size = model.vector_size # Tamanho do vetor
        
        for sequence_list in data['ReplacedSequence']:
            [add_word_to_dict(word, word_vector_dict, vector_size) for sequence in sequence_list for word in sequence.split()]

    elif model_type == 'fasttext':
        #model_path = './wiki.simple.bin' #fasttext com skipgram
        #model = fasttext.load_facebook_vectors(model_path)
        model = KeyedVectors.load_word2vec_format('cc.en.300.vec') #Segundo modelo do fasttext

        vector_size = model.vector_size # Tamanho do vetor
        
        for sequence_list in data['ReplacedSequence']:
            [add_word_to_dict(word, word_vector_dict, vector_size) for sequence in sequence_list for word in sequence.split()]

    else:
        raise ValueError("Tipo de modelo não suportado. Use 'word2vec', glove' ou 'fasttext'.")
        
    # Limpar a variável model para liberar memória
    del model

    return word_vector_dict, vector_size

def process_line(line):
    joined_line = " ".join(line)
    minha_lista = joined_line.split(', ')
    return minha_lista

def map_log_sequence_to_word_vec(log_sequence, word_vector_dict, dimWordVecSeq):
    mapeamento = []
    if log_sequence == ['']:
        return None
    else:
        for logEvent in log_sequence:
            splittedLogEvent = logEvent.split() # Divide o evento de log em palavras

            # Mapeia as palavras para seus valores no dicionário ou para um array de zeros
            mapeamento_aux = [word_vector_dict.get(word, np.zeros(dimWordVecSeq)) for word in splittedLogEvent]
            mapeamento.append(mapeamento_aux)
    return mapeamento

def create_tfidf_matrix_log_sequence(log_sequence):
    if log_sequence == ['']:
        return None
    else:
        vectorizer = TfidfVectorizer(smooth_idf=True, sublinear_tf=False, use_idf=True, min_df=1) # Instancia o vetorizador TFIDF
        X = vectorizer.fit_transform(log_sequence) # Crie a matriz TF-IDF
        tfidf_tokens = vectorizer.get_feature_names_out() # Obtenha os nomes das features (tokens)

        # Crie uma lista de índices com base no número de documentos
        indices = ["LogEvent" + str(i + 1) for i in range(len(log_sequence))]

        # Crie o DataFrame com os dados
        result = pd.DataFrame(data=X.toarray(), index=indices,  columns=tfidf_tokens)

    return result

def create_weight_log_sequence(tfidf_matrix_log_sequence, logSequence):

    WeightlogSequence = [] # Cria a matriz de pesos para logSequence

    if tfidf_matrix_log_sequence is None:
        return np.array([[0.0]])
        
    else:
        # Vamos iterar diretamente pelos índices em logSequence, economizando uma variável de contador (i)
        for i, logEvent in enumerate(logSequence):
            splittedlogEvent = logEvent.split()
            WeightLogEvent = []

            for word in splittedlogEvent:
                # Verifique se a palavra está presente em tfidf_matrix
                if word in tfidf_matrix_log_sequence:
                    WeightLogEvent.append(tfidf_matrix_log_sequence[word].iloc[i])
                else:
                    # Se a palavra não estiver presente, você pode decidir como lidar com isso, por exemplo, atribuir 0
                    WeightLogEvent.append(0)

            WeightlogSequence.append(WeightLogEvent)
    #print('WeightlogSequence: ', WeightlogSequence)
    return WeightlogSequence

def create_semantic_log_seq(wordVectorSequence, dimension, weightsVectorSequence):
  
    if wordVectorSequence is None:
        semantic_vector_matrix_log_seq = np.zeros((1, dimension), dtype=np.float32)
    
    else:
        a, b = wordVectorSequence, weightsVectorSequence

        result = [
            [bj * ai for bj, ai in zip(row_b, row_a)]
            for row_b, row_a in zip(b, a)
        ]

        # Calculando a soma dos elementos ao longo das colunas
        semantic_vector_matrix_log_seq = [np.sum(row, axis=0) / len(row) for row in result]
        
    return semantic_vector_matrix_log_seq


########## CRIAÇÃO DAS SEQUÊNCIAS DE VETORES SEMÂNTICOS - NOVA VERSÃO ########## 

def create_semantic_vector_sequence(df, word_vector_dict, dimension):
    df_copy = df.copy()
    semanticLogSequenceVec = []
    for logSequence in df_copy['splittedReplacedSequence']:
        tfidf_matrix_log_sequence = create_tfidf_matrix_log_sequence(logSequence)
        WeightlogSequence = create_weight_log_sequence(tfidf_matrix_log_sequence, logSequence)
        word_vector_matrix_log_seq = map_log_sequence_to_word_vec(logSequence, word_vector_dict, dimension) # word_vector_dict_test ou word_vector_dict_train
        semanticLogSequence = create_semantic_log_seq(word_vector_matrix_log_seq, dimension, WeightlogSequence)
        semanticLogSequenceVec.append(semanticLogSequence)
        
    semanticVectorAggregated = []
    for semanticLocSequence in semanticLogSequenceVec:
        semanticVectorAgg = np.mean(semanticLocSequence, axis=0)
        semanticVectorAggregated.append(semanticVectorAgg)

    df_copy['SemanticVecSeq'] = semanticVectorAggregated
    
    return df_copy