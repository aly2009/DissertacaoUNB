import re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', config_file_path='./', keep_para=True):
        """
        Attributes
        ----------
            config_file_path : the config file path
            path : the input path stores the input log file name
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.config_file_path = config_file_path
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.keep_para = keep_para

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def outputResult(self, df_concatenated):
        df_concatenated.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False, encoding='utf-8-sig', header=True)
        


    def extract_templates(self, log_messages, config_file_path):
        """
        Função para extrair os templates e parâmetros dos log messages.
        """
        config = TemplateMinerConfig()
        config.load(config_file_path)
        config.profiling_enabled = False

        persistence = None
        template_miner = TemplateMiner(persistence, config=config)

        rows_to_add = []
        for line in log_messages:
            #line = preprocess_log_event(line) #Incluído para testar o pré-processamento antes dos templates
            try:
                template_miner.add_log_message(line)
                cluster = template_miner.match(line)
                if cluster is not None:
                    parms = template_miner.get_parameter_list(cluster.get_template(), line)
                    insert_row = {
                        "ClusterId": cluster.cluster_id,
                        "EventTemplate": cluster.get_template(),
                        "ParameterList": parms,
                    }
                    rows_to_add.append(insert_row)
                else:
                    # Exibe a linha que causou o problema
                    raise ValueError(f"Error: No cluster found for log message:\n{line}")
            except Exception as e:
                # Exibe a linha que causou o erro
                print(f"Error processing log message:\n{line}")
                print(f"Error message: {str(e)}")

        df_templates = pd.DataFrame(rows_to_add)
        df_templates['Occurrences'] = df_templates.groupby('EventTemplate')['EventTemplate'].transform('count')

        return df_templates

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName

        self.load_data()
        
        messages = self.extract_templates(self.df_log['Content'].tolist(), self.config_file_path)
        messages.reset_index(inplace=True)
        folder_path = os.path.dirname(self.path)
        
        df_filtered = messages[['ClusterId', 'EventTemplate', 'Occurrences']].drop_duplicates().sort_values('ClusterId')
      
        df_filtered['EventId'] = range(1, len(df_filtered) + 1)
        
        self.save_filtered_csv(df_filtered)
        
        df_concatenated = pd.concat([self.df_log, messages], axis=1)
        
        df_merged = df_concatenated.merge(df_filtered[['EventTemplate', 'EventId']], on='EventTemplate', how='left')
        df_merged = df_merged.drop(['index', 'Occurrences'], axis=1)

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        
        
        self.outputResult(df_merged)

    def save_filtered_csv(self, df_filtered):
        df_filtered.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False, encoding='utf-8-sig', header=True)


     
    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe 
        """
        log_messages = []
        linecount = 0
        
        with open(log_file, 'r', encoding='utf-8-sig') as fin:
            for line in fin.readlines():
                try:
                    
                    match = regex.search(line.strip())
                    #if match is None: #DEBUG - MOSTRA LINHAS COM PROBLEMA
                    #    print('a linha é: "',line,'"') #DEBUG
                    #    print('-----------------------------') #DEBUG
                        
                    message = [match.group(header) for header in headers]
                    
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        
        logdf.insert(0, 'Id', None)
        logdf['Id'] = [i + 1 for i in range(linecount)]
        return logdf


    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header 
             
                headers.append(header)
        regex = re.compile('^' + regex + '$')
       
        return headers, regex

