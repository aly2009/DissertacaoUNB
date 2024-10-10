import pandas as pd
from logparser.logparser.Spell import Spell
from logparser.logparser.Drain import Drain
from logparser.logparser.Drain3 import Drain3

def process_log_files(structured_file_path, log_file_path, parser, log_format, output_path, rex=None, config_file_path=None):
    if parser == 'Spell':
        parser = Spell.LogParser(log_format=log_format, indir=log_file_path, outdir=output_path, rex=rex, tau=0.75, keep_para=True)
        parser.parse(log_file_path)
    elif parser == 'Drain':
        parser = Drain.LogParser(log_format=log_format, indir=log_file_path, outdir=output_path, rex=rex, depth=4, st=0.5, maxChild=100, keep_para=True)
        parser.parse(log_file_path)
    elif parser == 'Drain3':
        parser = Drain3.LogParser(log_format=log_format, indir=log_file_path, outdir=output_path, config_file_path=config_file_path, keep_para=True)
        parser.parse(log_file_path)
        
    structured_log_df = pd.read_csv(structured_file_path)
    return structured_log_df

def load_BGL_data(file_path, create_seconds_since=True, Label=False):
    bgl_structured = pd.read_csv(file_path)
    
    if create_seconds_since:
        bgl_structured["Timestamp"] = pd.to_datetime(bgl_structured["Timestamp"], unit='s')
        bgl_structured["seconds_since"] = (bgl_structured['Timestamp'] - bgl_structured['Timestamp'][0]).dt.total_seconds().astype(int)
    
    if Label:
        bgl_structured['Label'] = (bgl_structured['Label'] != '-').astype(int)
    
    return bgl_structured

def load_CCMEVAL_data(file_path, create_seconds_since=True, Label=False):
    ccmeval_structured = pd.read_csv(file_path)
    
    if create_seconds_since:
        ccmeval_structured['Datetime'] = pd.to_datetime(ccmeval_structured['Date'] + ' ' + ccmeval_structured['time'].str.split('+').str[0], format='%m-%d-%Y %H:%M:%S.%f')
        ccmeval_structured.sort_values(by='Datetime', inplace=True)
        ccmeval_structured['seconds_since'] = (ccmeval_structured['Datetime'] - ccmeval_structured['Datetime'].iloc[0]).dt.total_seconds().astype(int)
    
    if Label:
        ccmeval_structured['Label'] = (ccmeval_structured['Label'] != '-').astype(int)
    
    return ccmeval_structured
