import pandas as pd
import os
import sys
import json
from src import config

def generate_image_info():

    # Training data
    train_base_path = config.TRAIN_BASE_PATH
    train_files = os.listdir(train_base_path)
    train_files =  [file for file in train_files if file.split('.')[1]=='svs']
    train_names = [file.split('.')[0] for file in train_files]
    train_df = pd.DataFrame(data={'file':train_files,'name':train_names})
    train_df['type'] = 'train'

    # Testing data
    test_base_path = config.TEST_BASE_PATH
    test_files = os.listdir(test_base_path)
    test_files =  [file for file in test_files if file.split('.')[1]=='svs']
    test_names = [file.split('.')[0] for file in test_files]
    test_df = pd.DataFrame(data={'file':test_files,'name':test_names})
    test_df['type'] = 'test'

    df = pd.concat([train_df,test_df])
    df['name'] = df['file'].apply(lambda o: o.split('.')[0])
    df['class_annotation'] = False
    df['compartment_annotation'] = False
    df['class_annotation_file'] = ''
    df['compartment_annotation_file'] = ''

    df = df.set_index('name')

    # Class annotations
    base_path = config.CLASS_ANNOTATION_OUT_BASE_PATH
    files = os.listdir(base_path)
    files = [file for file in files if file.split('.')[1]=='tif']


    for file in files:
        filename,_ =  file.split('_output_image_')
        if filename in df.index:
            df.loc[filename,'class_annotation'] = True
            df.loc[filename,'class_annotation_file'] = file
            


    # Compartment annotations
    base_path = config.COMPARTMENT_ANNOTATION_OUT_BASE_PATH
    files = os.listdir(base_path)
    files = [file for file in files if file.split('.')[1]=='tif']


    for file in files:
        filename,_ =  file.split('_output_image_')
        if filename in df.index:
            df.loc[filename,'compartment_annotation'] = True
            df.loc[filename,'compartment_annotation_file'] = file

    df = df.reset_index()
    return df    

