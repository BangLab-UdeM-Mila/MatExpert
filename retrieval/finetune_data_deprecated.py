import torch
import numpy as np
import transformers
from transformers import BertTokenizer, BertModel, BertTokenizerFast, Trainer, \
    TrainingArguments, BertForSequenceClassification, BertConfig, BertForMaskedLM, \
    T5Model, T5Tokenizer, T5ForConditionalGeneration, T5Config, T5EncoderModel

import torch.nn as nn
import torch.nn.functional as F
import argparse

import os
from torch.utils.data import Dataset, DataLoader

from pymatgen.core.structure import Structure
from robocrys import StructureCondenser, StructureDescriber
import random
import pandas as pd
import glob

from tqdm import tqdm
from pathlib import Path

import warnings
from IPython import embed

# This file is to find the second most similar structure to the given structure

# Fix random seed

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", 
                        type=str, 
                        default="/data/rech/dingqian/intel/crystal-llm-retrieval", 
                        help="Directory containing the data files.")

    parser.add_argument("--output_dir", 
                        type=str, 
                        default="../data/basic", 
                        help="Directory to save the train data.")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    data_path = os.path.join(args.data_dir, "embeddings.csv")

    if not os.path.exists(data_path):
        raise ValueError(f"Data file {data_path} does not exist")
    
    df = pd.read_csv(data_path)

    train_path = os.path.join(args.output_dir, "train_desc.csv")

    train_df = pd.read_csv(train_path)

    # Add a column to train_df
    train_df['most_similar_material_id'] = None
    train_df['second_most_similar_material_id'] = None

    for idx, row in train_df.iterrows():
        material_id = row['material_id']
        
        # most_similar_id is the column most_similar_material_id where the material_id is the same in df
        embed()
        most_similar_id = df.loc[df['material_id'] == material_id, 'most_similar_material_id'].values[0]
        second_most_similar_id = df.loc[df['material_id'] == material_id, 'second_most_similar_material_id'].values[0]

        train_df.at[idx, 'most_similar_material_id'] = most_similar_id
        train_df.at[idx, 'second_most_similar_material_id'] = second_most_similar_id

    train_df.to_csv(train_path, index=False)