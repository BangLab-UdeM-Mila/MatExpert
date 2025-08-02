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

from utils import *

# This file is to find the second most similar structure to the given structure

# Fix random seed

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", 
                        type=str, 
                        default="t5-base", 
                        help="Pretrained model to use as the base encoder.")

    parser.add_argument("--temperature", 
                        type=float, 
                        default=0.5, 
                        help="Temperature parameter for the NT-Xent loss.")

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=256, 
                        help="Batch size for training.")

    parser.add_argument("--data_dir", 
                        type=Path, 
                        default="../data/basic",
                        help="Path to the directory containing training data.")
    parser.add_argument("--name",
                        type=str,
                        default="adamw_bs64_ag8",
                        help="Name of the model to save")
    parser.add_argument('--saved_dir',
                        type=str,
                        default="/data/rech/dingqian/intel/crystal-llm-retrieval",
                        help="Path to save the model")

    parser.add_argument('--set_name', 
                        type=str, 
                        default="train", 
                        help="Name of the dataset to use for training.")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    # create a dataframe with columns "material_id", "description", "query", "desc_embedding", "query_embedding", "mapped_material_id"
    df = pd.DataFrame(columns=["material_id", "description", "query", "desc_embedding", "query_embedding", "second_most_similar_material_id", "most_similar_material_id"])

    # Load the dataset
    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    data_path = str(args.data_dir / f"{args.set_name}_desc.csv")
    dataset = MyDataset(data_path, tokenizer)

    # Load the SimCLR model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.saved_dir, f"{args.name}_best.pth")
    simclr_model = SimCLR(tokenizer=dataset.tokenizer).to(device)
    simclr_model = nn.DataParallel(simclr_model)
    simclr_model.load_state_dict(torch.load(model_path))

    # Load the dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Get the embeddings for the dataset
    for batch in tqdm(dataloader):
        # input_ids1 and input_ids2 are the input_ids for the two views of the data

        with torch.no_grad():
            desc_ids, attention_mask1, prop_ids, attention_mask2, material_id = batch

            desc_ids = desc_ids.to(device)
            prop_ids = prop_ids.to(device)
            attention_mask1 = attention_mask1.to(device)
            attention_mask2 = attention_mask2.to(device)

            z_i, z_j = simclr_model(desc_ids, prop_ids, attention_mask1, attention_mask2)
            z_i = z_i.detach().cpu().numpy()
            z_j = z_j.detach().cpu().numpy()

        # convert prop_ids to words
        prop_text = tokenizer.batch_decode(prop_ids, skip_special_tokens=True)
        desc_text = tokenizer.batch_decode(desc_ids, skip_special_tokens=True)
        # embed()

        # Add the embeddings to the dataframe
        for i in range(z_i.shape[0]):
            row = pd.Series({"material_id": material_id[i],
                             "description": desc_text[i],
                             "query": prop_text[i],
                             "desc_embedding": z_i[i],
                             "query_embedding": z_j[i]})
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Find the second most similar structure using cosine smilarity in torch
    desc_embeddings = torch.tensor(np.stack(df["desc_embedding"].values))
    query_embeddings = torch.tensor(np.stack(df["query_embedding"].values))

    # Normalize the embeddings
    desc_embeddings = F.normalize(desc_embeddings, p=2, dim=1)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    # Calculate the cosine similarity
    similarity_matrix = torch.matmul(query_embeddings, desc_embeddings.T)
    second_most_similar = torch.argsort(similarity_matrix, dim=1, descending=True)[:, 1]
    most_similar = torch.argsort(similarity_matrix, dim=1, descending=True)[:, 0]

    # Add the mapped_material_id to the dataframe
    df["second_most_similar_material_id"] = [dataset.inputs[i]["material_id"] for i in second_most_similar]
    df["most_similar_material_id"] = [dataset.inputs[i]["material_id"] for i in most_similar]

    # Save the dataframe
    df.to_csv(os.path.join(args.saved_dir, f"embeddings_{args.set_name}.csv"), index=False)


    