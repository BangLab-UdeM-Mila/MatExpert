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
import ast

# This file is to find the second most similar structure to the given structure

# Fix random seed

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

chemical_element_symbols = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", 
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", 
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", 
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", 
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", 
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "Pa", "U", 
    "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", 
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", 
    "Ts", "Og"
]

def convert_string_to_list(string):
    # Convert the string to a list using ast.literal_eval
    return ast.literal_eval(string)

class MyDataset(Dataset):
    def __init__(self, csv_fn, tokenizer, max_length=512):
        super(MyDataset, self).__init__()

        if not os.path.exists(csv_fn) and not glob.glob(csv_fn):
            raise ValueError(f"CSV file {csv_fn} does not exist")

        df = pd.concat([pd.read_csv(fn) for fn in glob.glob(csv_fn)])
        # Remove rows with no description

        df = df.dropna(subset=["description"])
        self.inputs = df.to_dict(orient="records")

        # Print min and max length of the descriptions (take word numbers as length)

        desc_lengths = [len(input_dict["description"].split()) for input_dict in self.inputs]
        print(f"Min description length: {min(desc_lengths)}")
        print(f"Max description length: {max(desc_lengths)}")
        print(f"Average description length: {sum(desc_lengths) / len(desc_lengths)}")

        # How many descriptions are longer than 512 tokens?

        num_long_desc = sum([1 for l in desc_lengths if l > 512])
        print(f"Number of descriptions longer than 512 tokens: {num_long_desc}")
        print(f"Ratio of descriptions longer than 512 tokens: {num_long_desc / len(desc_lengths)}")

        tokenizer.add_tokens(chemical_element_symbols)

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.w_attributes = True
    
    def get_property(self, input_dict):

        prompt = "Below is a description of a bulk material. "
        
        all_attributes = [
            "formation_energy_per_atom",
            "band_gap",
            "e_above_hull",
            "spacegroup.number"
        ]

        # sample a random collection of attributes
        num_attributes = random.randint(1, len(all_attributes))
        # num_attributes = len(all_attributes)
        if num_attributes > 0 and self.w_attributes:
            attributes = random.sample(all_attributes, num_attributes)
            attributes = ["pretty_formula", "elements"] + attributes

            prompt_lookup = {
                "formation_energy_per_atom": "The formation energy per atom is",
                "band_gap": "The band gap is",
                "pretty_formula": "The chemical formula is",
                "e_above_hull": "The energy above the convex hull is",
                "elements": "The elements are",
                "spacegroup.number": "The spacegroup number is",
            }

            for attr in attributes:
                if attr == "elements":
                    prompt += f"{prompt_lookup[attr]} {', '.join(convert_string_to_list(input_dict[attr]))}. "
                elif attr in ["formation_energy_per_atom", "band_gap", "e_above_hull"]:
                    prompt += f"{prompt_lookup[attr]} {round(float(input_dict[attr]), 4)}. "
                else:
                    prompt += f"{prompt_lookup[attr]} {input_dict[attr]}. "

        return prompt


    def get_description(self, input_dict):
        return input_dict['description']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        
        input_dict = self.inputs[index]

        desc_text = self.get_description(input_dict)
        prop_text = self.get_property(input_dict)
        
        desc_encoding = self.tokenizer(desc_text, truncation=True, max_length=self.max_length, padding="max_length")
        prop_encoding = self.tokenizer(prop_text, truncation=True, max_length=self.max_length, padding="max_length")
        
        desc_ids = torch.tensor(desc_encoding["input_ids"])
        desc_mask = torch.tensor(desc_encoding["attention_mask"])
        prop_ids = torch.tensor(prop_encoding["input_ids"])
        prop_mask = torch.tensor(prop_encoding["attention_mask"])
        
        return desc_ids, desc_mask, prop_ids, prop_mask, input_dict["material_id"]


class T5Encoder(nn.Module):
    def __init__(self, tokenizer):
        super(T5Encoder, self).__init__()
        self.t5 = T5EncoderModel.from_pretrained('t5-base').encoder
        self.t5.resize_token_embeddings(len(tokenizer))
        # embed()

    def forward(self, input_ids, attention_mask):
        return self.t5(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

class SimCLR(nn.Module):
    def __init__(self, tokenizer, projection_dim=256):
        super(SimCLR, self).__init__()
        self.encoder1 = T5Encoder(tokenizer)
        self.encoder2 = T5Encoder(tokenizer)

        self.projector1 = nn.Sequential(
            nn.Linear(self.encoder1.t5.config.d_model, self.encoder1.t5.config.d_model, bias=False),
            nn.ReLU(),
            nn.Linear(self.encoder1.t5.config.d_model, projection_dim, bias=False),
        )
        self.projector2 = nn.Sequential(
            nn.Linear(self.encoder2.t5.config.d_model, self.encoder2.t5.config.d_model, bias=False),
            nn.ReLU(),
            nn.Linear(self.encoder2.t5.config.d_model, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j, mask_i, mask_j):
        h_i = self.encoder1(x_i, mask_i)
        h_j = self.encoder2(x_j, mask_j)
        z_i = self.projector1(h_i)
        z_j = self.projector2(h_j)
        # print(z_i.shape, z_j.shape)
        # embed()
        z_i = (z_i * mask_i.unsqueeze(-1)).sum(dim=-2) / mask_i.sum(dim=-1, keepdim=True)
        z_j = (z_j * mask_j.unsqueeze(-1)).sum(dim=-2) / mask_j.sum(dim=-1, keepdim=True)

        # z_i = torch.mean(z_i, dim=1)
        # z_j = torch.mean(z_j, dim=1)
        return z_i, z_j

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        # self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.register_buffer("temperature", torch.tensor(temperature).cuda())
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).cuda().float())

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    

def add_description(csv_fn):
    k = 'cif' if 'cif' in pd.read_csv(csv_fn).columns else 'cif_str'

    df = pd.read_csv(csv_fn)
    df["description"] = df[k].apply(get_desc_string)

    # rename the save file. Add suffix "_desc" to the original file name
    save_fn = str(csv_fn).split(".csv")[0] + "_desc.csv"
    df.to_csv(save_fn, index=False)

def get_desc_string_from_id(material_id):
    # desc_dir = "/data/qgding/intel/data/mp_desc" For wuhu_tailscale
    # desc_dir = "/data/rech/dingqian/intel/material_project/data" # For octal30
    desc_dir = "/Tmp/dingqian/intel/data/mp_desc" # For octal30
    desc_fn = f"{desc_dir}/{material_id}.txt"

    if os.path.exists(desc_fn):
        with open(desc_fn, "r") as f:
            desc = f.read()
        return desc
    else:
        return None
    
def add_description_from_id(csv_fn):
    df = pd.read_csv(csv_fn)
    df["description"] = df["material_id"].apply(get_desc_string_from_id)

    # rename the save file. Add suffix "_desc" to the original file name
    save_fn = str(csv_fn).split(".csv")[0] + "_desc.csv"
    df.to_csv(save_fn, index=False)

def get_desc_string(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")

    # Randomly translate within the unit cell
    structure.translate_sites(
        indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
    )

    condenser = StructureCondenser()
    describer = StructureDescriber()

    condensed_structure = condenser.condense_structure(structure)
    description = describer.describe(condensed_structure)

    return description
