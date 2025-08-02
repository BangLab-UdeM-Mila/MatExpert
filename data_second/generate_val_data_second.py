#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
from openai import OpenAI
import openai
import ast
import glob
import json

from tqdm import tqdm

from pymatgen.core.structure import Structure

# setup random seed

np.random.seed(42)


# In[3]:


def get_prompt_reverse(formula_B, formula_A, desc_prompt_B, property_prompt_B, desc_prompt_A, property_prompt_A):

    prompt = f"I have two materials {formula_A} and {formula_B}.\n\n"
    prompt += f"The description of material {formula_A}: " + desc_prompt_A + '\n\n'
    # prompt += f"The property of material {formula_A}: " + property_prompt_A + '\n\n'
    prompt += f"The description of material {formula_B}: " + desc_prompt_B + '\n\n'
    # prompt += f"The property of material {formula_B}: " + property_prompt_B + '\n\n'
    prompt += f"Based on the descriptions and properties of the two materials above, "
    prompt += f"can you summarize how can we transit material {formula_A} to material {formula_B} in one paragraph? \n\n"
    prompt += f"Please do not include any hint of formula {formula_B} in your answer as we have no prior knowledge of formula {formula_B}. "

    prompt += f"\n\nThe answer should begin with: To transit from {formula_A} to a new material,"

    return prompt


# In[4]:


prompt_lookup = {
                "formation_energy_per_atom": "The formation energy per atom is",
                "band_gap": "The band gap is",
                "pretty_formula": "The chemical formula is",
                "e_above_hull": "The energy above the convex hull is",
                "elements": "The elements are",
                "spacegroup.number": "The spacegroup number is",
            }

def get_crystal_string(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")

    # Randomly translate within the unit cell
    structure.translate_sites(
        indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
    )

    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]
    atom_ids = structure.species
    frac_coords = structure.frac_coords

    crystal_str = \
        " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
        " ".join([str(int(x)) for x in angles]) + "\n" + \
        "\n".join([
            str(t) + "\n" + " ".join([
                "{0:.2f}".format(x) for x in c
            ]) for t,c in zip(atom_ids, frac_coords)
        ])

    return crystal_str

def get_template(row):
    property_prompt = ""
    for key, value in prompt_lookup.items():
        if key == "elements":
            property_prompt += f"{value} {', '.join(ast.literal_eval(row[key]))}. "
        elif key in ["formation_energy_per_atom", "band_gap", "e_above_hull"]:
            property_prompt += f"{value} {round(float(row[key]), 4)}. "
        else:
            property_prompt += f"{value} {row[key]}. "
    
    desc_prompt = row["description"]

    return desc_prompt, property_prompt


def get_template_1(list_A, list_B):
    formula_A, property_A, description_A = list_A
    formula_B, property_B, description_B = list_B

    prompt = f"I am looking to design a new material with the following property: {property_A}The closest existing " \
            f"material I found in the database is {formula_B}, which has a similar property. Below is the structure " \
            f"description of {formula_B}.\n\n{description_B}\n\nHow should I modify {formula_B} to develop a new material " \
            f"with the desired property?"
    
    return prompt

def get_template_2(list_A, list_B):
    formula_A, property_A, description_A = list_A
    formula_B, property_B, description_B = list_B

    prompt = f"Based on the information, for the new material, could you generate a description of the lengths and angles of the lattice vectors " \
        f"and then the element type and coordinates for each atom within the lattice?"

    return prompt


# In[11]:


# Only the stage 1


for set_name in ['val', 'test']:
    if set_name == 'val':
        continue

    embeddings_path = f'/data/rech/dingqian/intel/crystal-llm-retrieval/embeddings_{set_name}.csv'
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)

    prediction_path = f'/data/rech/dingqian/intel/llama-factory/mp_1_{set_name}_prediction/generated_predictions.jsonl'
    # read lines of the prediction file
    with open(prediction_path, 'r') as f:
        lines = f.readlines()

    # Read train_desc.csv
    data_path = f"../data/basic/{set_name}_desc.csv"
    data_df = pd.read_csv(data_path, index_col=1)

    alpaca_list = []
    for itr, (index, row) in enumerate(tqdm(embeddings_df.iterrows())):
        prediction_1 = json.loads(lines[itr])['predict']
        material_id_A = row.name
        material_id_B = row["most_similar_material_id"] if row["most_similar_material_id"] != material_id_A else row["second_most_similar_material_id"]

        material_A = data_df.loc[material_id_A]
        material_B = data_df.loc[material_id_B]

        desc_prompt_A, property_prompt_A = get_template(material_A)
        desc_prompt_B, property_prompt_B = get_template(material_B)

        formula_A = material_A["pretty_formula"]
        formula_B = material_B["pretty_formula"]

        list_A = [formula_A, property_prompt_A, desc_prompt_A]
        list_B = [formula_B, property_prompt_B, desc_prompt_B]

        prompt_1 = get_template_1(list_A, list_B)
        prompt_2 = get_template_2(list_A, list_B)

        answer_1 = prediction_1
        answer_2 = ""

        alpaca_item = {
            "instruction": prompt_2,
            "output": answer_2,
            "history": [[prompt_1, answer_1]]
        }

        alpaca_list.append(alpaca_item)

    with open(f"mp_{set_name}_stage_2.json", "w") as f:
        json.dump(alpaca_list, f, indent=4)

