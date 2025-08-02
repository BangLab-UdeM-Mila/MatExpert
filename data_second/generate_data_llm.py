import pandas as pd
import numpy as np
import ast
import json
import random

from tqdm import tqdm

from pymatgen.core.structure import Structure

# setup random seed

np.random.seed(42)

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



if __name__ == '__main__':
    data_path = "../data/basic/train_desc.csv"
    train_df = pd.read_csv(data_path, index_col=1)

    embeddings_path = '/data/rech/dingqian/intel/crystal-llm-retrieval/embeddings.csv'
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)

    # read output_from_B.json
    gpt_output_path = "./output_from_B.json"
    gpt_output_dict = {}
    with open(gpt_output_path, "r") as f:
        # read the lines
        lines = f.readlines()
        for line in lines:
            # convert the json string to a dictionary
            data = json.loads(line)
            material_id = data["custom_id"]
            content = data['response']['body']['choices'][0]['message']['content']
            gpt_output_dict[material_id] = content


    alpaca_list = []
    for index, row in tqdm(embeddings_df.iterrows()):
        material_id_A = row.name
        material_id_B = row["most_similar_material_id"] if row["most_similar_material_id"] != material_id_A else row["second_most_similar_material_id"]

        material_A = train_df.loc[material_id_A]
        material_B = train_df.loc[material_id_B]

        desc_prompt_A, property_prompt_A = get_template(material_A)
        desc_prompt_B, property_prompt_B = get_template(material_B)

        formula_A = material_A["pretty_formula"]
        formula_B = material_B["pretty_formula"]

        list_A = [formula_A, property_prompt_A, desc_prompt_A]
        list_B = [formula_B, property_prompt_B, desc_prompt_B]

        prompt_1 = get_template_1(list_A, list_B)
        prompt_2 = get_template_2(list_A, list_B)

        answer_1 = gpt_output_dict[material_id_A]

        cif_A = train_df.loc[material_id_A]["cif"]
        alx_A = get_crystal_string(cif_A)
        answer_2 = alx_A

        # print("Q1: " + prompt_1)
        # print("A1: " + answer_1)
        # print("Q2: " + prompt_2)
        # print("A2: " + answer_2)

        # Convert to alpaca format

        alpaca_item = {
            "instruction": prompt_2,
            "output": answer_2,
            "history": [[prompt_1, answer_1]]
        }

        alpaca_list.append(alpaca_item)


    with open("mp.json", "w") as f:
        json.dump(alpaca_list, f, indent=4)