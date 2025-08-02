import os
import pandas as pd
import numpy as np
import argparse
import glob
from pathlib import Path
import ast



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", 
                        type=Path, 
                        default="../data/basic",
                        help="Path to the directory containing training data.")

    args = parser.parse_args()
    return args

prompt_lookup = {
                "formation_energy_per_atom": "The formation energy per atom is",
                "band_gap": "The band gap is",
                "pretty_formula": "The chemical formula is",
                "e_above_hull": "The energy above the convex hull is",
                "elements": "The elements are",
                "spacegroup.number": "The spacegroup number is",
            }

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


args = parse_arguments()

data_path = str(args.data_dir / "train_desc.csv")
df = pd.concat([pd.read_csv(fn) for fn in glob.glob(data_path)])

for i in range(10):
    desc_prompt_A, property_prompt_A = get_template(df.iloc[i])
    desc_prompt_B, property_prompt_B = get_template(df.iloc[i+1])

    formula_A = df.iloc[i]["pretty_formula"]
    formula_B = df.iloc[i+1]["pretty_formula"]

    prompt = f"I have two materials {formula_A} and {formula_B}. Based on the descriptions and properties of the two materials below, can you summarize what are the main reasons for the differences in properties when transitioning from material {formula_A} to material {formula_B}?"
    prompt += '\n\n'
    prompt += f"The description of material {formula_A}: " + desc_prompt_A + '\n\n'
    prompt += f"The property of material {formula_A}: " + property_prompt_A + '\n\n'
    prompt += f"The description of material {formula_B}: " + desc_prompt_B + '\n\n'
    prompt += f"The property of material {formula_B}: " + property_prompt_B + '\n\n'

    print(prompt)
    # save to file 
    with open(f"template_{i}.txt", "w") as f:
        f.write(prompt)
    break





