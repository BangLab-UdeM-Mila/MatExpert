import pandas as pd
import numpy as np
import ast
import json
import random

from tqdm import tqdm

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

# setup random seed

np.random.seed(42)


def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    
    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords, 
        coords_are_cartesian=False,
    )
    
    return structure.to(fmt="cif")

# read data
data_path = "/root/autodl-tmp/llama-factory/mp_test_stage_2_prediction_llama2_1/generated_predictions.jsonl"

with open(data_path, "r") as f:
    data = f.readlines()

outputs = []

for line in tqdm(data):
    line = json.loads(line)
    gen_str = line["predict"]

    try:
        cif_str = parse_fn(gen_str)
        _ = Structure.from_str(cif_str, fmt="cif")

    except Exception as e:
        print(e)
        continue

    outputs.append({
        "gen_str": gen_str,
        "cif": cif_str,
        "model_name": "mp_llama2_1",
    })

df = pd.DataFrame(outputs)
df.to_csv("mp_llama2_1_samples.csv", index=False)


