import requests
import json
import pandas as pd
from time import sleep
import time

def check_nested_keys(d, keys):
    for key in keys:
        if key not in d:
            return False
        d = d[key]
    return True

url = 'http://nomad-lab.eu/prod/v1/api/v1/entries/archive/query'

excluded_elements = [
    "He", "Ne", "Ar", "Kr", "Xe", "Rn", "U", "Th", "Rn", "Tc", "Po", "Pu", "Pa",
    ]

query = {
    # "owner": "visible",
    # 'not':{
    #     'results.material.elements': {
    #         'any': excluded_elements
    #     }
    # }
    "results.method.simulation.program_name:any": [
        "VASP"
    ],
    "quantities:all": [
        "results.properties.structures",
        "results.properties.structures.structure_original",
        "results.properties.structures.structure_conventional",
        "results.properties.structures.structure_primitive"
    ]
}

required = {
    "results": {
        "material": {
            "chemical_formula_reduced": "*"
        },
        "properties": {
            "structures": "*"
        }
    
    }
}

df = pd.DataFrame(columns=['entry_id', 'chemical_formula', 'structure_original', 'structure_primitive', 'structure_conventional', 'json'])

time_start = time.time()

page_after_value = None
cnt = 0
save_cnt = 0
total = 0
while True:
    # try the post, if it fails, try again
    try:
        response = requests.post(
            url, json = dict(
                query=query,
                required=required,
                pagination=dict(page_size=100, page_after_value=page_after_value)
            )
        )
        data = response.json()
    except Exception as e:
        print(f"Error: {e}")
        sleep(30)
        continue

    if len(data['data']) == 0:
        break

    page_after_value = data['pagination']['next_page_after_value']

    for entry in data['data']:
        # check if all required keys are present
        if not check_nested_keys(entry, ['archive', 'results', 'properties', 'structures']):
            continue

        # save to dataframe
        entry_id = entry['entry_id']
        chemical_formula = entry['archive']['results']['material']['chemical_formula_reduced']
        structure_original = entry['archive']['results']['properties']['structures']['structure_original']
        structure_primitive = entry['archive']['results']['properties']['structures']['structure_primitive']
        structure_conventional = entry['archive']['results']['properties']['structures']['structure_conventional']
        entry_json = json.dumps(entry)

        # using pd.concat
        df = pd.concat([df, pd.DataFrame([[entry_id, chemical_formula, structure_original, structure_primitive, structure_conventional, entry_json]], columns=['entry_id', 'chemical_formula', 'structure_original', 'structure_primitive', 'structure_conventional', 'json'])], ignore_index=True)
        
        cnt += 1
        save_cnt += 1
    
    total += len(data['data'])
    time_cost = time.time() - time_start
    print(f"Processed {cnt}/{total} entries. Time cost: {time_cost:.2f}s.")

    # save to csv every 100000 entries
    if save_cnt > 100000:
        df.to_csv(f'/data/rech/dingqian/intel/nomad/{page_after_value}.csv', index=False)
        print("Saved to csv")
        save_cnt = 0

df.to_csv(f'/data/rech/dingqian/intel/nomad/final.csv', index=False)
