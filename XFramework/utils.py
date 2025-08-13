import pickle
import json

RANDOM_STATE=2025
def write_pickle(item, file_path): 
    with open(file_path, "wb") as f: 
        pickle.dump(item, f)

def read_pickle(file_path): 
    with open(file_path, "rb") as f: 
        return pickle.load(f)
    
def load_json(json_path): 
    with open(json_path) as f:
        json_data = json.loads(f.read())
    return json_data
    
backbone_models = ['roberta-base', 'distilroberta-base', 'distilgpt2']