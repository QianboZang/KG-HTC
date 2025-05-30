import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import json
import random
# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm_ollama import LLM
from src.graph_db import GraphDB
from src.vector_db import VectorDB
from src.pipeline import Pipeline


config = {
    "data_name": "dbpedia",
    "model_type": "qwen2.5:7b",
    "data_path": f"dataset/dbpedia/DBPEDIA_val.csv",
    "output_path": "dataset/open_llm/dbpedia_qwen7b.json",
    "vectdb_path": "database/dbpedia",
    "template": {
        "sys": "prompts/system/dbpedia/llm_graph.txt",
        "user": "prompts/user/dbpedia/llm_graph.txt"
    },
    "query_params": {
        "l2_top_k": 10,
        "l3_top_k": 40
    }
}

# read csv file
df = pd.read_csv(config["data_path"])
df = df.sample(n=5000, random_state=42)
ds = df.to_dict(orient="records")

llm = LLM(model_type=config["model_type"])
graph_db = GraphDB()
pipeline = Pipeline(llm, config)

inference_list = []
for idx in tqdm(range(len(ds))):
    data = ds[idx].copy()

    query_txt_vecdb = data["text"]
    retrieved_nodes = pipeline.query_related_nodes(query_txt_vecdb)
    sub_graph = pipeline.build_linked_labels(retrieved_nodes["l3"], retrieved_nodes["l2"])

    potential_level1 = df["l1"].unique()
    pred_level1 = pipeline.predict_level(
        query_txt_vecdb, 
        potential_level1, 
        sub_graph
    ).lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')

    child_level1 = graph_db.query_l2_from_l1(pred_level1)
    # print(child_level1)
    potential_level2 = list(set(child_level1 + retrieved_nodes["l2"]))
    pred_level2 = pipeline.predict_level(
        query_txt_vecdb, 
        potential_level2, 
        sub_graph
    ).lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')

    child_level2 = graph_db.query_l3_from_l2(pred_level2)
    # print(child_level2)
    potential_level3 = list(set(child_level2 + retrieved_nodes["l3"]))
    pred_level3 = pipeline.predict_level(
        query_txt_vecdb, 
        potential_level3, 
        sub_graph
    ).lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')
    
    data["qwen7b_l1"], data["qwen7b_l2"], data["qwen7b_l3"] = pred_level1, pred_level2, pred_level3
    inference_list.append(data)
    # print(data)

    with open(config["output_path"], "w") as f:
        json.dump(inference_list, f, indent=4)
