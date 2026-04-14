import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import pickle
import os
from spacy.util import filter_spans
import subprocess
import sys


def load_pickle(datatype: str) -> pd.DataFrame:
    allowed_values = ["train", "dev"]
    if datatype not in allowed_values:
        raise ValueError(f"{datatype} value not allowed. Allowed values: {allowed_values}")
    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if cwd.endswith(folder):
        cwd = os.path.dirname(cwd)
        path = os.path.join(cwd, "Data", "processed", f"{datatype}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

train_data = load_pickle("train")
dev_data = load_pickle("dev")

nlp = spacy.blank('en')


def create_spacy_docbin(data):
    
    db = DocBin()
    entities_skipped = 0
    start_end_idx = []
    
    for i, (text, annot) in enumerate(tqdm(data)):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            if start > end:
                print(f"Wrong start: {start}, end: {end}")
                start, end = end, start
            span = doc.char_span(start, end, label = label, alignment_mode = "expand")
            if span is None:
                entities_skipped += 1
                start_end_idx.append((start, end))
            else:
                ents.append(span)
    
        ents = filter_spans(ents)
        try:
            doc.ents = ents
        except ValueError as e:
            print(f"Skipping doc {i}: {e}")
            continue
        
        db.add(doc)
    return db


train = create_spacy_docbin(train_data)
dev = create_spacy_docbin(dev_data)

def safe_spacy(data_spacy, name:str):
    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if cwd.endswith(folder):
        cwd = os.path.dirname(cwd)
        path = os.path.join(cwd, "Data", "processed", f"{name}_data_NER.spacy")
    data_spacy.to_disk(path)


safe_spacy(train, "train")
safe_spacy(dev, "dev")


def train_spacy_model(model_name:str):
    
    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if cwd.endswith(folder):
        project_dir = os.path.dirname(cwd)
    config_dir = os.path.join(cwd, "cfg")
    base_config_path = os.path.join(config_dir, f"base_config_{model_name}_NER.cfg")
    config_path = os.path.join(cwd, "cfg", f"config_{model_name}_NER.cfg")
    output_path = os.path.join(project_dir, "Models", f"{model_name}_NER", "output")
    

    subprocess.run(
        [sys.executable, "-m", "spacy", "init", "fill-config", base_config_path, config_path],
        cwd = project_dir,
        check = True
    )
    

    result = subprocess.run(
        [sys.executable, "-m", "spacy", "train", config_path, "--output", output_path],
        cwd = project_dir,
        check = True
    )
    
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")


train_spacy_model("RoBERTa")
train_spacy_model("SciBERT")
train_spacy_model("BlueBERT-pubmed_NER")
train_spacy_model("BlueBERT-pubmed-mimic_NER")
train_spacy_model("BioBERT-base_NER")
train_spacy_model("BioBERT-large_NER")

