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
        path = os.path.join(cwd, "Data", f"{datatype}", f"{datatype}.pkl")
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
        path = os.path.join(cwd, "Data", f"{name}", f"{name}_data_NER.spacy")
    data_spacy.to_disk(path)


safe_spacy(train, "train")
safe_spacy(dev, "dev")


def train_spacy_model(config_dir: str, model_name:str):

    subprocess.run(
        [sys.executable, "-m", "spacy", "init", "fill-config", f"base_config_{model_name}_NER.cfg", f"config_{model_name}_NER.cfg"],
        cwd = config_dir,
        check = True
    )
        
    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if cwd.endswith(folder):
        models_path = os.path.dirname(cwd)
 
    

    result = subprocess.run(
        [sys.executable, "-m", "spacy", "train", f"config_{model_name}_NER.cfg", "--output", f"{models_path}/Models/{model_name}_NER/output"],
        cwd = config_dir,
        check = True
    )
    
    #if result.returncode != 0:
        #print(f"Training failed with exit code {result.returncode}")


train_spacy_model(os.getcwd(), "RoBERTa")
train_spacy_model(os.getcwd(), "SciBERT")
train_spacy_model(os.getcwd(), "RoBERTa")
train_spacy_model(os.getcwd(), "BlueBERT-pubmed-L12_NER")
train_spacy_model(os.getcwd(), "BlueBERT-pubmed-mimic-L12_NER")
train_spacy_model(os.getcwd(), "BioBERT-base_NER")
train_spacy_model(os.getcwd(), "BioBERT-large_NER")





