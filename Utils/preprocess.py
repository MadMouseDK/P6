import os
import pandas as pd
import warnings
import spacy
import re
from spacy.tokens import DocBin
from spacy.util import filter_spans
from typing import Tuple



def load_data(path:str=os.getcwd(), type:str="train"):
    type = type.lower()
    path = os.path.join(path, "Data", "raw","GutBrainIE_Full_Collection_2026", "Annotations" )
    if type == "train":
        path = os.path.join(path, "Train", "gold_quality", "json_format","train_gold.json")
        data = pd.read_json(path, orient="index")
    elif type == "dev":
        path = os.path.join(path, "Dev")
    else:
        raise ValueError("Write: 'train' or 'dev'")
    
    return data

def clean_data(data:pd.DataFrame):

    data = data[["metadata", "entities"]]
    temp_data = pd.json_normalize(data["metadata"]).drop(["author", "journal", "year"], axis=1)
    data = pd.merge(temp_data, data["entities"], left_index=True, right_index=True)
    data = extracts_entities(data)#.to_numpy()
    return data

def tokenize_data():
    raise NotImplemented

def extracts_entities(df: pd.DataFrame) -> pd.DataFrame:
    title_information = dict()
    abstract_information = dict()
    for index, row in df.iterrows():
        if not isinstance(row["entities"], list):
            continue

        title_entites = []
        abstract_entities = []
        for entity in row["entities"]:
            start_idx, end_idx, label = entity["start_idx"], entity["end_idx"], entity["label"]
            text_span = entity["text_span"]
            if entity["location"] == "title":
                title_entites.append((start_idx, end_idx + 1, label, text_span))
            else:
                abstract_entities.append((start_idx, end_idx + 1, label, text_span))
        
        if len(title_entites) != 0:
            title_information[index] = {"text": row["title"], "entities": title_entites}
        
        if len(abstract_entities) != 0:
            abstract_information[index] = {"text": row["abstract"], "entities": abstract_entities}

    df_title = pd.DataFrame(title_information).T
    df_abstract = pd.DataFrame(abstract_information).T
    full_df = pd.concat([df_title, df_abstract])
    return full_df

def spacy_tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

def normalize(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

data = load_data()
data = clean_data(data)
nlp = spacy.load("en_core_web_sm")
#print(data.head()) 


data["tokens"] = data["text"].apply(spacy_tokenize)
data['normalized_tokens'] = data['text'].apply(normalize)

print(data)