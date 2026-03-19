import os
import pandas as pd
import warnings
import spacy
import re
from spacy.tokens import DocBin
from spacy.util import filter_spans
from typing import Tuple


def load_dataset(data: str, root_path: str) -> pd.DataFrame:
    """Loads the dataset a"""
    allowed_values = ["train", "dev"]
    if data not in allowed_values:
        raise ValueError(f"{data} value not allowed. Allowed values: {allowed_values}")

    data_path = os.path.join(root_path, "Data", "raw", "GutBrainIE_Full_Collection_2026", "Annotations")
    if data == "dev":
        data_path = os.path.join(data_path, "Dev", "json_format", "dev.json")
        return pd.read_json(data_path, orient="index")
    elif data == "train":
        data_path = os.path.join(data_path, "Train")
        train_gold_df = pd.read_json(os.path.join(data_path, "gold_quality", "json_format", "train_gold.json"), orient="index")
        train_silver_df = pd.read_json(os.path.join(data_path, "silver_quality", "json_format", "train_silver.json"), orient="index")
        train_silver2025_df = pd.read_json(os.path.join(data_path, "silver_quality", "json_format", "train_silver_2025.json"), orient="index") 
        #train_bronze_df = pd.read_json(os.path.join(data_path, "bronze_quality/json_format/train_bronze.json"), orient="index")
        return pd.concat([train_gold_df, train_silver2025_df, train_silver_df], axis=0)
    

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


def clean_html_tags(text: str, replacement: str = "#")  -> str:
    """Cleans HTML tags for some text and replaces it with a special charater"""
    return re.sub(r"<\/?[a-z][a-z0-9]*[^<>]*>|<!--.*?-->",lambda match: replacement * len(match.group(0)), text)

def valid_text_span(text_span: str) -> bool:
    """Returns true or false depeding on wether the text span is valid
    This can be cases where the text only include special charaters"""
    text_span = text_span.lower()

    if len(text_span) == 1 and text_span not in ["C", "A"]:
        return False
    
    regex = re.match(r"[^\,\.\-\'\*\;\:\<\>\!\&\#\s]", text_span)
    if regex is not None:
        if len(regex.group()) == 0:
            return False
    
    entities = ["anatomical location", "animal", "statistical technique", "biomedical technique", "bacteria", "chemical", "dietary supplement", "disease", "disorder", "finding", "drug", "food", "gene", "human", "microbiome"]
    if text_span in entities:
        return False
    
    ends_with = r"(system|systems|axis|model|models|response|mechanism|theraphy|symptoms|pathogenesis|destruction)\b$"
    if re.match(ends_with, text_span) is not None:
        return False
    
    return True


def change_index(text: str, text_span: str, start_idx: int, end_idx: int) -> Tuple[int, int]:
    """Takes a text and the text span along with the indexes and make sure the the text is correct and does not miss or include anything it shouldn't"""
    if re.match(r"\A(\)|\]|\s)", text_span) is not None:
        msg = f"text_span {text_span} starts with ) or ] which is not valid"
        warnings.warn(msg)
        start_idx += 1
    
    if text_span.startswith("#") and re.match(r"(?<=\w)#", text_span) is None:
        msg = f"text_span {text_span} has html in beggining but not anywhere else"
        warnings.warn(msg)
        start_idx += 1
    
    if start_idx >= 1:
        if re.match(r"[^\d\w]", text[start_idx - 1]) is None:
            start_idx -= 1
            msg = f"text_span {text_span} does not capture the entire word: {text[start_idx:end_idx]}"
            warnings.warn(msg)
    
    if end_idx < len(text):
        if re.match(r"[^\d\w]", text[end_idx]) is None:
            end_idx += 1
            msg = f"text_span {text_span} does not capture the entire word: {text[start_idx:end_idx]}"
            warnings.warn(msg)
    
    return (start_idx, end_idx)


def preprocessing(datatype: str = "train"):
    cwd = os.getcwd()
    if cwd.endswith("Utils"):
        cwd = os.path.dirname(cwd)
    
    df = load_dataset(datatype, cwd)
    df = df[["metadata", "entities"]]
    metadata_df = pd.json_normalize(df["metadata"])
    metadata_df = metadata_df.drop(["author", "journal", "year"], axis=1)
    df = pd.merge(metadata_df, df["entities"], left_index=True, right_index=True)
    df = extracts_entities(df).to_numpy()
    model = spacy.blank("en")
    db = DocBin()
    num_warnings = 0

    for i, (text, annotations) in enumerate(df):
        text = clean_html_tags(text)
        doc = model(text)
        entities = []

        for start, end, label, text_span in annotations:
            if not valid_text_span(text_span):
                msg = f"Text span {text_span} is not valid for index {i}"
                warnings.warn(msg)
                continue
        
            while True:
                new_start_idx, new_end_idx = change_index(text, text_span, start, end)
                if new_start_idx == start and new_end_idx == end:
                    break
                start = new_start_idx
                end = new_end_idx
                text_span = text[start:end]

                if text_span.startswith("Background"):
                    text_span = text[start+10:end]
                    break
            
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                msg = f"Skipping [{start}, {end}, {label}] since it does match {doc.text[start:end]} should be {text_span} at index {i}"
                warnings.warn(msg)
                num_warnings += 1
                continue
            entities.append(span)
        entities = filter_spans(entities)
        doc.set_ents(entities)
        db.add(doc)
    
    path = os.path.join(os.getcwd(), "Data", f"{datatype}", f"{datatype}.spacy")
    db.to_disk(path)
    print(num_warnings)

if __name__ == "__main__":
    preprocessing()
    

    
