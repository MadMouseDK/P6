import os
import pandas as pd
import warnings
import re
from typing import Tuple
from pandas import DataFrame
import pickle


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
        train_bronze_df = pd.read_json(os.path.join(data_path, "bronze_quality/json_format/train_bronze.json"), orient="index")
        return pd.concat([train_gold_df, train_silver2025_df, train_silver_df, train_bronze_df], axis=0)
    

def extracts_entities(df: pd.DataFrame) -> pd.DataFrame:
    entities_df_collapsed = df[["entities"]]
    entities_df = entities_df_collapsed.explode("entities").reset_index(names = "pmid")
    entities_expanded = pd.json_normalize(entities_df["entities"])
    entities_df = entities_df.drop(columns=["entities"]).join(entities_expanded)
    
    return entities_df

def extracts_articles(df: pd.DataFrame) -> pd.DataFrame:
    articles_df = df.drop(columns = ["entities"]).reset_index(names = "pmid")
    
    return articles_df


def data_preprocessing(df: DataFrame) -> DataFrame:
    
    articles = extracts_articles(df)
    entities = extracts_entities(df)
    
    results = {}

    for ent_index, ent_row in entities.iterrows():
        pmid = ent_row.pmid
    
        # Find matching article
        art_row = articles[articles['pmid'] == pmid]
        if art_row.empty:
            continue
        art_row = art_row.iloc[0]
    
        # Initialize entry if first time seeing this pmid
        if pmid not in results:
            results[pmid] = {
                'title': art_row['title'],  
                'title_entities': {},
                'title_spans': [],
                'abstract': art_row['abstract'] if pd.notna(art_row['abstract']) else [],
                'abstract_entities': {},
                'abstract_spans': []
                }
    
        # Extract entity info
        entity_info = (ent_row['start_idx'], ent_row['end_idx'], ent_row['label'])
    
    
        # Add to appropriate dict
        if ent_row['location'] == 'title':
            if results[pmid]['title_entities']:
                new_key = max(results[pmid]['title_entities'].keys()) + 1
            else:
                new_key = 0
            results[pmid]['title_entities'][new_key] = entity_info
            span_info = ent_row['text_span']
            results[pmid]['title_spans'].append(span_info)
        elif ent_row['location'] == 'abstract':
            if results[pmid]['abstract_entities']:
                new_key = max(results[pmid]['abstract_entities'].keys()) + 1
            else:
                new_key = 0
            results[pmid]['abstract_entities'][new_key] = entity_info
            span_info = ent_row['text_span']
            results[pmid]['abstract_spans'].append(span_info)
        

    proccessed_data = []
    for pmid, entry in results.items():
        # Title
        title_text = entry['title']
        title_entities = [(start, end, label) for key, (start, end, label) in entry['title_entities'].items()]
        title_spans = [i for i in entry['title_spans']]
        proccessed_data.append((title_text, {"entities": title_entities, "text_spans": title_spans, "source": "title", "pmid": pmid}))
        
        # Abstract
        if entry['abstract']:
            abstract_text = entry['abstract']
            abstract_entities = [(start, end, label) for key, (start, end, label) in entry['abstract_entities'].items()]
            abstract_spans = [i for i in entry['abstract_spans']]
            proccessed_data.append((abstract_text, {"entities": abstract_entities, "text_spans": abstract_spans, "source": "abstract", "pmid": pmid}))

    return proccessed_data


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


def main(datatype: str):
    cwd = os.getcwd()
    if cwd.endswith("Utils"):
        cwd = os.path.dirname(cwd)
    
    df = load_dataset(datatype, cwd)
    df = df[["metadata", "entities"]]
    metadata_df = pd.json_normalize(df["metadata"])
    metadata_df = metadata_df.drop(["author", "journal", "year"], axis = 1)
    df = pd.merge(metadata_df, df["entities"], left_index = True, right_index = True)
    df = data_preprocessing(df)

    num_warnings = 0

    corrected = []
    for i, (text, annotations) in enumerate(df):
        text = clean_html_tags(text)
        
        for j in range(len(annotations['entities'])):
            text_span = annotations['text_spans'][j]
            start = annotations['entities'][j][0]
            end = annotations['entities'][j][1]

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

            corrected.append({
                "idx": i,
                "entity_j": j,  
                "original_span": annotations['text_spans'][j],
                "corrected_span": text_span,
                "start": start,
                "end": end,
            })
    
    inspected_df = pd.DataFrame(corrected)
    
    corrections = (
        inspected_df
        .set_index(['idx', 'entity_j'])[['corrected_span', 'start', 'end']]
        .to_dict(orient='index')
    )
    
    corrected_df = []
    for i, (text, annotations) in enumerate(df):
        new_entities = list(annotations['entities'])
        new_text_spans = list(annotations['text_spans'])
    
        for j in range(len(annotations['entities'])):
            if (i, j) in corrections:
                corrected_span = corrections[(i, j)]['corrected_span']
                start = int(corrections[(i, j)]['start'])
                end = int(corrections[(i, j)]['end'])
                label = annotations['entities'][j][2]
                new_entities[j] = (start, end, label)
                new_text_spans[j] = corrected_span
    
        new_annotations = {
            **annotations,
            'entities': new_entities,
            'text_spans': new_text_spans,
        }
        corrected_df.append((text, new_annotations))
        
    print(num_warnings)
    
    path = os.path.join(cwd, "Data", f"{datatype}", f"{datatype}.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(corrected_df, f)


if __name__ == "__main__":
    main("train")
    main("dev")
    

