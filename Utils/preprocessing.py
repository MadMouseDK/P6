import os
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
def load_dataset(data: str):
    cwd = os.getcwd()
    if cwd.endswith("Utils"):
        cwd = os.path.dirname(cwd)
    
    if data == "dev":
        data_path = os.path.join(cwd, "Data/raw/GutBrainIE_Full_Collection_2026/Annotations/Dev/json_format/dev.json")
        df = pd.read_json(data_path, orient="index")
    elif data == "train":
        data_path = os.path.join(cwd, "Data/raw/GutBrainIE_Full_Collection_2026/Annotations/Train/")
        train_gold_df = pd.read_json(os.path.join(data_path, "gold_quality/json_format/train_gold.json"), orient="index")
        train_silver_df = pd.read_json(os.path.join(data_path, "silver_quality/json_format/train_silver.json"), orient="index")
        train_silver2025_df = pd.read_json(os.path.join(data_path, "silver_quality/json_format/train_silver_2025.json"), orient="index") 
        train_bronze_df = pd.read_json(os.path.join(data_path, "bronze_quality/json_format/train_bronze.json"), orient="index")
        df = pd.concat([train_gold_df, train_silver2025_df, train_silver_df, train_bronze_df], axis=0)
    
    df = df[["metadata", "entities"]]
    metadata_df = pd.json_normalize(df["metadata"])
    metadata_df = metadata_df.drop(["author", "journal", "year"], axis=1)

    return pd.merge(metadata_df, df["entities"], left_index=True, right_index=True)

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
            if entity["location"] == "title":
                title_entites.append((start_idx, end_idx + 1, label))
            else:
                abstract_entities.append((start_idx, end_idx + 1, label))
        
        if len(title_entites) != 0:
            title_information[index] = {"text": row["title"], "entities": title_entites}
        
        if len(abstract_entities) != 0:
            abstract_information[index] = {"text": row["abstract"], "entities": abstract_entities}

    df_title = pd.DataFrame(title_information).T
    df_abstract = pd.DataFrame(abstract_information).T
    full_df = pd.concat([df_title, df_abstract])
    return full_df


def validate():
    pass



def main(datatype: str = "dev"):
    
    df = load_dataset(datatype)
    df = extracts_entities(df).to_numpy()

    model = spacy.blank("en")
    db = DocBin()

    for text, annotations in df:
        doc = model(text)
        entities = []

        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            if span is None:
                continue
            entities.append(span)
        entities = filter_spans(entities)
        doc.set_ents(entities)
        db.add(doc)
    path = os.path.join(os.getcwd(), f"Data/{datatype}/{datatype}.spacy")
    db.to_disk(path)

if __name__ == "__main__":
    main()
    

    
