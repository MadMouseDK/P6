import os
import warnings
import re
import pickle
import pandas as pd
from spacy.tokens import DocBin
import spacy


def load_dataset(data: str, root_path: str) -> pd.DataFrame | None:
    """Loads the dataset and returns a pandas dataframe if succesful otherwise return None"""
    allowed_values = ["train", "dev"]
    if data not in allowed_values:
        raise ValueError(f"{data} value not allowed. Allowed values: {allowed_values}")

    data_path = os.path.join(root_path, "Data", "raw", "Annotations")
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
    return None


def expand_entries(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        print(f"{column} is not a column in the provided dataframe")
        raise IndexError()
    df = df[[column]].explode(column=column).reset_index(names="pmid")
    expanded_df = pd.json_normalize(df[column])
    return df.drop(columns=[column]).join(expanded_df)


def extracts_articles(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["title", "abstract", "annotator"]]
    df = df.reset_index(names = "pmid")

    titles = df[["pmid", "title", "annotator"]]
    titles["location"] = ["title"] * len(titles)
    titles = titles.rename(columns={"title": "text"})

    abstracts = df[["pmid", "abstract", "annotator"]]
    abstracts["location"] = ["abstract"] * len(abstracts)
    abstracts = abstracts.rename(columns={"abstract": "text"})

    return pd.concat([abstracts, titles])


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
    return re.match(ends_with, text_span) is None


def change_index(text: str, text_span: str, start_idx: int, end_idx: int) -> tuple[int, int]:
    """Takes a text and the text span along with the indexes and make sure the the text is correct and does not miss or include anything it shouldn't"""
    if re.match(r"\A(\)|\]|\s)", text_span) is not None:
        msg = f"text_span {text_span} starts with ) or ] which is not valid"
        warnings.warn(msg, stacklevel=2)
        start_idx += 1

    if text_span.startswith("#") and re.match(r"(?<=\w)#", text_span) is None:
        msg = f"text_span {text_span} has html in beggining but not anywhere else"
        warnings.warn(msg, stacklevel=2)
        start_idx += 1

    if start_idx >= 1:
        if re.match(r"[^\d\w]", text[start_idx - 1]) is None:
            start_idx -= 1
            msg = f"text_span {text_span} does not capture the entire word: {text[start_idx:end_idx]}"
            warnings.warn(msg, stacklevel=2)

    if end_idx < (len(text) - 1):
        char = text[end_idx + 1]
        if char.isdigit() or char.isalpha():
            end_idx += 1
            msg = f"text_span {text_span} does not capture the entire word: {text[start_idx:end_idx]}"
            warnings.warn(msg, stacklevel=2)

    return (start_idx, end_idx)


def clean_entity(row: pd.Series):
    """Changes the index for word if it does not capture the entire word and validates the word.

    If the word is not valid then the start_idx and end_idx will be set to -1, while text_span will become ""

    Returns the updated indexes and text_span
    """
    start = row.start_idx
    end = row.end_idx
    text_span = row.text_span
    text = row.text
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

    if not valid_text_span(text_span):
        start = -1
        end = -1
        text_span = ""

    return start, end, text_span


def sample_data(df: pd.DataFrame, fractions: dict) -> pd.DataFrame:
    gold_df = df[df["annotator"] == "gold"]
    silverA_df = df[df["annotator"] == "silver_a"]
    silverB_df = df[df["annotator"] == "silver_b"]
    bronze_df = df[df["annotator"] == "bronze"]

    gold_df = gold_df.sample(frac=fractions["gold"], replace=True)
    silverA_df = silverA_df.sample(frac=fractions["silverA"], replace=True)
    silveB_df = silverB_df.sample(frac=fractions["silverB"], replace=True)
    bronze_df = bronze_df.sample(frac=fractions["bronze"], replace=True)

    return pd.concat([gold_df,
                      silverA_df,
                      silverB_df,
                      bronze_df])


def preprocessing(annotations_df: pd.DataFrame,
                  entities_df: pd.DataFrame,
                  relations_df: pd.DataFrame,
                  use_sampling: bool = False,
                  fractions: dict | None = None
                )-> list:
    """Takes the dataframes and cleans the data before storing it in a list"""
    if use_sampling:
        assert isinstance(fractions, dict), "Need to asign fraction if sampling is true"

    entities_df = entities_df.reset_index(names="id")
    relations_df = relations_df.merge(
        entities_df,
        left_on=["pmid", "subject_location", "subject_start_idx", "subject_end_idx"],
        right_on=["pmid", "location", "start_idx", "end_idx"]
    )
    relations_df = relations_df.rename(columns={"id": "subject_id"})

    relations_df = relations_df.merge(
        entities_df,
        left_on=["pmid", "object_location", "object_start_idx", "object_end_idx"],
        right_on=["pmid", "location", "start_idx", "end_idx"]
    )
    relations_df = relations_df.rename(columns={"id": "object_id"})
    relations_df = relations_df.drop(columns=["subject_start_idx", "subject_end_idx", "subject_location", "object_start_idx", "object_end_idx"])

    entities_cleaned = entities_df.merge(annotations_df, on=["pmid", "location"])
    entities_cleaned[["start_idx", "end_idx", "text_span"]] = entities_cleaned.apply(clean_entity, axis=1, result_type="expand")
    entities_cleaned = entities_cleaned.drop(columns=["text", "annotator"])


    entities = entities_cleaned[entities_cleaned["start_idx"] != -1]
    entities["entities"] = entities[["start_idx", "end_idx", "label", "uri"]].to_numpy().tolist()

    entities_grouped = entities.groupby(["pmid", "location"]).agg({
        "entities": list,
        "text_span": list
    }).reset_index()

    relations_df = relations_df.merge(
        entities[["id", "start_idx", "end_idx"]],
        left_on="subject_id",
        right_on="id"
    )
    relations_df = relations_df.rename(columns={"start_idx" : "subject_start_idx", "end_idx": "subject_end_idx"})

    relations_df = relations_df.merge(
        entities[["id", "start_idx", "end_idx"]],
        left_on="object_id",
        right_on="id"
    )
    relations_df = relations_df.rename(columns={
        "start_idx" : "object_start_idx",
        "object_idx": "object_end_idx"}
    )

    relations_df = relations_df[(relations_df["subject_start_idx"] != -1) & (relations_df["object_start_idx"] != -1)]
    relations_df["relations"] = relations_df[["subject_start_idx", "subject_uri", "predicate", "object_start_idx", "object_uri"]].to_numpy().tolist()
    relations_grouped = relations_df.groupby(["pmid", "object_location"]).agg({
        "relations": list
    }).reset_index()

    cleaned_data = annotations_df.merge(entities_grouped, on=["pmid", "location"])
    cleaned_data = cleaned_data.merge(relations_grouped, left_on=["pmid", "location"], right_on=["pmid", "object_location"])

    if use_sampling:
        cleaned_data = sample_data(cleaned_data, fractions)
    results = []
    for _, row in cleaned_data.iterrows():
        entry = (
            row.text,
            {
                "entities": row.entities,
                "text_span": row.text_span,
                "relations": row.relations,
                "location": row.location,
                "pmid": row.pmid
            }
        )
        results.append(entry)

    return results

def results_to_spacy(results: list) -> DocBin:
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    
    for text, metadata in results:
        doc = nlp.make_doc(text)
        
        if metadata.get("entities"):
            ents = []
            for start, end, label, uri in metadata["entities"]:
                # Check if the span is correct - if not, try adjusting.
                # Had problem preserving spans when converting to spacy.
                current_span = text[start:end]
                
                span = doc.char_span(start, end, label=label)
                
                if span is None and end < len(text):
                    span = doc.char_span(start, end + 1, label=label)
                
                if span is None and start > 0 and end < len(text):
                    span = doc.char_span(start - 1, end + 1, label=label)
                
                if span is not None:
                    ents.append(span)
            
            if ents:
                ents = sorted(ents, key=lambda x: (x.start_char, x.end_char))
                
                filtered_ents = []
                for ent in ents:
                    overlap = False
                    for added_ent in filtered_ents:
                        if not (ent.end_char <= added_ent.start_char or ent.start_char >= added_ent.end_char):
                            overlap = True
                            break
                    if not overlap:
                        filtered_ents.append(ent)
                
                if filtered_ents:
                    try:
                        doc.ents = filtered_ents
                    except Exception as e:
                        print(f"Warning: Could not set entities: {e}")
        
        if not doc.has_extension("custom_metadata"):
            spacy.tokens.Doc.set_extension("custom_metadata", default=None)
        doc._.custom_metadata = metadata
        
        doc_bin.add(doc)
    
    return doc_bin

def main(datatype: str):
    cwd = os.getcwd()
    if cwd.endswith("Utils"):
        cwd = os.path.dirname(cwd)
        cwd = os.path.dirname(cwd)


    df = load_dataset(datatype, cwd)
    df = df[["metadata", "entities", "relations"]]
    metadata_df = pd.json_normalize(df["metadata"])
    metadata_df = metadata_df.drop(["author", "journal", "year"], axis = 1)
    df = metadata_df.merge(df[["entities", "relations"]], left_index = True, right_index = True)

    df.loc[df["annotator"].str.startswith("expert"), "annotator"] = "gold"
    df.loc[df["annotator"].str.endswith("A"), "annotator"] = "silver_a"
    df.loc[df["annotator"].str.endswith("B"), "annotator"] = "silver_b"
    df.loc[df["annotator"] == "distant", "annotator"] = "bronze"

    entities_df = expand_entries(df, "entities")
    relations_df = df[df["relations"].astype(bool)]
    relations_df = expand_entries(relations_df, "relations")
    relations_df = relations_df[relations_df["subject_location"] == relations_df["object_location"]]
    annotations_df = extracts_articles(df)

    results = preprocessing(annotations_df, entities_df, relations_df)
    
    path = os.path.join(cwd, "Data", "processed", f"{datatype}.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)

    spacy_path = os.path.join(cwd, "Data", "processed", f"{datatype}_data_NER.spacy")
    doc_bin = results_to_spacy(results)
    doc_bin.to_disk(spacy_path)

if __name__ == "__main__":
    main("train")
    main("dev")

