import os

import pandas as pd

from Utils.processing import (
    change_index,
    load_dataset,
    to_docbin,
    to_pickle,
    valid_text_span,
)


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
    silvera_df = df[df["annotator"] == "silver_a"]
    silverb_df = df[df["annotator"] == "silver_b"]
    bronze_df = df[df["annotator"] == "bronze"]

    gold_df = gold_df.sample(frac=fractions["gold"], replace=True)
    silvera_df = silvera_df.sample(frac=fractions["silverA"], replace=True)
    silverb_df = silverb_df.sample(frac=fractions["silverB"], replace=True)
    bronze_df = bronze_df.sample(frac=fractions["bronze"], replace=True)

    return pd.concat([gold_df,
                      silvera_df,
                      silverb_df,
                      bronze_df])


def map_relations(relation_df: pd.DataFrame, entities_df: pd.DataFrame) -> pd.DataFrame:
    relation_df = relation_df.merge(
        entities_df,
        left_on=["pmid", "subject_location", "subject_start_idx", "subject_end_idx"],
        right_on=["pmid", "location", "start_idx", "end_idx"]
    )
    relation_df = relation_df.rename(columns={"id": "subject_id"})

    relation_df = relation_df.merge(
        entities_df,
        left_on=["pmid", "object_location", "object_start_idx", "object_end_idx"],
        right_on=["pmid", "location", "start_idx", "end_idx"]
    )
    relation_df = relation_df.rename(columns={"id": "object_id"})
    return relation_df.drop(columns=["subject_start_idx",
                                     "subject_end_idx",
                                     "subject_location",
                                     "object_start_idx",
                                     "object_end_idx"])


def merge_relations(relations_df: pd.DataFrame, entities_df: pd.DataFrame, on: str) -> pd.DataFrame:
    if not (on == "subject_id" or on == "object_id"):
        raise ValueError("Not valid argument for on. Must be either subject_id or object_id")

    relations_df = relations_df.merge(
        entities_df[["id", "start_idx", "end_idx"]],
        left_on=on,
        right_on="id"
    )
    column_prefix = on.removesuffix("_id")
    return relations_df.rename(columns={
        "start_idx" : f"{column_prefix}_start_idx",
        "end_idx": f"{column_prefix}_end_idx"}
    )


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
    relations_df = map_relations(relations_df, entities_df)

    # Cleans entities and filter out those not valid
    entities_cleaned = entities_df.merge(annotations_df, on=["pmid", "location"])
    entities_cleaned[["start_idx", "end_idx", "text_span"]] = entities_cleaned.apply(clean_entity, axis=1, result_type="expand")
    entities_cleaned = entities_cleaned.drop(columns=["text", "annotator"])
    entities = entities_cleaned[entities_df["start_idx"] != -1]

    # Groups entities per document
    entities["entities"] = entities[["start_idx", "end_idx", "label", "uri"]].to_numpy().tolist()
    entities_grouped = entities.groupby(["pmid", "location"]).agg({
        "entities": list,
        "text_span": list
    }).reset_index()

    relations_df = merge_relations(relations_df, entities, "subject_id")
    relations_df = merge_relations(relations_df, entities, "object_id")

    # Filter out relations with invalid entities and groups per document
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

def main(datatype: str, use_sampling: bool = False, sampling_fractions: dict | None = None):
    cwd = os.getcwd()
    if cwd.endswith("src"):
        cwd = os.path.dirname(cwd)


    df = load_dataset(datatype, cwd)
    df = df[["metadata", "entities", "relations"]]
    metadata_df = pd.json_normalize(df["metadata"])
    metadata_df = metadata_df.drop(["author", "journal", "year"], axis = 1)
    df = metadata_df.merge(df[["entities", "relations"]], left_index = True, right_index = True)

    # Changes annotation to align with dataset description
    df.loc[df["annotator"].str.startswith("expert"), "annotator"] = "gold"
    df.loc[df["annotator"].str.endswith("A"), "annotator"] = "silver_a"
    df.loc[df["annotator"].str.endswith("B"), "annotator"] = "silver_b"
    df.loc[df["annotator"] == "distant", "annotator"] = "bronze"

    entities_df = expand_entries(df, "entities")
    relations_df = df[df["relations"].astype(bool)]
    relations_df = expand_entries(relations_df, "relations")
    relations_df = relations_df[relations_df["subject_location"] == relations_df["object_location"]]
    annotations_df = extracts_articles(df)

    if use_sampling and datatype == "train":
        assert isinstance(sampling_fractions, dict), "Need to asign fraction if sampling is true"

    entities_df = entities_df.reset_index(names="id")
    relations_df = map_relations(relations_df, entities_df)

    # Cleans entities and filter out those not valid
    entities_cleaned = entities_df.merge(annotations_df, on=["pmid", "location"])
    entities_cleaned[["start_idx", "end_idx", "text_span"]] = entities_cleaned.apply(clean_entity, axis=1, result_type="expand")
    entities_cleaned = entities_cleaned.drop(columns=["text", "annotator"])
    entities = entities_cleaned[entities_df["start_idx"] != -1]

    # Groups entities per document
    entities["entities"] = entities[["start_idx", "end_idx", "label", "uri"]].to_numpy().tolist()
    entities_grouped = entities.groupby(["pmid", "location"]).agg({
        "entities": list,
        "text_span": list
    }).reset_index()

    # Obtains the updated entity information after cleaning them
    relations_df = merge_relations(relations_df, entities, "subject_id")
    relations_df = merge_relations(relations_df, entities, "object_id")

    # Filter out relations with invalid entities and groups per document
    relations_df = relations_df[(relations_df["subject_start_idx"] != -1) & (relations_df["object_start_idx"] != -1)]
    relations_df["relations"] = relations_df[["subject_start_idx", "subject_uri", "predicate", "object_start_idx", "object_uri"]].to_numpy().tolist()
    relations_grouped = relations_df.groupby(["pmid", "object_location"]).agg({
        "relations": list
    }).reset_index()

    # Merges the needed data from each dataframe
    cleaned_data = annotations_df.merge(entities_grouped, on=["pmid", "location"])
    cleaned_data = cleaned_data.merge(relations_grouped, left_on=["pmid", "location"], right_on=["pmid", "object_location"])

    if use_sampling:
        cleaned_data = sample_data(cleaned_data, sampling_fractions)

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

    path = os.path.join(cwd, "Data", "processed")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_pickle(path, f"{datatype}.pkl", results)

    spacy_path = os.path.join(cwd, "Data", "processed", f"{datatype}_data_NER.spacy")
    doc_bin = to_docbin(results)
    doc_bin.to_disk(spacy_path)


if __name__ == "__main__":
    main("train")
    main("dev")
