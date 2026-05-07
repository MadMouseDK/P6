import pandas as pd
import spacy
import os
import json


def sub_scr(model: str):
    path = os.getcwd()
    if path.endswith("Utils"):
        path = os.path.dirname(path)
        path = os.path.dirname(path)
    data_path = os.path.join(path, "Data", "raw", "Annotations", "Test", "articles_test.csv")
    test_set = pd.read_csv(data_path, sep = '|')
    test_set = test_set[["pmid","title", "abstract"]]
    allowed_values = ["RoBERTa",
                      "RoBERTa-biomed",
                      "SciBERT",
                      "BioBERT-base",
                      "BlueBERT-pubmed",
                      "BlueBERT-pubmed-mimic"]
    if model not in allowed_values:
        raise ValueError(f"{model} value not allowed. Allowed values: {allowed_values}")
    model_path = os.path.join(path, "Models", f"{model}_NER", "output", "model-best")
    trained_model = spacy.load(model_path)
    result = {}
    for article in test_set.iterrows():
        entities = []
        print(f"Processing article {article[0]} out of {len(test_set)-1}")
        pmid = int(article[1]["pmid"])
        title = article[1]["title"]
        abstract = article[1]["abstract"]
        result[pmid] = {"entities": entities}
        nlp_title = trained_model(title)
        nlp_abstract = trained_model(abstract)
        for ent in nlp_title.ents:
            entities.append({
                "start_idx": ent.start_char,
                "end_idx": ent.end_char,
                "location": "title",
                "text_span": ent.text,
                "label": ent.label_
                })
        for ent in nlp_abstract.ents:
            entities.append({
                "start_idx": ent.start_char,
                "end_idx": ent.end_char,
                "location": "abstract",
                "text_span": ent.text,
                "label": ent.label_
                })

    output_path = os.path.join(path, "Output", f"submission_NER_{model}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent = 4)
    return result
    
NER = sub_scr("SciBERT")
