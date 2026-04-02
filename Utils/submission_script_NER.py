import pandas as pd
import spacy
import os
import json


'''
so far works only for spacy models
'''

def sub_scr(model: str):
    path = os.getcwd()
    if path.endswith("Utils"):
        path = os.path.dirname(path)
    # I just loaded articles from golden set for now 
    data_path = os.path.join(path, "Data", "raw", "GutBrainIE_Full_Collection_2026", "Articles", "csv_format", "articles_train_gold.csv")
    test_set = pd.read_csv(data_path, sep = '|')
    test_set = test_set[["pmid","title", "abstract"]]
    test_set = test_set[:10] # choosing 10 just for now
    allowed_values = ["RoBERTa", "SciBERT"]
    if model not in allowed_values:
        raise ValueError(f"{model} value not allowed. Allowed values: {allowed_values}")
    model_path = os.path.join(path, "Models", model, "output", "model-best")
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

    with open("submission_NER.json", "w") as f:
        json.dump(result, f, indent = 4)
    return result
    
NER = sub_scr("RoBERTa")
