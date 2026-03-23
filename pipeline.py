from Models.SciBERT import *
from Models.RoBERTa import *
from Utils.preprocessing import preprocessing
from Utils.evaluation_metrics import *
import os
from spacy.tokens import DocBin


#Pipeline skeleton

def pipeline():
    preprocessing("train")
    preprocessing("dev")
    
    models = {
        "SciBERT1": "Models/SciBERT/output/model-best",   
        "SciBERT2": "Models/SciBERT/output/model-best"  
    }
    
    
   
    for name, info in models.items():
        print(f"Running model: {name}")
        nlp = spacy.load(info)
        doc_bin = DocBin().from_disk("data_dev.spacy")
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        true, pred = metrics(nlp, docs)
        results = calculate_metrics(true, pred)
        print(f"     |  precision |     recall   |   f1                 ")
        print(f"Macro| {results['macro']}")
        print(f"Micro| {results['micro']}")
    return 



if __name__ == "__main__":
    print(pipeline())
    