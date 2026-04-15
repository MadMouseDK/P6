from src.Utils.preprocessing import main as preprocessing
from src.Utils.evaluation_metrics import *
import os
import spacy
from spacy.tokens import DocBin

#Pipeline skeleton

def pipeline():
    #preprocessing("train") #Not working, require spacy file to work
    #preprocessing("dev")

    
    
    models = {
        "SciBERT": "Models/SciBERT_NER/output/model-best",
        "RoBERTa": "Models/RoBERTa_NER/output/model-best",
        "BlueBERT_pubmed": "Models/BlueBERT-pubmed-L12_NER/output/model-best",
        "BLueBERT_mimic": "Models/BlueBERT-pubmed-mimic-L12_NER/output/model-best",
        "BioBERT-base": "Models/BioBERT-base_NER/output/model-best"
 
    }
    

   
    for name, info in models.items():
        print(f"Running model: {name}")
        nlp = spacy.load(info)
        doc_bin = DocBin().from_disk(os.path.join("Data", "dev","dev_data_NER.spacy"))
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        true, pred = metrics(nlp, docs)
        results = calculate_metrics(true, pred, iou_threshold=0.5)

        print(f"PARTIAL MATCH METRICS for model {name}")
        print(f"     |  Precision |     Recall   |   F1                 ")
        print(f"Macro| {results['macro']}")
        print(f"Micro| {results['micro']}")

    return 



if __name__ == "__main__":
    pipeline()
    