from Utils.preprocessing import main as preprocessing
from Utils.evaluation_metrics import *
import os
import spacy
from spacy.tokens import DocBin

#Pipeline skeleton

def pipeline():
    preprocessing("train")
    preprocessing("dev")
    
    path = os.getcwd()
    if path.endswith("src"):
        path = os.path.dirname(path)

    
    models = {
        "SciBERT": os.path.join(path, "Models", "SciBERT_NER", "output", "model-best"),
        "SciBERT-cased": os.path.join(path, "Models", "SciBERT-cased_NER", "output", "model-best"),
        "RoBERTa": os.path.join(path, "Models", "RoBERTa_NER", "output", "model-best"),
        "BlueBERT-pubmed": os.path.join(path, "Models", "BlueBERT-pubmed_NER", "output", "model-best"),
        "BLueBERT-mimic": os.path.join(path, "Models", "BlueBERT-mimic_NER", "output", "model-best"),
        "BioBERT-base": os.path.join(path, "Models", "BioBERT-base_NER", "output", "model-best")
    }
    

   
    for name, info in models.items():
        print(f"Running model: {name}")
        nlp = spacy.load(info)
        doc_bin = DocBin().from_disk(os.path.join(path, "Data", "processed","dev_data_NER.spacy"))
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        true, pred = metrics(nlp, docs)
        results = calculate_metrics(true, pred, iou_threshold=0.5)

        print(f"PARTIAL MATCH METRICS for model {name}")
        print(f"     | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
        print(f"Macro| {results['macro'][0]:>9.4} | {results['macro'][1]:>6.4} | {results['macro'][2]:>6.4}")
        print(f"Micro| {results['micro'][0]:>9.4} | {results['micro'][1]:>6.4} | {results['micro'][2]:>6.4}")
        #print(f"class| {results['class_f1']}") #Can be changed to class_precision, class_recall or class_f1
  

    return 



if __name__ == "__main__":
    pipeline()
    