from Utils.preprocessing import main as preprocessing
from Utils.evaluation_metrics import *
import os
import sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['PYTHONHASHSEED'] = '1337'
import spacy
from spacy.tokens import DocBin
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
#FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.

#Fixing RNG for reproducibility
seed = 1337
spacy.prefer_gpu(False) # type: ignore
np.random.seed(seed)
random.seed(seed)
try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
except ImportError:
    pass

#Pipeline skeleton
def get_path():
    path = os.getcwd()
    if path.endswith("src"):
        path = os.path.dirname(path)
    return path

def get_data():
    preprocessing("train")
    preprocessing("dev")
    return 


def ner_pipeline():
    
    path = get_path()
    
    models = {
        "SciBERT": os.path.join(path, "Models", "SciBERT_NER", "output", "model-best"),
        "SciBERT-cased": os.path.join(path, "Models", "SciBERT-cased_NER", "output", "model-best"),
        "RoBERTa": os.path.join(path, "Models", "RoBERTa_NER", "output", "model-best"),
        "RoBERTa-biomed": os.path.join(path, "Models", "RoBERTa-biomed_NER", "output", "model-best"),
        "BlueBERT-pubmed": os.path.join(path, "Models", "BlueBERT-pubmed_NER", "output", "model-best"),
        "BLueBERT-mimic": os.path.join(path, "Models", "BlueBERT-mimic_NER", "output", "model-best"),
        "BioBERT-base": os.path.join(path, "Models", "BioBERT-base_NER", "output", "model-best")
    }
    
    
    print("Exact match evaluation metrics:")
    for name, info in models.items():
        print(f"Running model: {name}")
        nlp = spacy.load(info) #Model prediction
        doc_bin = DocBin().from_disk(os.path.join(path, "Data", "processed","dev_data_NER.spacy"))#Ground truth
        docs = list(doc_bin.get_docs(nlp.vocab))
        results = calculate_ner_metrics(nlp, docs)
        
        print(f"Metrics for model {name}")
        print(f"     | {'Precision':>9} | {'Recall':>6} | {'F1':>6} |")
        print(f"Macro| {results['macro'][0]:>9.4} | {results['macro'][1]:>6.4} | {results['macro'][2]:>6.4} |")
        print(f"Micro| {results['micro'][0]:>9.4} | {results['micro'][1]:>6.4} | {results['micro'][2]:>6.4} |")
        #print(f"Class F1:{results['class_f1']}") #Can be changed to class_precision, class_recall or class_f1

    return 

def nerd_pipeline():

    path = get_path()

    dev_path = os.path.join(path, "Data", "raw", "Annotations", "Dev", "json_format", "dev.json")

    models = {
        "allenaibiomed_roberta_base": os.path.join(path, "Data", "processed", "allenaibiomed_roberta_base epoch 8 dense.json"),
    }

    print("Mention-level Relation Extraction")
    for name, pred_path in models.items():
        results = calculate_nerd_metrics(dev_path, pred_path) #Two json files
  
        print(f"Metrics for model {name}")
        print(f"     | {'Precision':>9} | {'Recall':>6} | {'F1':>6} |")
        print(f"Macro| {results[0]:>9.4} | {results[1]:>6.4} | {results[2]:>6.4} |")
        print(f"Micro| {results[3]:>9.4} | {results[4]:>6.4} | {results[5]:>6.4} |")
        #print(f"Class F1:{results['class_f1']}") #Can be changed to class_precision, class_recall or class_f1

    return


if __name__ == "__main__":
    #get_data()
    #ner_pipeline()
    nerd_pipeline()
    