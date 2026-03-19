from Models.SciBERT import *
from Models.RoBERTa import *
from Utils.preprocessing import preprocessing
from Utils.evaluation_metrics import *
import os
#Pipeline skeleton

def pipeline():
    #preprocessing("train")
    #preprocessing("dev")
    
    models = {
        "SciBERT": {},
        "RoBERT": {},
        "BlueBERT": {},
        
    }
    
    
    metrics = {}
    for i in models:
        print(f"Running model: {i}")
        
        
        scores = [5]
        
        metrics[i] = scores
    
    return metrics


if __name__ == "__main__":
    pipeline()
    