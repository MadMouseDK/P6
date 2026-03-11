from Utils.preprocessing import *
from Utils.evaluation_metrics import *
from Utils.scorer import evaluate_model
from Models.spacy_models import load_spacy_model
import scispacy



def run_pipeline():

    dev_docs = load_docs("Data/dev/dev.spacy")

    models = {  #Add models here
              
        "spacy_sm": load_spacy_model("en_core_web_md"),
        "spacy_lg": load_spacy_model("en_core_web_sm")
        #"spacy_sci": load_spacy_model("en_core_sci_sm") #ScriPy's model trained on scientific text
    }

    metrics = []

    for name, predict in models.items():
        print(f"Evaluating {name}")
        scores = evaluate(predict, dev_docs)
        metrics.append({
            "model": name,
            **scores
        })

    return metrics


if __name__ == "__main__":
    for i in run_pipeline():
        print(i)