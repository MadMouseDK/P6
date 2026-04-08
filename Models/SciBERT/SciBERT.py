import pandas as pd
import spacy
import scispacy
from spacy import displacy
from spacy.tokens import DocBin
from tqdm import tqdm
import pickle
import os
from spacy.util import filter_spans
import subprocess
import sys


'''

EQUAL WEIGHTS, NOTHING DONE ABOUT ASSIGNING DIFFERENT WEIGHTS TO GOLDEN, SILVER, BRONZE DATASETS.

Look into example.weight in spaCy, I didn't manage to do that.

Otherwise do resampling with replacement, if you wanna run with diffferent weight assigned to each dataset.
'''


def load_pickle(datatype: str) -> pd.DataFrame:
    allowed_values = ["train", "dev"]
    if datatype not in allowed_values:
        raise ValueError(f"{datatype} value not allowed. Allowed values: {allowed_values}")
    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if cwd.endswith(folder):
        cwd = os.path.dirname(cwd)
        cwd = os.path.dirname(cwd)
        path = os.path.join(cwd, "Data", f"{datatype}", f"{datatype}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

train_data = load_pickle("train")
dev_data = load_pickle("dev")

'''
This version of sciBERT is built-in in scispacy. See: https://allenai.github.io/scispacy/

If you don't have scispacy and sciBERT installed:
    
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz   

'''

# create a blank model
nlp = spacy.blank('en')


def create_spacy_docbin(data):
    
    db = DocBin()
    entities_skipped = 0
    start_end_idx = []
    
    for i, (text, annot) in enumerate(tqdm(data)):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            if start > end:
                print(f"Wrong start: {start}, end: {end}")
                start, end = end, start
            span = doc.char_span(start, end, label = label, alignment_mode = "expand")
            if span is None:
                entities_skipped += 1
                start_end_idx.append((start, end))
            else:
                ents.append(span)
    
        ents = filter_spans(ents)
        try:
            doc.ents = ents
        except ValueError as e:
            print(f"Skipping doc {i}: {e}")
            continue
        
        db.add(doc)
    return db


train_data_doc = create_spacy_docbin(train_data)
dev_data_doc = create_spacy_docbin(dev_data)


# exporting the data
train_data_doc.to_disk("data_train.spacy")
dev_data_doc.to_disk("data_dev.spacy")



def train_spacy_model(config_dir: str):
    # fill config
    subprocess.run(
        [sys.executable, "-m", "spacy", "init", "fill-config", "base_config.cfg", "config.cfg"],
        cwd = config_dir,
        check = True
    )
    
    # train
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "train", "config.cfg", "--output", "./output"],
        cwd = config_dir,
        check = True
    )
    
    #if result.returncode != 0:
        #print(f"Training failed with exit code {result.returncode}")


train_spacy_model(os.getcwd())

# MODEL RESULTS (for fun)

'''
I chose a pubmed article, that is not present in the datasets

pmid =  32526057


articles_title_abstract.loc[articles_title_abstract['pmid'] == 32526057]
articles_dev.loc[articles_dev['pmid'] == 32526057]
'''

model_test = "Amyotrophic lateral sclerosis (ALS) is a neurodegenerative disorder affecting primarily \
    the motor system, but in which extra-motor manifestations are increasingly recognized. The loss of upper \
    and lower motor neurons in the motor cortex, the brain stem nuclei and the anterior horn of the spinal \
    cord gives rise to progressive muscle weakness and wasting. ALS often has a focal onset but \
    subsequently spreads to different body regions, where failure of respiratory muscles typically \
    limits survival to 2-5 years after disease onset. In up to 50% of cases, there are extra-motor \
    manifestations such as changes in behaviour, executive dysfunction and language problems. \
    In 10%-15% of patients, these problems are severe enough to meet the clinical criteria of frontotemporal \
    dementia (FTD). In 10% of ALS patients, the family history suggests an autosomal dominant inheritance pattern. \
    The remaining 90% have no affected family members and are classified as sporadic ALS. The causes of ALS appear \
    to be heterogeneous and are only partially understood. To date, more than 20 genes have been associated with ALS. \
    The most common genetic cause is a hexanucleotide repeat expansion in the C9orf72 gene, responsible for 30%-50% of \
    familial ALS and 7% of sporadic ALS. These expansions are also a frequent cause of frontotemporal dementia, \
    emphasizing the molecular overlap between ALS and FTD. To this day there is no cure or effective treatment for ALS and the \
    cornerstone of treatment remains multidisciplinary care, including nutritional and respiratory support and symptom management. \
    In this review, different aspects of ALS are discussed, including epidemiology, aetiology, pathogenesis, clinical features, differential \
    diagnosis, investigations, treatment and future prospects."


# load the trained model
nlp_output = spacy.load("output/model-best")

# pass our test instance into the trained pipeline
doc = nlp_output(model_test)

# visualize the identified entities
displacy.render(doc, style = "ent")

visualizer = displacy.render(doc, style = "ent")

with open("entities.html", "w") as f:
    f.write(visualizer)

