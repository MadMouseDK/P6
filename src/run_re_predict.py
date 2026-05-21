import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Utils.re_bert_predict import predict

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_DOCS = 0  # Set to 0 to run on all documents

predict(
    model_dir=os.path.join(root, "Models", "RE_BERT_test"),
    dev_path=os.path.join(root, "Data", "raw", "Annotations", "Dev", "json_format", "dev.json"),
    out_path=os.path.join(root, "Output", "re_bert_predictions.json"),
    batch_size=32,
    max_docs=MAX_DOCS,
)
