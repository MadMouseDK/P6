from Utils.preprocessing import main as preprocessing
from Utils.evaluation_metrics import *
from Models.test_model import SpacyNERModel
import os


#Dataset -> Model -> Prediction -> Metrics -> Results
#Data
train_data = "Data/Train/train.spacy"
dev_data = "Data/Dev/dev.spacy"

#Models
models = []

#
