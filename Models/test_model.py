import spacy

class SpacyNERModel:

    def __init__(self):
        self.model = spacy.load("en_core_web_sm")

    def predict(self, text):
        doc = self.model(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        return entities
