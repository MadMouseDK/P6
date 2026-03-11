import spacy


def load_spacy_model(name):

    nlp = spacy.load(name)

    def predict(text):
        return nlp(text)

    return predict