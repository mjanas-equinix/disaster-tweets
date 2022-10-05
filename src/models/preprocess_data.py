import numpy as np
import pandas as pd
import re
import spacy
import emoji


class TextCleanTransformer:
    def __init__(self,
                 nlp_model=None):
        self.nlp_model = spacy.load("en_core_web_lg") if not nlp_model else nlp_model

    @staticmethod
    def preprocess_text(text):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'RT : ', ' ', text)
        text = re.sub(emoji.get_emoji_regexp(), ' ', text)
        return text

    def preprocess_sentence(self, text):
        text = self.preprocess_text(text)
        doc = self.nlp_model(text)
        output_text = []
        for token in doc:
            if not any([token.is_stop, token.like_url, token.is_punct,
                        not token.is_ascii, token.like_email, token.is_space]):
                output_text.append(token.lemma_)
        return ' '.join(output_text)

