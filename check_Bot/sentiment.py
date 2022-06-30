from __future__ import annotations

from rasa.engine.graph import GraphComponent, ExecutionContext

from rasa.shared.nlu.training_data.training_data import TrainingData

from rasa.engine.recipes.default_recipe import DefaultV1Recipe

from typing import List, Type, Dict, Text, Any, Optional

from rasa.engine.graph import ExecutionContext

from rasa.engine.storage.resource import Resource

from rasa.engine.storage.storage import ModelStorage

from rasa.shared.nlu.training_data.message import Message

from rasa.shared.nlu.constants import TEXT

from rasa.nlu.extractors.extractor import EntityExtractorMixin

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pickle
import pandas as pd

from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from pythainlp.util import isthai

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)

class SentimentAnalyzer(GraphComponent, EntityExtractorMixin):

    """A pre-trained sentiment component"""

    name = "sentiment"
    provides = ["entities"]
    requires = []
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config: Dict[Text, Any]) -> None:

        self.component_config = component_config

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config)

    def train(self, training_data: TrainingData) -> Resource:

        pass

    def convert_to_rasa(self, value, confidence, lang):
        """Convert model output into the Rasa NLU compatible output format."""

        if lang == 'th':
            extractor = 'TH_sentiment_extractor'
        elif lang == 'en':
            extractor = 'EN_sentiment_extractor'

        entity = {"value": value,
                "confidence": confidence,
                "entity": "sentiment",
                "extractor": extractor}

        return entity

    def process(self, messages: List[Message]) -> List[Message]:
        """Retrieve the text message, pass it to the classifier and append the prediction results
        to the message class."""

        with open('model.pkl' , 'rb') as f:
            model = pickle.load(f)

        with open('vectorizer.pkl' , 'rb') as f:
            vectorizer = pickle.load(f)

        sid = SentimentIntensityAnalyzer()

        for message in messages:
            lang = 'en'
            for i in message.get(TEXT):
                if isthai(i):
                    lang = 'th'
                    break

            if lang == 'th':
                list_word = word_tokenize(message.get(TEXT))

                stopwords = list(thai_stopwords())
                list_word = [i for i in list_word if i not in stopwords]

                list_word = " ".join(token for token in list_word)

                test_tf = vectorizer.transform([list_word])
                res = model.predict(test_tf)
                key, value = res[0], None

            elif lang == 'en':
                res = sid.polarity_scores(message.get(TEXT))
                key, value = max(res.items(), key=lambda x: x[1])

            entity = self.convert_to_rasa(key, value, lang)
            message.set("entities", [entity], add_to_output=True)

        return messages

    def persist(self, file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""
        pass