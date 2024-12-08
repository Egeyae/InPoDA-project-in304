# Import necessary libraries
import logging
import os

import pandas as pd
import json
from sentiment_analysis.gatrainer.GeneticAlgorithmPipeline import GeneticAlgorithmPipeline
from sentiment_analysis.SentimentAnalysisDataPipeline import SentimentAnalysisDataPipeline
from sentiment_analysis.SentimentCreature import SentimentCreature


class Config:
    def __init__(self, cfg_pth: str = './config.json', recurr: bool = False):
        self.logging = None
        self._config_file = cfg_pth
        self._data = {}
        if not recurr:
            self.load_config()

    def load_config(self):
        try:
            with open(self._config_file, 'r') as f:
                self._data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at path: {self._config_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from configuration file: {self._config_file}")

        for key, value in self._data.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config.from_dict(value)
            else:
                self.__dict__[key] = value

    @staticmethod
    def from_dict(data: dict):
        config = Config(recurr=True)
        config._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                config.__dict__[key] = Config.from_dict(value)
            else:
                config.__dict__[key] = value
        return config

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return json.dumps(self._data, indent=2)

    def as_dict(self):
        return self._data


class Logging:
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    def __init__(self, level: str = "info", log_file: str = None, console_logger: bool = True,
                 format_string: str = '%(asctime)s - %(levelname)s - %(message)s'):
        self.logger = logging.getLogger("InPoDAPipeline")
        self.logger.setLevel(self.levels.get(level.lower(), logging.INFO))

        if console_logger:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(format_string))
            self.logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_string))
            self.logger.addHandler(file_handler)

        if not log_file and not console_logger:
            null_handler = logging.NullHandler()
            self.logger.addHandler(null_handler)

    def get_logger(self):
        return self.logger


class InPoDAPipeline:
    def __init__(self, config_path: str = "./config.json"):
        self.model = None
        self.config: Config = None
        self.logger: logging.Logger = None
        self.data: pd.DataFrame = None

        self.load_config(config_path)
        self.set_logger()

        self.ga_pipeline = GeneticAlgorithmPipeline(
            SentimentCreature,
            config=self.config.training.ga_config.as_dict()
        )
        self.data_pipeline = SentimentAnalysisDataPipeline(
            config=self.config.training.data_config.as_dict()
        )

    def load_config(self, config_path: str):
        self.config = Config(config_path)

    def set_logger(self):
        level = getattr(self.config.logging, 'level', 'info')
        console_logger = getattr(self.config.logging, 'streamhandler', True)
        log_file = getattr(self.config.logging, 'filehandler_file', None) if getattr(self.config.logging, 'filehandler', None) else None
        format_string = getattr(self.config.logging, 'format', '%(asctime)s - %(levelname)s - %(message)s')

        logging_instance = Logging(level=level, log_file=log_file, console_logger=console_logger, format_string=format_string)
        self.logger = logging_instance.get_logger()

    def load_training_data(self):
        self.logger.info("Loading training data...")
        self.data: pd.DataFrame = self.data_pipeline.load_clean_chunks()
        self.logger.info(f"Loaded training data with {len(self.data)} tweets.")

    def train_genetic_algorithm(self):
        self.logger.info("Training genetic algorithm...")
        embeddings = self.data['embeddings'].tolist()
        sentiments = self.data['sentiment'].tolist()
        self.ga_pipeline.train(inputs=embeddings, expected_outputs=sentiments)

    def save_best_creature(self):
        self.logger.info("Saving the best-performing creature...")
        self.ga_pipeline.save_best_model()

    def load_creature(self):
        self.logger.info("Loading a pre-trained creature...")
        self.ga_pipeline.load_best_model()

    def process_input(self, input_data):
        if not self.ga_pipeline.best_creature:
            self.logger.error("Best creature not loaded. Can't process input")
            return
        self.logger.info("Processing input data...")
        input_data = self.data_pipeline.preprocess_sentence(input_data)
        input_ = self.data_pipeline.compute_embeddings([SentimentAnalysisDataPipeline.preprocess_sentence(input_data)])[0]

        self.ga_pipeline.best_creature.process(input_)
        print(self.ga_pipeline.best_creature.get_output())
        return self.data_pipeline.array_to_sentiment(self.ga_pipeline.best_creature.get_output())

    def load_tweets(self):
        self.logger.info("Loading tweets...")
        # Placeholder for tweet loading logic

    def process_tweets_to_dataframe(self, tweets):
        self.logger.info("Processing tweets into a DataFrame...")
        # Placeholder for tweet processing logic

    def top_k_hashtags(self):
        self.logger.info("Extracting top K hashtags...")
        # Placeholder for hashtag extraction

    def top_k_authors(self):
        self.logger.info("Extracting top K authors...")
        # Placeholder for author extraction