# Import necessary libraries
import logging

import pandas as pd
from IPython.display import display, HTML
from json2html import json2html
from spacy.lang.ja.syntax_iterators import labels

import Partie_Julien_Konstantinov
from Partie_Julien_Konstantinov import *
from sentiment_analysis.SentimentAnalysisDataPipeline import SentimentAnalysisDataPipeline
from sentiment_analysis.SentimentCreature import SentimentCreature
from sentiment_analysis.gatrainer.GeneticAlgorithmPipeline import GeneticAlgorithmPipeline

import matplotlib.pyplot as plt


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


def pretty_dict_display(dico):
    html_tweets = json2html.convert(json=dico, table_attributes='class="table table-bordered"')
    custom_css = """
    <style>
        table { 
            width: 100%; 
            height: 200px
            border-collapse: collapse; 
            margin-bottom: 20px; 
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px; 
        }
        th { 
            background-color: #f2f2f2; 
            text-align: left; 
        }
        tr:nth-child(even) { 
            background-color: #f9f9f9; 
        }
        td:first-child { 
            width: 20%; /* Set key column to take 20% of the table width */
            font-weight: bold; 
        }
        td:last-child { 
            width: 80%; /* Set value column to take 80% of the table width */
        }
    </style>
    """
    display(HTML(custom_css + html_tweets))


def show_bar_graph(title, xlab, ylab, datax, datay):
    plt.figure()
    plt.bar(datax, datay)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    plt.show()


class InPoDAPipeline:
    def __init__(self, config_path: str = "./config.json"):
        self.tweets_dataframe = None
        self.tweets = None
        self.model = None
        self.config: Config = None
        self.logger: logging.Logger = None
        self.data: pd.DataFrame = pd.DataFrame()

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
        log_file = getattr(self.config.logging, 'filehandler_file', None) if getattr(self.config.logging, 'filehandler',
                                                                                     None) else None
        format_string = getattr(self.config.logging, 'format', '%(asctime)s - %(levelname)s - %(message)s')

        logging_instance = Logging(level=level, log_file=log_file, console_logger=console_logger,
                                   format_string=format_string)
        self.logger = logging_instance.get_logger()

    def compute_chunks(self):
        self.logger.info("Computing chunks...")
        try:
            self.data_pipeline.run(num_chunks=self.config.training.chunks)
        except FileNotFoundError:
            self.logger.warning(
                "Couldn't find a valid file to extract training data, please follow installation instructions, found in `README.md`")
        self.logger.info("Chunks done.")

    def load_training_data(self):
        self.logger.info("Loading training data...")
        try:
            self.data: pd.DataFrame = self.data_pipeline.load_clean_chunks()
        except FileNotFoundError:
            self.logger.info("No chunks were found, please follow the installation instructions, found in `README.md`")
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
        input_ = self.data_pipeline.compute_embeddings([SentimentAnalysisDataPipeline.preprocess_sentence(input_data)])[
            0]

        self.ga_pipeline.best_creature.process(input_)
        print(self.ga_pipeline.best_creature.get_output())
        return self.data_pipeline.array_to_sentiment(self.ga_pipeline.best_creature.get_output())

    def load_tweets(self):
        self.logger.info("Loading tweets...")
        self.tweets = file_open(self.config.tweets.file)
        return self.tweets

    def process_tweets_to_dataframe(self):
        self.logger.info("Processing tweets into a DataFrame...")
        self.tweets_dataframe = tweets_to_df(self.tweets)
        return self.tweets_dataframe

    def get_all_authors(self):
        self.logger.info("Getting all authors...")
        return pd.DataFrame(list(set(self.tweets_dataframe["Auteur"].tolist())), columns=["Authors"])

    def get_all_mentions(self):
        self.logger.info("Getting all mentions...")
        mentions_set = set()
        for mentions in self.tweets_dataframe["Mentions"].tolist():
            if type(mentions) is list:  # means that the list is empty
                continue
            else:
                mentions = eval(mentions)
            for m in mentions:
                mentions_set.add(m)
        return pd.DataFrame(list(mentions_set), columns=["Mentions"])

    def get_all_hashtags(self):
        self.logger.info("Getting all hashtags...")
        hashtags_set = set()
        for hashtag in self.tweets_dataframe["Hashtags"].tolist():
            if type(hashtag) is list:  # means that the list is empty
                continue
            else:
                hashtag = eval(hashtag)
            for h in hashtag:
                hashtags_set.add(h)
        return pd.DataFrame(list(hashtags_set), columns=["Hashtags"])

    def top_k_hashtags(self, k: int = 5):
        self.logger.info("Extracting top K hashtags...")
        hashtags = Partie_Julien_Konstantinov.top_K_hashtags(self.tweets_dataframe, k)
        labels, occurrences = zip(*hashtags)
        show_bar_graph(f"Top K={k} Hashtags", "Hashtags", "Occurrences", labels, occurrences)

    def top_k_authors(self, k: int = 5):
        self.logger.info("Extracting top K authors...")
        authors = Partie_Julien_Konstantinov.top_K_authors(self.tweets_dataframe, k)
        labels, occurrences = zip(*authors)
        show_bar_graph(f"Top K={k} Authors", "Authors", "Occurrences", labels, occurrences)

    def top_k_mentioned(self, k: int = 5):
        self.logger.info("Extracting top K users mentioned...")
        mentioned = Partie_Julien_Konstantinov.top_K_mentions(self.tweets_dataframe, k)
        labels, occurrences = zip(*mentioned)
        show_bar_graph(f"Top K={k} Mentioned", "Mentions", "Occurrences", labels, occurrences)

    def top_k_topics(self, k: int = 5):
        self.logger.info("Extracting top K topics...")
        topics_ = Partie_Julien_Konstantinov.top_K_topics(self.tweets_dataframe, k)
        labels, occurrences = zip(*topics_)
        show_bar_graph(f"Top K={k} Topics", "Topics", "Occurrences", labels, occurrences)

    def count_tweets_user(self):
        self.logger.info("Counting tweets for each user...")
        return Partie_Julien_Konstantinov.nombre_publications_authors(self.tweets_dataframe)

    def count_tweets_hashtag(self):
        self.logger.info("Counting tweets for each hashtag...")

    def count_tweets_mentioned(self):
        self.logger.info("Counting tweets for each mentioned...")

    def all_tweets_from_user(self, user):
        self.logger.info(f"Retrieving tweets from user: {user}...")
        return self.tweets_dataframe[self.tweets_dataframe["Auteur"] == user]

    def all_users_mentioned_by_user(self, user):
        self.logger.info(f"Retrieving mentioned users by user: {user}...")
        all_tweets = self.all_tweets_from_user(user)
        mentions = all_tweets["Mentions"].explode().dropna().unique()
        return pd.DataFrame(mentions, columns=["Users Mentioned"])

    def all_hashtags_used_by_user(self, user):
        self.logger.info(f"Retrieving hashtags used by user: {user}...")
        all_tweets = self.all_tweets_from_user(user)
        hashtags = all_tweets["Hashtags"].explode().dropna().unique()
        return pd.DataFrame(hashtags, columns=["Hashtags Used"])

    def all_users_using_hashtag(self, hashtag):
        self.logger.info(f"Retrieving users using hashtag: {hashtag}...")
        all_tweets = self.tweets_dataframe[self.tweets_dataframe["Hashtags"].apply(lambda x: hashtag in x if isinstance(x, list) else False)]
        users = all_tweets["Auteur"].unique()
        return pd.DataFrame(users, columns=["Users Using Hashtag"])

    def all_tweets_where_user(self, user):
        self.logger.info(f"Retrieving tweets where user is mentioned: {user}...")
        return self.tweets_dataframe[self.tweets_dataframe["Mentions"].apply(lambda x: user in x if isinstance(x, list) else False)]

    def number_of_tweets_per_user(self):
        return Partie_Julien_Konstantinov.nombre_publications_authors(self.tweets_dataframe)

    def number_of_tweets_per_hashtag(self):
        return Partie_Julien_Konstantinov.nombre_publications_hashtags(self.tweets_dataframe)

    def number_of_tweets_per_topic(self):
        return Partie_Julien_Konstantinov.nombre_publications_topics(self.tweets_dataframe)
