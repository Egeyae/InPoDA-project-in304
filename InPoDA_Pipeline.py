import logging
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
        """Load configuration from the specified JSON file."""
        with open(self._config_file, 'r') as f:
            self._data = json.load(f)

        for key, value in self._data.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to Config objects
                self.__dict__[key] = Config.from_dict(value)
            else:
                self.__dict__[key] = value

    @staticmethod
    def from_dict(data: dict):
        """Create a Config object from a dictionary."""
        config = Config(recurr=True)
        config._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                config.__dict__[key] = Config.from_dict(value)
            else:
                config.__dict__[key] = value
        return config

    def __getitem__(self, key):
        """Allow dictionary-like access."""
        return self._data[key]

    def __repr__(self):
        return json.dumps(self._data, indent=2)


class Logging:
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    def __init__(self, level: str = "info", log_file: str = None, console_logger: bool = True, format_string: str = '%(asctime)s - %(levelname)s - %(message)s'):
        """
        Initialize logging with a specified level and optional file output.
        Args:
        - level (str): The logging level (debug, info, warning, error, critical).
        - log_file (str): Optional file path to log messages to a file.
        """
        self.logger = logging.getLogger("InPoDAPipeline")
        self.logger.setLevel(self.levels.get(level.lower(), logging.INFO))

        # if console logger is True, create console handler
        if console_logger:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(format_string))
            self.logger.addHandler(console_handler)

        # If a log file is specified, add a file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_string))
            self.logger.addHandler(file_handler)

        # If no handler was specified, add a NullHandler to avoid potential errors
        if not log_file and not console_logger:
            null_handler = logging.NullHandler()
            self.logger.addHandler(null_handler)

    def get_logger(self):
        """Return the configured logger."""
        return self.logger


class InPoDAPipeline:
    data: pd.DataFrame # to store the loaded tweets
    config: Config
    logger: logging.Logger

    def __init__(self, config_path: str = "./config.json"):
        self.load_config(config_path)
        self.set_logger()

        self.ga_pipeline = GeneticAlgorithmPipeline(SentimentCreature, config = self.logger.training.ga_config)
        self.data_pipeline = SentimentAnalysisDataPipeline(config = self.logger.training.data_config)

    def load_config(self, config_path: str):
        self.config = Config(config_path)

    def set_logger(self):
        level = self.config.logging.level
        console_logger = self.config.logging.streamhandler
        log_file = self.config.logging.filehandler if self.config.logging.filehandler else None
        format_string = self.config.logging.format

        logging = Logging(level=level, log_file=log_file, console_logger=console_logger, format_string=format_string)
        self.logger = logging.get_logger()

    # sentiment analysis
    # training functions
    def load_training_data(self):
        pass

    def load_genetic_algorithm(self):
        pass

    def train_genetic_algorithm(self):
        pass

    def save_best_creature(self):
        pass

    # model usage functions
    def load_creature(self):
        pass

    def process_input(self):
        pass

    # extraction
    # tweets loading and processing
    def load_tweets(self):
        pass

    def process_tweets_to_dataframe(self):
        pass

    # data extraction
    def topKhashtags(self):
        pass

    def topKauthors(self):
        pass

    def topKmentions(self):
        pass

    def topKtopics(self):
        pass

    def num_tweets_per_user(self):
        pass

    def num_tweets_per_hastag(self):
        pass

    def num_tweets_per_topic(self):
        pass

    def tweets_of_user(self):
        pass

    def tweets_mentioning_user(self):
        pass

    def users_using_hastag(self):
        pass

    def users_mentioned_by_user(self):
        pass


if __name__ == '__main__':
    pipeline = InPoDAPipeline()
    pipeline.logger.info('Hello World')
