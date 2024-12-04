import json
import logging
import os
import zipfile
from abc import ABC, abstractmethod
from multiprocessing import Pool
import pandas as pd


def _ensure_directory_exists(directory: str) -> None:
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


class AbstractDataPipeline(ABC):
    """
    Abstract base class for data processing pipelines.
    """
    config: dict = None
    logger = None
    DEFAULT_CONFIG = {
        "config_dir": "./"
    }

    def __init__(self, config=None):
        """
        Initialize the pipeline with the given configuration.
        :param config: A dictionary of configuration parameters.
        """
        self.set_config(config)
        self.set_logger()
        self.logger.info(f"Initialized {self.__class__.__name__}")
        _ensure_directory_exists(self.config.get("output_dir", "./output/"))

    def set_logger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(logging.NullHandler())

    def set_config(self, config):
        self.config = self.DEFAULT_CONFIG
        for k in config.keys():
            self.config[k] = config[k]

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a single chunk of data.
        Must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def process_chunk(self, chunk: pd.DataFrame, idx: int) -> None:
        """
        Process a single chunk of data.
        Must be implemented by the subclass.
        """
        pass

    def check_source_file(self):
        if os.path.isfile(self.config["raw_data_file"]):
            self.logger.info(f"Source file {os.path.basename(self.config['raw_data_file'])} exists.")
            return
        else:
            self.logger.info(f"Source file {os.path.basename(self.config['raw_data_file'])} does not exist. Attempting to extract it")
            if os.path.exists(self.config["raw_compressed_file"]):
                with zipfile.ZipFile(self.config["raw_compressed_file"], "r") as zip_ref:
                    zip_ref.extract(self.config["raw_data_file"], os.path.dirname(self.config["raw_data_file"]))
            else:
                self.logger.info(f"Couldn't find any valid archive file, searched: {self.config['raw_compressed_file']}")
                raise FileNotFoundError(f"Couldn't find any valid archive file, searched: {self.config['raw_data_file']}")

    # def process_data(self, file_path: str, chunk_size: int, num_chunks: int = -1) -> None:
    #     """
    #     Process data from a file in chunks.
    #     Uses multiprocessing to handle large datasets.
    #     """
    #     self.check_source_file()
    #     self.logger.info(f"Processing data from {file_path}...")
    #     with Pool(self.config.get("num_workers", 4)) as pool:
    #         with pd.read_csv(file_path, chunksize=chunk_size) as reader:
    #             if num_chunks <= 0:
    #                 for idx, chunk in enumerate(reader):
    #                     pool.apply_async(self.process_chunk, args=(chunk, idx))
    #             else:
    #                 size = sum(1 for _ in open(file_path).readlines()) // chunk_size # number of lines over chunk size
    #                 for idx, chunk in enumerate(reader):
    #                     if idx < num_chunks or idx > size-1-num_chunks:
    #                         pool.apply_async(self.process_chunk, args=(chunk, idx))
    #         pool.close()
    #         pool.join()
    #     self.logger.info("Data processing complete.")
    def process_data(self, file_path: str, chunk_size: int, num_chunks: int = -1) -> None:
        """
        Process data from a file in chunks, handling edge cases for `num_chunks`.
        """
        self.check_source_file()
        self.logger.info(f"Processing data from {file_path}...")

        # Calculate total number of chunks in the file
        with open(file_path) as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract 1 for header
        total_chunks = (total_lines + chunk_size - 1) // chunk_size  # Ceiling division

        # Determine indices of chunks to process
        if num_chunks > 0:
            start_indices = list(range(num_chunks))
            end_indices = list(range(max(0, total_chunks - num_chunks), total_chunks))
            chunks_to_process = sorted(set(start_indices + end_indices))
        else:
            chunks_to_process = list(range(total_chunks))

        self.logger.info(f"Chunks to process: {chunks_to_process}")

        with Pool(self.config.get("num_workers", 4)) as pool:
            with pd.read_csv(file_path, chunksize=chunk_size) as reader:
                for idx, chunk in enumerate(reader):
                    if idx in chunks_to_process:
                        pool.apply_async(self.process_chunk, args=(chunk, idx))
            pool.close()
            pool.join()

        self.logger.info("Data processing complete.")

    def load_chunks(self, num_chunks: int = -1) -> pd.DataFrame:
        """
        Load processed chunks into a single DataFrame.
        """
        output_dir = self.config.get("output_dir", "./output/")
        chunk_files = sorted(
            [f for f in os.listdir(output_dir) if f.startswith("chunk")],
            key=lambda x: int(x.split(".")[0].split("k")[1])
        )
        if len(chunk_files) == 0:
            self.logger.error(f"No chunks found in {output_dir}.")
            raise FileNotFoundError(f"No chunks found in {output_dir}.")
        if num_chunks > 0:
            chunk_files = chunk_files[:num_chunks] + chunk_files[-num_chunks:]
        self.logger.info(f"Loading {len(chunk_files)} chunks...")

        dataframes = []
        for file in chunk_files:
            file_path = os.path.join(output_dir, file)
            dataframes.append(pd.read_csv(file_path))

        self.logger.info("Chunks loaded successfully.")

        return pd.concat(dataframes, ignore_index=True)

    def clear_chunks(self) -> None:
        """
        Remove all processed chunks from the output directory.
        """
        output_dir = self.config.get("output_dir", "./output/")
        self.logger.info(f"Clearing chunks in {output_dir}...")
        for file in os.listdir(output_dir):
            if file.startswith("chunk"):
                os.remove(os.path.join(output_dir, file))
        self.logger.info("Chunks cleared.")

    @abstractmethod
    def run(self):
        """
        Run the complete pipeline.
        Must be implemented by the subclass.
        """
        pass
