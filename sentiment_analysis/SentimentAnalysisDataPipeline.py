import zipfile

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentiment_analysis.gatrainer.DataProcessingPipeline import AbstractDataPipeline
import pandas as pd
import os
import gc


def str_embed_to_np_array(array):
    l = eval(array)
    return np.array(l, dtype=np.float16)


class SentimentAnalysisDataPipeline(AbstractDataPipeline):
    """
    A data processing pipeline using Transformer-based embeddings.
    """

    def __init__(self, config: object = None) -> object:
        super().__init__(config)
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModel.from_pretrained(self.config.get("model_name", "bert-base-uncased")).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("model_name", "bert-base-uncased"))
        self.logger.info("Model and tokenizer loaded.")

    @staticmethod
    def preprocess_sentence(sentence: str):
        return " ".join(word for word in sentence.split() if not (word.startswith("@") or word.startswith("http") or word.startswith("#")))

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        self.logger.info("Preprocessing data...")
        data["text"] = data["text"].apply(SentimentAnalysisDataPipeline.preprocess_sentence)
        return data

    def compute_embeddings(self, sentences: list[str]) -> np.ndarray:
        """Compute embeddings for a list of sentences."""
        self.model.eval()
        embeddings = []
        batch_size = self.config.get("batch_size", 32)
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.config.get("max_seq_len", 128)
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings.astype(np.float16))  # Directly append to the list
        return np.array(embeddings, dtype=np.float16)  # Convert list to array at once

    def compute_sentiment_arrays(self, sentiments: list[str]) -> np.ndarray:
        """Vectorized computation of sentiment arrays."""
        sentiment_map = {
            "0": np.array([1, 0], dtype=np.float16),
            0: np.array([1, 0], dtype=np.float16),
            "4": np.array([0, 1], dtype=np.float16),
            4: np.array([0, 1], dtype=np.float16),
        }
        return np.array([sentiment_map[sentiment] for sentiment in sentiments], dtype=np.float16)

    # noinspection PyTypeChecker
    def array_to_sentiment(self, sentiment_array: np.ndarray) -> str:
        arrays_labels = ("positive", "neutral", "negative")
        arrays = (np.linalg.norm(np.array([1, 0]), sentiment_array),
                  np.linalg.norm(np.array([0.5, 0.5]), sentiment_array),
                  np.linalg.norm(np.array([0, 1]), sentiment_array)
                  )
        return arrays_labels[arrays.index(min(arrays))]

    def process_chunk(self, chunk: pd.DataFrame, idx: int) -> None:
        """Process a single chunk of data."""
        try:
            self.logger.info(f"Processing chunk {idx} with {len(chunk)} rows...")
            chunk = chunk.drop(columns=["id", "date", "query", "user"],
                               errors="ignore")
            chunk = self.preprocess(chunk)
            texts = chunk["text"].tolist()
            embeddings = self.compute_embeddings(texts)
            chunk["embeddings"] = embeddings.tolist()
            sentiments = self.compute_sentiment_arrays(chunk["sentiment"].tolist())
            chunk["sentiment"] = sentiments.tolist()

            output_path = os.path.join(self.config["output_dir"], f"chunk{idx}.csv")
            chunk.to_csv(output_path, index=False, mode='w')
            self.logger.info(f"Chunk {idx} saved to {output_path}.")
        except Exception as e:
            self.logger.error(f"Error processing chunk {idx}: {e}", exc_info=True)
            raise
        finally:
            del chunk
            gc.collect()  # Explicit garbage collection

    def load_clean_chunks(self, num_chunks: int = -1):
        df = self.load_chunks(num_chunks)
        df["embeddings"] = df["embeddings"].apply(lambda x: str_embed_to_np_array(x))
        df["sentiment"] = df["sentiment"].apply(lambda x: str_embed_to_np_array(x))
        return df

    def run(self, num_chunks: int = -1):
        """Run the full pipeline."""
        raw_file = self.config.get("raw_data_file", "../data/raw_data.csv")
        if not os.path.isfile(raw_file):
            if not os.path.isfile(raw_file+".zip"):
                raise FileNotFoundError(f"{raw_file} nor {raw_file}.zip were found. Cannot proceed")
            else:
                with zipfile.ZipFile(raw_file+".zip", 'r') as zip_ref:
                    zip_ref.extractall("/".join(raw_file.split("/")[:-1]))

        chunk_size = self.config.get("chunk_size", 1000)
        self.process_data(raw_file, chunk_size, num_chunks=num_chunks)

