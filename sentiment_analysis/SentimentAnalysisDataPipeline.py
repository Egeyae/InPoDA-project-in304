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
        with torch.no_grad():
            for i in range(0, len(sentences), self.config.get("batch_size", 32)):
                batch = sentences[i:i + self.config.get("batch_size", 32)]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.config.get("max_seq_len", 128)
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)
                embeddings.append(cls_embeddings)
        return np.vstack(embeddings)


    def compute_sentiment_arrays(self, sentiments: list[str]) -> np.ndarray:
        array = []
        for sentiment in sentiments:
            if sentiment == "0" or sentiment == 0: # negative
                array.append(np.array([1, 0]))
            else:
                array.append(np.array([0, 1]))
        return np.array(array)

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
            chunk = self.preprocess(chunk)
            texts = chunk["text"].tolist()
            embeddings = self.compute_embeddings(texts)
            chunk["embeddings"] = [embedding.tolist() for embedding in embeddings]
            sentiments = self.compute_sentiment_arrays(chunk["sentiment"].tolist())
            chunk["sentiment"] = [sentiment.tolist() for sentiment in sentiments]

            chunk.drop(columns=["id", "date", "query", "user"])
            output_path = os.path.join(self.config["output_dir"], f"chunk{idx}.csv")
            chunk.to_csv(output_path, index=False)
            self.logger.info(f"Chunk {idx} saved to {output_path}.")
        except Exception as e:
            self.logger.error(f"Error processing chunk {idx}: {e}")
            raise
        finally:
            del chunk
            gc.collect()

    def load_clean_chunks(self, num_chunks: int = -1):
        df = self.load_chunks(num_chunks)
        df["embeddings"] = df["embeddings"].apply(lambda x: str_embed_to_np_array(x))
        df["sentiment"] = df["sentiment"].apply(lambda x: str_embed_to_np_array(x))
        return df

    def run(self, num_chunks: int = -1):
        """Run the full pipeline."""
        raw_file = self.config.get("raw_data_file", "../data/raw_data.csv")
        chunk_size = self.config.get("chunk_size", 1000)
        self.process_data(raw_file, chunk_size, num_chunks=num_chunks)

