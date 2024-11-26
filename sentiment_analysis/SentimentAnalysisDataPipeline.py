import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from gatrainer.DataProcessingPipeline import AbstractDataPipeline
import pandas as pd
import os
import gc


class SentimentAnalysisDataPipeline(AbstractDataPipeline):
    """
    A data processing pipeline using Transformer-based embeddings.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModel.from_pretrained(self.config.get("model_name", "bert-base-uncased")).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("model_name", "bert-base-uncased"))
        self.logger.info("Model and tokenizer loaded.")

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        self.logger.info("Preprocessing data...")
        data["text"] = data["text"].apply(lambda x: " ".join(word for word in x.split() if not word.startswith("@")))
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

    def process_chunk(self, chunk: pd.DataFrame, idx: int) -> None:
        """Process a single chunk of data."""
        try:
            self.logger.info(f"Processing chunk {idx} with {len(chunk)} rows...")
            chunk = self.preprocess(chunk)
            texts = chunk["text"].tolist()
            embeddings = self.compute_embeddings(texts)
            chunk["embeddings"] = [embedding.tolist() for embedding in embeddings]
            output_path = os.path.join(self.config["output_dir"], f"chunk{idx}.csv")
            chunk.to_csv(output_path, index=False)
            self.logger.info(f"Chunk {idx} saved to {output_path}.")
        except Exception as e:
            self.logger.error(f"Error processing chunk {idx}: {e}")
            raise
        finally:
            del chunk
            gc.collect()

    def run(self):
        """Run the full pipeline."""
        raw_file = self.config.get("raw_data_file", "./data/raw_data.csv")
        chunk_size = self.config.get("chunk_size", 1000)
        self.process_data(raw_file, chunk_size)
