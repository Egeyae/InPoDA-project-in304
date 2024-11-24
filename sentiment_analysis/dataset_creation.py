import gc
import os
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch


# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model and tokenizer
MODEL = "xlm-roberta-base"
logger.info("Loading model and tokenizer...")
model = AutoModel.from_pretrained(MODEL).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
logger.info("Model and tokenizer loaded successfully.")


def ensure_directory_exists(directory: str) -> None:
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def remove_mentions(sentence: str) -> str:
    """Remove '@mentions' from a given sentence."""
    return " ".join(word for word in sentence.split() if not word.startswith("@"))


def compute_embedding(sentences: list[str], batch_size: int = 32) -> np.ndarray:
    """Compute embeddings for sentences in smaller batches."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU/CPU
            outputs = model(**inputs)
            cls_batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move to CPU
            embeddings.append(cls_batch_embeddings)

    return np.vstack(embeddings)


def process_chunk(chunk_df: pd.DataFrame, idx: int) -> None:
    """Process and save a single chunk."""
    try:
        logger.info(f"Processing chunk {idx} with {len(chunk_df)} rows.")

        # Clean and preprocess data
        chunk_df.drop(columns=["id", "user", "date", "query"], inplace=True)
        chunk_df["text"] = chunk_df["text"].apply(remove_mentions)

        # Compute embeddings in smaller batches
        logger.debug("Computing embeddings...")
        texts = chunk_df["text"].tolist()
        embeddings = compute_embedding(texts)

        # Attach embeddings back to the DataFrame
        chunk_df["embeddings"] = list(embeddings)

        # Save processed chunk
        output_path = f"./data/chunk{idx}.csv"
        chunk_df.to_csv(output_path, index=False)
        logger.info(f"Chunk {idx} saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error processing chunk {idx}: {e}")
        raise
    finally:
        # Cleanup to release memory
        del chunk_df, embeddings
        gc.collect()


def load_and_split_data(file_path: str, chunk_size: int = 10_000) -> None:
    """Load raw data and process it in manageable chunks."""
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found.")
        raise FileNotFoundError(f"{file_path} does not exist.")

    logger.info(f"Loading data from {file_path}...")
    # Ensure the output directory exists
    ensure_directory_exists("./data/")

    # Read data in chunks directly using pandas
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        chunk_iterator = pd.read_csv(file, chunksize=chunk_size)
        for idx, chunk_df in enumerate(chunk_iterator):
            process_chunk(chunk_df, idx)

    logger.info("Data processing complete.")


def clear_chunks(directory: str = "./data/") -> None:
    """Remove all chunk files in the specified directory."""
    logger.info(f"Clearing existing chunks in {directory}...")
    ensure_directory_exists(directory)
    for file in os.listdir(directory):
        if file.startswith("chunk"):
            os.remove(os.path.join(directory, file))
    logger.info("All chunks cleared.")


def exec_creation(raw_file: str) -> None:
    """Main function to clear old data and create new processed chunks."""
    logger.info("Starting data creation process...")
    clear_chunks()
    load_and_split_data(raw_file)
    logger.info("Data creation process completed.")


if __name__ == "__main__":
    RAW_DATA_FILE = "./data/raw_sentiment140.csv"
    try:
        exec_creation(RAW_DATA_FILE)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
