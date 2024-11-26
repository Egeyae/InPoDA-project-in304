import gc
import os
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from multiprocessing import Pool
from sentiment_analysis.env import (MODEL_NAME, DEVICE, RAW_DATA_FILE, OUTPUT_DIR, CHUNK_SIZE,
                                    NCHUNKS, NUM_WORKERS, BATCH_SIZE, MAX_SEQ_LEN, CREATE, RAW_COMPRESSED_DATA_FILE,
                                    RAW_FILE_NAME, RAW_FILE_DIR)
from sentiment_analysis.env import NUM_CHUNKS_LOAD
import zipfile

# Logger setup
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Device configuration
device = torch.device(DEVICE)
logger.info(f"Using device: {device}")

# Load model and tokenizer
logger.info(f"Loading model and tokenizer ({MODEL_NAME})...")
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
logger.info("Model and tokenizer loaded successfully.")

global DATA
DATA = None


def ensure_directory_exists(directory: str) -> None:
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def remove_mentions(sentence: str) -> str:
    """Remove '@mentions' from a given sentence."""
    return " ".join(word for word in sentence.split() if not word.startswith("@"))


def compute_embedding(sentences: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Compute embeddings for sentences in smaller batches."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                               padding=True, max_length=MAX_SEQ_LEN)
            inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU/CPU
            outputs = model(**inputs)
            cls_batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)  # Move to CPU
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
        chunk_df["embeddings"] = [embedding.tolist() for embedding in embeddings]

        # Save processed chunk
        output_path = os.path.join(OUTPUT_DIR, f"chunk{idx}.csv")
        chunk_df.to_csv(output_path, index=False)
        logger.info(f"Chunk {idx} saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error processing chunk {idx}: {e}")
        raise
    finally:
        # Cleanup to release memory
        del chunk_df, embeddings
        gc.collect()


def load_and_split_data(file_path: str, chunk_size: int = CHUNK_SIZE) -> None:
    """Load raw data and process it in manageable chunks."""
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found.")
        raise FileNotFoundError(f"{file_path} does not exist.")

    logger.info(f"Loading data from {file_path}...")
    # Ensure the output directory exists
    ensure_directory_exists(OUTPUT_DIR)

    # Read data in chunks directly using pandas
    with Pool(NUM_WORKERS) as p:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = len(f.read().splitlines()) - 1
            last_chunk_idx = total_lines // chunk_size

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            chunk_iterator = pd.read_csv(file, chunksize=chunk_size)

            for idx, chunk_df in enumerate(chunk_iterator):
                # Process only the first NCHUNKS and the last NCHUNKS
                if idx < NCHUNKS or idx > last_chunk_idx - NCHUNKS:
                    p.apply_async(process_chunk, args=(chunk_df, idx))

        p.close()
        p.join()

    logger.info("Data processing complete.")


def clear_chunks(directory: str = OUTPUT_DIR) -> None:
    """Remove all chunk files in the specified directory."""
    logger.info(f"Clearing existing chunks in {directory}...")
    ensure_directory_exists(directory)
    for file in os.listdir(directory):
        if file.startswith("chunk"):
            os.remove(os.path.join(directory, file))
    logger.info("All chunks cleared.")


def exec_creation(raw_file: str = RAW_DATA_FILE) -> None:
    """Main function to clear old data and create new processed chunks."""
    logger.info("Starting data creation process...")
    clear_chunks()
    load_and_split_data(raw_file)
    logger.info("Data creation process completed.")


def _load_chunk(file_path: str) -> pd.DataFrame:
    """Helper function to load a single chunk."""
    logger.info(f"Loading chunk: {file_path}")
    return pd.read_csv(file_path)


def load_data(num_chunks: int = NUM_CHUNKS_LOAD) -> pd.DataFrame:
    """Load data chunks concurrently using multiprocessing for faster execution."""
    logger.info(f"Loading data from {OUTPUT_DIR} with {num_chunks} chunks.")

    dirlist = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.startswith("chunk")],
        key=lambda x: int(x.split(".")[0].split("k")[1])
    )
    dirlist = dirlist[:num_chunks] + dirlist[-num_chunks:]
    logger.info(f"{len(dirlist)} chunks will be loaded.")

    file_paths = [os.path.join(OUTPUT_DIR, file) for file in dirlist]

    with Pool(processes=4) as pool:
        dataframes = pool.map(_load_chunk, file_paths)

    logger.info("All chunks loaded successfully.")
    return pd.concat(dataframes, ignore_index=True)


def extract_source_zip():
    if os.path.exists(RAW_DATA_FILE):
        return
    else:
        if os.path.exists(RAW_COMPRESSED_DATA_FILE):
            with zipfile.ZipFile(RAW_COMPRESSED_DATA_FILE, "r") as zip_ref:
                zip_ref.extract(RAW_FILE_NAME, RAW_FILE_DIR)
        else:
            raise FileNotFoundError(f"{RAW_COMPRESSED_DATA_FILE} not found.")


def main():
    global DATA
    if CREATE:
        try:
            extract_source_zip()
            exec_creation()
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    DATA = load_data()


if __name__ == "__main__":
    main()
