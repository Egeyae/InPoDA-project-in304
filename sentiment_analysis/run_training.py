import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] ::%(name)s:: (%(levelname)s) - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ga_training.log')
    ]
)
logger = logging.getLogger("RUN")

logger.info("Start of training")

from SentimentAnalysisDataPipeline import SentimentAnalysisDataPipeline
from gatrainer.GeneticAlgorithmPipeline import GeneticAlgorithmPipeline
from SentimentCreature import SentimentCreature

data_config = {
    "model_name": "distilbert-base-multilingual-cased",
    "batch_size": 32,
    "max_seq_len": 128,
    "output_dir": "../data/chunks/",
    "raw_data_file": "../data/raw_sentiment140.csv",
    "raw_compressed_file": "../data/raw_sentiment140.csv.zip",
    "chunk_size": 1000
}

ga_config = {
    "population_size": 10,
    "elitism_percentage": 0.1,
    "mutation_rate": 0.08,
    "max_epochs": 100,
    "early_stopping": {"enabled": True, "patience": 20, "min_delta": 0.0001},
    "save_dir": "../models/",
    "model_name": "sentiment.model",
    "training_sample_size": 100
}

create = True
chunks = 5
import time
import psutil
try:
    import GPUtil
    import cupy
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def log_system_usage():
    if HAS_GPU:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(
                f"GPU {gpu.id} | Load: {gpu.load * 100:.1f}% | Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

    mem = psutil.virtual_memory()
    logger.info(f"Memory Usage: {mem.percent}% | Available: {mem.available / 1e6:.2f} MB")
    cpu = psutil.cpu_percent()
    logger.info(f"CPU Usage: {cpu}%")


def monitor_resources(interval=10):
    while True:
        log_system_usage()
        time.sleep(interval)


if __name__ == "__main__":
    # threading.Thread(target=monitor_resources, daemon=True).start()
    logger.info("Loading data pipeline")
    data_pipeline = SentimentAnalysisDataPipeline(config=data_config)
    if create:
        data_pipeline.run(num_chunks=chunks)
        sys.exit(0)
    data = data_pipeline.load_clean_chunks(num_chunks=chunks)
    logger.info("Data pipeline loaded")
    logger.info("Loading GA pipeline")
    ga_pipeline = GeneticAlgorithmPipeline(config=ga_config, creature_class=SentimentCreature)
    logger.info("GA pipeline loaded")
    ga_pipeline.train(inputs=data['embeddings'].tolist(), expected_outputs=data['sentiment'].tolist())

    logger.info("GA pipeline trained")
    logger.info("Evaluating GA")
    ga_pipeline.evaluate(test_inputs=data['embeddings'].tolist(), test_outputs=data['sentiment'].tolist())

    logger.info("Saving best model")
    ga_pipeline.save_best_model()
