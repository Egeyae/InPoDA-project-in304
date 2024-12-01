import logging

logging.basicConfig(
    level=logging.DEBUG,
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
    "output_dir": "./data/chunks/",
    "raw_data_file": "./data/raw_sentiment140.csv",
    "raw_compressed_file": "./data/raw_sentiment140.csv.zip",
    "chunk_size": 20
}

ga_config = {
        "population_size": 100,
        "elitism_percentage": 0.2,
        "mutation_rate": 0.05,
        "max_epochs": 100,
        "early_stopping": {"enabled": True, "patience": 50, "min_delta": 1e-4},
        "save_dir": "./models/"
}

create = True
chunks = 1

if __name__ == "__main__":
    logger.info("Loading data pipeline")
    data_pipeline = SentimentAnalysisDataPipeline(config=data_config)
    if create:
        data_pipeline.run(num_chunks=chunks)
    data = data_pipeline.load_clean_chunks(num_chunks=chunks)
    logger.info("Data pipeline loaded")
    logger.info("Loading GA pipeline")
    ga_pipeline = GeneticAlgorithmPipeline(config=ga_config, creature_class=SentimentCreature)
    logger.info("GA pipeline loaded")
    ga_pipeline.train(inputs=data['embeddings'].tolist(), expected_outputs=data['sentiment'].tolist())
