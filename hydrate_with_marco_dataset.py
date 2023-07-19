import json
import os
import time
import eland as ed
import pandas as pd
import tqdm
from elasticsearch import Elasticsearch
from loguru import logger
from tqdm.auto import tqdm
from pyfiglet import Figlet
from similarity_search.model import EmbeddingModel

tqdm.pandas()

ES_URL = os.environ.get("es_host")
PASSAGE_FILE = "./data/marco-passage-with-embedding-msmarco-MiniLM-L-12-v3.pkl"


def create_collection(es_client):
    logger.info("Creating Collection")
    collections_mappings = {
        "mappings": {
            "properties": {
                "id": {"type": "integer"},
                "text": {
                    "type": "text",
                },
                "text_embeddings": {
                    "index": True,
                    "type": "dense_vector",
                    "dims": 384,
                    "similarity": "cosine",
                },
            }
        }
    }
    logger.info(f"Mapping:\n {json.dumps(collections_mappings, indent=2)}")
    if not es_client.indices.exists(index="collections"):
        es_client.indices.create(index="collections", body=collections_mappings)


def deploy_model_to_elastic_search(es_client):
    logger.info("Deploying Model to ElasticSearch")
    model = EmbeddingModel(
        url=ES_URL,
        hub_model_id="sentence-transformers/msmarco-MiniLM-L-12-v3",
        task_type="text_embedding",
        es_client=es_client,
        es_model_id="msmarco-MiniLM-L-12-v3".lower().replace("-", "_"),
    )
    if not model.exists:
        model.deploy()


def index_row(row, es_client):
    doc = {
        "id": row["id"],
        "text": row["text"],
        "text_embeddings": row["text_embeddings"],
    }
    _ = es_client.index(
        index="collections",
        document=doc,
    )


def main():
    client = Elasticsearch(ES_URL)
    deploy_model_to_elastic_search(client)
    if not client.indices.exists(index="collections"):
        create_collection(client)
        logger.info("Hydrating ElasticSearch")
        data = pd.read_pickle(PASSAGE_FILE)
        data.progress_apply(lambda x: index_row(x, client), axis=1)
    else:
        logger.warning(
            "Skipping Hydration for ES: Index named 'collections' already exists. "
        )
    logger.success("Complete")


if __name__ == "__main__":
    f = Figlet()
    print(f.renderText("Hydrator"))
    time.sleep(5)
    main()
