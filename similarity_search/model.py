from dataclasses import dataclass
from typing import Any
import elasticsearch
from shell import shell
from elasticsearch import Elasticsearch

import numpy as np


@dataclass
class EmbeddingModel:
    url: str
    hub_model_id: str
    task_type: str
    es_client: Elasticsearch
    es_model_id: str | None = None

    def deploy(self):
        command = (
            f"eland_import_hub_model "
            f"--url {self.url} "
            f"--hub-model-id {self.hub_model_id} "
            f"--task-type {self.task_type} "
            f"--es-model-id {self.es_model_id} "
            f"--start"
        )
        print(command)
        shell(command)
        self.start()

    def predict(self, query: str, as_nparray: bool = False) -> np.ndarray | list[float]:
        document = {"text_field": f"{query}"}

        inference = self.es_client.ml.infer_trained_model(
            model_id=self.es_model_id, docs=document
        )
        result = inference["inference_results"][0]
        if as_nparray:
            return np.array(result["predicted_value"], dtype=np.float32)

        return result["predicted_value"]

    @property
    def exists(self):
        try:
            result = self.es_client.ml.get_trained_models(model_id=self.es_model_id)
        except elasticsearch.NotFoundError as e:
            return False
        return True

    def start(self, timeout=60):
        self.es_client.options(request_timeout=60).ml.start_trained_model_deployment(
            model_id=self.es_model_id, timeout=timeout, wait_for="started"
        )
