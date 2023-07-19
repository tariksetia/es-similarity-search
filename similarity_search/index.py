from dataclasses import dataclass
from elasticsearch import Elasticsearch
from similarity_search.model import EmbeddingModel


@dataclass
class Index:
    name: str
    es_client: Elasticsearch
    k: int
    model: EmbeddingModel | None = None

    def full_text_search(
        self,
        field: str,
        query: str,
        fields: list[str] | None = None,
        k: int | None = None,
    ):
        if k:
            self.k = k

        resp = self.es_client.search(
            index=self.name,
            query={"match": {field: query}},
            size=self.k,
            from_=0,
            _source=fields,
        )

        hits = resp["hits"]["hits"]
        hits = [{"score": hit["_score"], "source": hit["_source"]} for hit in hits]
        return hits

    def knn_search(
        self, embedding_field: str, query: str, fields: list[str], k: int | None = None
    ):
        if k:
            self.k = k

        query_vector = self.model.predict(query)

        query = {
            "field": embedding_field,
            "query_vector": query_vector,
            "k": self.k,
            "num_candidates": self.k,
        }
        resp = self.es_client.knn_search(index=self.name, knn=query, source=fields)
        hits = resp["hits"]["hits"]
        hits = [{"score": hit["_score"], "source": hit["_source"]} for hit in hits]
        return hits

    @property
    def fields(self):
        resp = self.es_client.indices.get_mapping(index=self.name)
        properties = resp["collections"]["mappings"]["properties"]
        return list(properties.keys())
