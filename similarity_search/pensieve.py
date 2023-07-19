import os
import streamlit as st
from elasticsearch import Elasticsearch

from similarity_search.index import Index
from similarity_search.model import EmbeddingModel
from similarity_search.utils import ESModelNotFound, get_df


def main():
    ES_URL = os.environ.get("es_host")
    es_client = Elasticsearch(ES_URL)

    model = EmbeddingModel(
        url=ES_URL,
        hub_model_id="sentence-transformers/msmarco-MiniLM-L-12-v3",
        task_type="text_embedding",
        es_client=es_client,
        es_model_id="msmarco-MiniLM-L-12-v3".lower().replace("-", "_"),
    )

    if not model.exists:
        raise ESModelNotFound(
            f"No Deployed model found in Elasticsearch for {model.es_model_id} "
        )

    collections_ = Index(name="collections", es_client=es_client, k=100, model=model)
    st.set_page_config(layout="wide")

    with st.container() as search_input:
        st.header("Pensieve of Hogwarts")
        query = st.text_input("Search", "")

        if query:
            with st.container() as search_result:
                lexical, semantic = st.columns(2, gap="small")
                with lexical:
                    results = collections_.full_text_search(
                        field="text", query=query, fields=["id", "text"]
                    )
                    df = get_df(results)
                    st.caption("Full Text Search")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                with semantic:
                    results = collections_.knn_search(
                        embedding_field="text_embeddings",
                        query=query,
                        fields=["id", "text"],
                    )
                    df = get_df(results)
                    st.caption("Semantic Search")
                    st.dataframe(df, use_container_width=True, hide_index=True)
