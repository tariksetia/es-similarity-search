# ES as VectorDB    <!-- omit in toc -->

![Streamlit app](./assets/streamlit.gif)

## Table of Contents<!-- omit in toc -->

- [1. Text Similarity for MS-MARCO Dataset](#1-text-similarity-for-ms-marco-dataset)
  - [1.1 Preprocess](#11-preprocess)
  - [1.2 Local Setup](#12-local-setup)
  - [1.3 Classes and Modules](#13-classes-and-modules)
- [2. CNN News Dataset(WIP)](#2-cnn-news-datasetwip)
- [3. Todos](#3-todos)

## 1. Text Similarity for MS-MARCO Dataset

### 1.1 Preprocess

- Use [pre-process-collab.ipynb](notebooks/pre-process-collab.ipynb) to calculate the embeddings for passages.
- Make sure to enable GPU on Google Collab.
- The embedding data frame is saved to the drive. It can be later retrieved to build the index in ES
- Store the pickled file from collab to the [data](data/) folder in this project
### 1.2 Local Setup

1. Run: `poetry install`
2. Run: `poetry run python -m pip freeze > requirements.txt`
3. In the project directory, build the docker image: `docker-build -t 3pillar/essm .`
   1. This takes care of hydrating the elastic search with the required data
4. Run: `docker-compose up`
   1. You might see `Hyderator` service keeps failing. Give it some time, it's just waiting for ES node to become healthy.
   2. `hyderator` will exit after indexing the data
   3. You can discover the dataset on [kibana](http://0.0.0.0:5601). The name of the index is `collections`
   4. It will take some to index approximately 180K records
5. Open [Streamlit app.](http://0.0.0.0:8501)

### 1.3 Classes and Modules

- `class<Model>`: Model class is used to add transformer model to elasticsearch. This is achieved by using `eland` package.
- `class<Index>`: Abstraction over ES Index. Include batteries for full text and KNN search.
- [pensieve.py](./similarity_search/pensieve.py): Code for streamlit app.

## 2. CNN News Dataset(WIP)

- A notebook demonstrating e2e process for calculating embeddings on CNN dataset.
- Need improvements like trying another foundation mode, changing chunks size
- Better document expansion

## 3. Todos

1. Try other dataset.
2. Used Document expansion and DPR techniques