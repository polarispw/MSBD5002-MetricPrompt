from pycrawlers import huggingface
import pandas as pd
import os


hg = huggingface()
urls = ['https://huggingface.co/datasets/ag_news/tree/main/data',
        'https://huggingface.co/datasets/fancyzhx/dbpedia_14/tree/main/dbpedia_14',
        '']
paths = ['./data/TextClassification/agnews',
         './data/TextClassification/dbpedia']
hg.get_batch_data(urls, paths)


df = pd.read_parquet("./data/TextClassification/agnews/test-00000-of-00001.parquet")
df.to_csv("./data/TextClassification/agnews/test.csv")
df = pd.read_parquet("./data/TextClassification/agnews/train-00000-of-00001.parquet")
df.to_csv("./data/TextClassification/agnews/train.csv")

df = pd.read_parquet("./data/TextClassification/dbpedia/test-00000-of-00001.parquet")
df.to_csv("./data/TextClassification/dbpedia/test.csv")
df = pd.read_parquet("./data/TextClassification/dbpedia/train-00000-of-00001.parquet")
df.to_csv("./data/TextClassification/dbpedia/train.csv")

# df = pd.read_parquet("./data/TextClassification/agnews/test-00000-of-00001.parquet")
# df.to_csv("./data/TextClassification/agnews/test.csv")
# df = pd.read_parquet("./data/TextClassification/agnews/train-00000-of-00001.parquet")
# df.to_csv("./data/TextClassification/agnews/train.csv")