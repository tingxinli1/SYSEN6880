import os
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from models.search.preranking import PreRankingModel
from models.search.ranking import RankingModel

project_dir = '/root/Project/'
search_model_path = os.path.join(project_dir, 'models/search')
search_backbone = 'mc-bert'
backbone_dir = os.path.join(search_model_path, search_backbone)
index_name = 'med_qa_data'

def create_index(index_name):
    print('creating index...')
    es_client = Elasticsearch("http://127.0.0.1:9200")
    es_client.indices.delete(index=index_name, ignore=[404])
    with open('./es_index.json', 'r') as f:
        source = f.read().strip()
        es_client.indices.create(index=index_name, body=source)

def index_docs(doc_dir='./data/cMedQA2/doc_with_vecs.txt'):
    print('indexing documents...')
    es_client = Elasticsearch("http://127.0.0.1:9200")
    with open(doc_dir) as f:
        docs = [json.loads(line.strip()) for line in f.readlines()]
    bulk(es_client, docs)

if __name__ == '__main__':
    create_index(index_name)
    index_docs()