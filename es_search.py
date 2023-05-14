import os
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from transformers import BertTokenizer
import torch
from torch.cuda.amp import autocast
from models.search.preranking import PreRankingModel
from models.search.ranking import RankingModel

project_dir = '/root/Project/'
search_model_path = os.path.join(project_dir, 'models/search')
search_backbone = 'mc-bert'
backbone_dir = os.path.join(search_model_path, search_backbone)
index_name = 'med_qa_data'

def pre_ranking(encoder, query, num_candidates=100):
    # get sentence embedding
    tokenizer = BertTokenizer.from_pretrained(backbone_dir, local_files_only = True)
    encoded_input = tokenizer([query], max_length=100, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].cuda()
    token_type_ids = encoded_input['token_type_ids'].cuda()
    attention_mask = encoded_input['attention_mask'].cuda()
    with autocast():
        query_vector = encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
    query_vector = query_vector.flatten().detach().cpu().numpy().tolist()
    # perform search
    es_client = Elasticsearch("http://127.0.0.1:9200")
    es_query = {
        "function_score":{
            "query": {
                "match": {"question": query}
            },
            "script_score": {
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['text_vector']) + 1",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    response = es_client.search(
        index=index_name,
        body={
            "size": num_candidates,
            "query": es_query,
            "_source": {"includes": ["question", "answer"]}
        }
    )
    # print('用户输入:', query)
    # for i, r in enumerate(response['hits']['hits']):
    #     print(f'回答{i + 1}', r, '\n')
    print(response['hits']['hits'][0])
    response = response['hits']['hits'][0]['_source']['answer']
    return response

if __name__ == '__main__':
    # load model
    print('loading model...')
    pre_ranking_model = PreRankingModel(backbone_dir=backbone_dir).cuda()
    pre_ranking_model.load_state_dict(torch.load(os.path.join(search_model_path, 'preranking-reg-combined.bin')))
    # pre_ranking_model.load_state_dict(torch.load(os.path.join(search_model_path, 'preranking-reg-qqr.bin')))        
    pre_ranking_model.eval()
    query_encoder = pre_ranking_model.encoder1
    # interact
    while True:
        query = input('User:')
        if query == '退出':
            break
        # perform search
        response = pre_ranking(query_encoder, query)
        print('bot:', response)