import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
import time
import re
import json
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from transformers import BertTokenizer, AutoTokenizer, AutoModel
import torch
from torch.cuda.amp import autocast
from models.search.preranking import PreRankingModel
from models.search.ranking import load_data, RankingModel

project_dir = '/root/Project/'
# searching module
search_model_path = os.path.join(project_dir, 'models/search')
search_backbone = 'mc-bert'
backbone_dir = os.path.join(search_model_path, search_backbone)
index_name = 'med_qa_data'
search_thresold = 40
# generate module
generate_model_path = os.path.join(project_dir, 'models/generate')

def pre_ranking(query, encoder, num_candidates=128):
    # get sentence embedding
    tokenizer = BertTokenizer.from_pretrained(backbone_dir, local_files_only = True)
    encoded_input = tokenizer([query], max_length=100, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].cuda()
    token_type_ids = encoded_input['token_type_ids'].cuda()
    attention_mask = encoded_input['attention_mask'].cuda()
    with autocast():
        query_vector = encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
    query_vector = query_vector.flatten().detach().cpu().numpy().tolist()
    torch.cuda.empty_cache()
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
    responses = es_client.search(
        index=index_name,
        body={
            "size": num_candidates,
            "query": es_query,
            "_source": {"includes": ["question", "answer"]}
            # "_source": {"includes": ["question", "answer", "text_vector"]}
        },
        explain=True
    )
    
    # response = response['hits']['hits'][0]['_source']['answer']
    top_score = responses['hits']['hits'][0]['_score']
    prerank_results = [[query, response['_source']['question'], response['_source']['answer'], response['_explanation']['details'][0]['value']] for response in responses['hits']['hits']]
    prerank_results = pd.DataFrame(prerank_results, columns=['text1', 'text2', 'answer', 'tf_idf_score'])
    # print(prerank_results.head())
    return prerank_results, top_score

def ranking(prerank_results, ranking_model):
    dataloader = load_data(prerank_results, batch_size=64, gold=False)
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            batch_data = [f.cuda() for f in batch]
            with autocast():
                batch_pred = ranking_model(*batch_data)
            batch_pred = batch_pred.flatten().detach().cpu().numpy().tolist()
            y_pred.extend(batch_pred)
    torch.cuda.empty_cache()
    y_pred = np.array(y_pred)
    prerank_results['ranking_score'] = y_pred * prerank_results['tf_idf_score']
    rank_results = prerank_results.sort_values(by='ranking_score', ascending=False)
    # print('ranking results:')
    # print(rank_results.head())
    return rank_results['answer'].values[0]

def generate_response(query, model):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(generate_model_path, 'chatglm-6b'), trust_remote_code=True)
    response, _ = model.chat(tokenizer, query, history=[])
    return response

if __name__ == '__main__':
    # load preranking model
    print('\033[1;31mloading preranking model...\033[0m')
    pre_ranking_model = PreRankingModel(backbone_dir=backbone_dir).cuda()
    pre_ranking_model.load_state_dict(torch.load(os.path.join(search_model_path, 'preranking_fn.bin')))
    pre_ranking_model.eval()
    query_encoder = pre_ranking_model.encoder1
    # load ranking model
    print('\033[1;31mloading ranking model...\033[0m')
    ranking_model = RankingModel(backbone_dir=backbone_dir).cuda()
    ranking_model.load_state_dict(torch.load(os.path.join(search_model_path, 'ranking_fn.bin')))
    ranking_model.eval()
    # load generative model
    print('\033[1;31mloading generative model...\033[0m')
    generative_model = AutoModel.from_pretrained(os.path.join(generate_model_path, 'chatglm-6b'), trust_remote_code=True).half().cuda()
    # interact
    print('\033[1;31msystem started successfully\033[0m')
    # initialize statstics for time measurement
    search_time = []
    generate_time = []
    while True:
        query = input('\033[1;31mUser:\033[0m')
        if query == '退出':
            break
        # perform search
        print('小助手正在思考...')
        t0 = time.time()
        prerank_results, top_score = pre_ranking(query, query_encoder)
        if top_score < search_thresold:
            # similar question not found, generate response
            response = generate_response(query, generative_model)
            response = re.sub('ChatGLM-6B', 'MedBot', response)
            print('\033[1;31mBot[generate]:\033[0m', response)
            t1 = time.time()
            generate_time.append(t1 - t0)
        else:
            # similar question found, rerank and print answer in database
            response = ranking(prerank_results, ranking_model)
            print('\033[1;31mBot[search]:\033[0m', response)
            t1 = time.time()
            search_time.append(t1 - t0)
    print('\033[1;31mSummary:\033[0m')
    print(f'search time avg: {np.mean(search_time)}')
    print(f'generate time avg: {np.mean(generate_time)}')