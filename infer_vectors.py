import os
from tqdm import tqdm
import json
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import TensorDataset, DataLoader
from models.search.preranking import PreRankingModel, encode_text

project_dir = '/root/Project/'
search_model_path = os.path.join(project_dir, 'models/search')
search_backbone = 'mc-bert'
backbone_dir = os.path.join(search_model_path, search_backbone)
index_name = 'med_qa_data'

def load_docs(doc_dir='./data/cMedQA2/qa_pairs.csv', batch_size=512):
    print('loading documents...')
    pa_pairs = pd.read_csv(doc_dir, index_col=0, encoding='utf-8')
    docs = pa_pairs.questions.tolist()
    dataset = TensorDataset(*encode_text(docs))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def infer_vecs(encoder, dataloader):
    print('inferring vectors...')
    all_vecs = []
    for batch in tqdm(list(dataloader)):
        input_ids, token_type_ids, attention_mask = [f.cuda() for f in batch]
        with autocast():
            vecs = encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        vecs = vecs.detach().cpu().numpy().tolist()
        all_vecs.extend(vecs)
    return all_vecs

def dumpVecs(all_vecs, doc_dir='./data/cMedQA2/qa_pairs.csv', target_dir='./data/cMedQA2/doc_with_vecs.txt'):
    f = open(target_dir, 'a', encoding='utf-8')
    f.truncate(0)
    qa_pairs = pd.read_csv(doc_dir, index_col=0, encoding='utf-8').values.tolist()
    assert len(qa_pairs) == len(all_vecs), f'num docs ({len(qa_pairs)}) != num vecs ({len(all_vecs)})'
    for [question, answer], vec in zip(qa_pairs, all_vecs):
        doc_vec = {
            '_op_type': 'index',
            '_index': index_name,
            'question': question,
            'answer': answer,
            'text_vector': vec
        }
        f.write(json.dumps(doc_vec, ensure_ascii=False) + '\n')
    f.close()
    return

if __name__ == '__main__':
    # load model
    print('loading model...')
    pre_ranking_model = PreRankingModel(backbone_dir=backbone_dir).cuda()
    # pre_ranking_model.load_state_dict(torch.load(os.path.join(search_model_path, 'preranking-reg-qqr.bin')))
    # pre_ranking_model.load_state_dict(torch.load(os.path.join(search_model_path, 'preranking-reg-combined.bin')))
    pre_ranking_model.load_state_dict(torch.load(os.path.join(search_model_path, 'preranking_fn.bin')))   
    pre_ranking_model.eval()
    doc_encoder = pre_ranking_model.encoder2
    # load documents
    dataloader = load_docs()
    # infer vectors
    all_vecs = infer_vecs(doc_encoder, dataloader)
    # dump vectors
    dumpVecs(all_vecs)
    print('complete')