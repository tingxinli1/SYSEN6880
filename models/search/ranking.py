import os
import warnings
warnings.filterwarnings("ignore")
import pickle
from itertools import combinations, permutations
from tqdm import tqdm
from callback.progressbar import ProgressBar
import random
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertConfig, BertTokenizer, BertModel
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, classification_report
from torch.nn import BCEWithLogitsLoss, MSELoss
import matplotlib.pyplot as plt

project_dir = '/root/Project/'
search_model_path = os.path.join(project_dir, 'models/search')
search_backbone = 'macbert'
backbone_dir = os.path.join(search_model_path, search_backbone)
data_train_dir = os.path.join(project_dir, 'data/KUAKE-combined/sts_train2.csv')
data_dev_dir = os.path.join(project_dir, 'data/KUAKE-combined/sts_dev2.csv')
# data_train_dir = os.path.join(project_dir, 'data/KUAKE-QQR/sts_train.csv')
# data_dev_dir = os.path.join(project_dir, 'data/KUAKE-QQR/sts_dev.csv')
# data_train_dir = os.path.join(project_dir, 'data/CHIP-STS/CHIP-STS_train.json')
# data_dev_dir = os.path.join(project_dir, 'data/CHIP-STS/CHIP-STS_dev.json')

class RankingModel(nn.Module):
    """Single BERT model definition"""
    def __init__(self, backbone_dir=backbone_dir, objective='reg'):
        super(RankingModel, self).__init__()
        self.objective = objective
        self.encoder_config = AutoConfig.from_pretrained(backbone_dir, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(backbone_dir, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.encoder_config.hidden_size, 1)
    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        sent_emb = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        outputs = self.dropout(sent_emb)
        outputs = self.dense(outputs).ravel()
        if labels != None:
            if self.objective == 'cls':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(outputs, labels)
                outputs = (loss, outputs)
            elif self.objective == 'reg':
                loss_fct = MSELoss()
                loss = loss_fct(outputs, labels)
                outputs = (loss, outputs)
        return outputs

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def encode_text_pair(text1, text2):
    """convert data into tensor"""
    tokenizer = AutoTokenizer.from_pretrained(backbone_dir, local_files_only = True)
    encoded_input = tokenizer(text1, text2, max_length=200, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    token_type_ids = encoded_input['token_type_ids']
    attention_mask = encoded_input['attention_mask']
    return (input_ids, token_type_ids, attention_mask)

def load_data(data, batch_size=32, shuffle=False, gold=True):
    """load data"""
    text1, text2 = data.text1.tolist(), data.text2.tolist()
    dataset = encode_text_pair(text1, text2)
    if gold:
        dataset += (torch.tensor(data.label.tolist(), dtype=torch.float),)
    dataset = TensorDataset(*dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    # fix seed
    seed_everything()
    # read data
    if data_train_dir.endswith('.csv'):
        data_train = pd.read_csv(data_train_dir, encoding='utf-8', index_col=0)
        data_dev = pd.read_csv(data_dev_dir, encoding='utf-8', index_col=0)
    elif data_train_dir.endswith('.json'):
        data_train = pd.read_json(data_train_dir) # , encoding='utf-8', index_col=0
        data_dev = pd.read_json(data_dev_dir) # , encoding='utf-8', index_col=0
    else:
        raise ValueError('unaccepted input file type')
    print(data_train.head())
    print(data_dev.head())
    # load data
    dataloader_train = load_data(data_train, shuffle=True)
    dataloader_dev = load_data(data_dev)
    # initialize model
    model = RankingModel(objective='reg').cuda()
    # initialize optimizer and scheduler 
    learning_rate = 2e-5
    weight_decay = 0.01
    epochs = 10
    patience = 2
    scaler = GradScaler()
    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    num_train_steps = len(dataloader_train) * epochs
    warmup_steps = 100
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    # start training
    best_score = 0
    if model.objective == 'cls':
        criteria = f1_score
    elif model.objective == 'reg':
        criteria = lambda x, y: spearmanr(x, y).correlation
    pbar = ProgressBar(n_total=len(dataloader_train), desc='Training', num_epochs=epochs)
    loss_trend = []
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch+1)
        for step, batch in enumerate(dataloader_train):
            batch_data = [f.cuda() for f in batch]
            optimizer.zero_grad()
            with autocast():
                loss, _ = model(*batch_data)
            train_loss += loss.item() * batch_data[0].shape[0]
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar(step, {'batch_avg_loss': loss.item()})
            loss_trend.append(loss.item())
        train_loss /= len(dataloader_train.dataset)
        print('training loss of epoch %d: %f'%(epoch + 1, train_loss))
        # validate model and save check points
        y_pred = []
        y_true = []
        print("validating...")
        with torch.no_grad():
            model.eval()
            dev_loss = 0
            for batch in dataloader_dev:
                batch_data = [f.cuda() for f in batch]
                with autocast():
                    loss, batch_pred = model(*batch_data)
                dev_loss += loss.item() * batch_data[0].shape[0]
                if model.objective == 'cls':
                    batch_pred = (batch_pred > 0.5)
                batch_true = batch_data[-1]
                batch_pred = batch_pred.flatten().detach().cpu().numpy().tolist()
                batch_true = batch_true.flatten().detach().cpu().numpy().tolist()
                if model.objective == 'cls':
                    y_pred.extend([int(p) for p in batch_pred])
                    y_true.extend([int(p) for p in batch_true])
                else:
                    y_pred.extend(batch_pred)
                    y_true.extend(batch_true)
        cur_score = criteria(y_true, y_pred)
        torch.cuda.empty_cache()
        dev_loss /= len(dataloader_dev.dataset)
        print('dev_loss:', dev_loss)
        if cur_score >= best_score:
            print('validation score increased from %f to %f, save model...'%(best_score, cur_score))
            torch.save(model.state_dict(), '%s_ranking_model.bin' % search_backbone)
            best_score = cur_score
        else:
            print('validation score did not increase; got %f, but the best is %f'%(cur_score, best_score))
            if epoch >= 5:
                patience -= 1
        if patience <= 0:
            print('model overfitted, call early stop')
            break
        # plt.plot(loss_trend)
        # plt.show()