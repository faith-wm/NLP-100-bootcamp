
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch, csv
import logging, re
from torch import nn
import os
import pandas as pd
from transformers import BertTokenizer, BertForPreTraining
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, BertPreTrainedModel
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time, datetime, random
import codecs



newsCorpora='NewsAggregatorDataset/newsCorpora.csv'
pageSessions= 'NewsAggregatorDataset/2pageSessions.csv'

read_newsCorpora=pd.read_csv(newsCorpora, delimiter='\t',quoting=3,
                             names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

publishers = ['Reuters','Huffington Post','Businessweek','Contactmusic.com','Daily Mail']

filtered_Df=read_newsCorpora[read_newsCorpora['PUBLISHER'].isin(publishers)]
filtered_Df.sample(frac=10,replace=True).reset_index()

# filtered_Df=filtered_Df[:2000]


filtered_Df=filtered_Df[['CATEGORY','TITLE']]
train_df,valid_te= train_test_split(filtered_Df,test_size=0.2, shuffle=True, random_state=123,
                              stratify=filtered_Df['CATEGORY'])
valid_df,test_df= train_test_split(valid_te,test_size=0.5, shuffle=True, random_state=123,
                              stratify=valid_te['CATEGORY'])


random.seed(100)



device_name = tf.test.gpu_device_name()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are {} GPU(s) available.'.format(torch.cuda.device_count()))
    print('GPU {} will be used'.format(torch.cuda.get_device_name(0)))
else:
    print('There are no GPUs, we will use CPU instead')
    device = torch.device("cpu")




def preprocessing(data:list, tokenizer):
    max_bert_input_length = 510
    for sentence in data:
        sentence_tokens = tokenizer.tokenize(sentence['text'])
        if len(sentence_tokens)>max_bert_input_length:
            sentence_tokens=sentence_tokens[:max_bert_input_length]
        sentence['tokens'] = sentence_tokens


    data_input_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    data_token_type_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    data_attention_masks = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    data_scores = torch.empty((len(data), 1), dtype=torch.float)

    for index,sentence in enumerate(data):
        tokens=[]
        input_type_ids=[]
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in sentence['tokens']:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_masks = [1] * len(input_ids)
        while len(input_ids) < max_bert_input_length:
            input_ids.append(0)
            attention_masks.append(0)
            input_type_ids.append(0)


        # print('\n\n lens', len(input_ids),len(attention_masks), len(input_type_ids))

        data_input_ids[index] = torch.tensor(input_ids, dtype=torch.long)
        data_token_type_ids[index] = torch.tensor(input_type_ids, dtype=torch.long)
        data_attention_masks[index] = torch.tensor(attention_masks, dtype=torch.long)


        data_scores[index] = torch.tensor(sentence['label'], dtype=torch.long)  # dtype=torch.float

    return data_input_ids, data_token_type_ids, data_attention_masks, data_scores


def loadData(file):
    data = []
    categories = ['b', 't', 'e', 'm']
    idx=-1
    for text,label in zip(file['TITLE'].values.tolist(), file['CATEGORY'].values.tolist()):
        idx+=1
        data.append({'index':idx, 'text':text,'label':categories.index(label)})
    return data


def accuracy(predictions,true_scores,scores_len):
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    flat_true_scores = np.asarray(true_scores)
    flat_true_scores = flat_true_scores.reshape(1,scores_len)
    flat_true_labels = [int(item) for sublist in flat_true_scores for item in sublist]
    flat_true_labels = np.asarray(flat_true_labels)


    accuracy_sc=accuracy_score(flat_true_labels,flat_predictions)


    print('accuracy is: {}'.format(accuracy_sc))
    return accuracy_sc



bert_modell='bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_modell, do_lower_case=False)

train_df=loadData(train_df)
train_input_id, train_token_type_id, train_attmask, train_scores = preprocessing(train_df, tokenizer)
train_data = TensorDataset(train_input_id, train_token_type_id, train_attmask, train_scores)
train_sampler = RandomSampler(train_data)


dev_df=loadData(valid_df)
dev_input_id, dev_token_type_id, dev_attmask, dev_scores = preprocessing(dev_df, tokenizer)
validation_data = TensorDataset(dev_input_id, dev_token_type_id, dev_attmask, dev_scores)
validation_sampler = SequentialSampler(validation_data)


learning_rate= 1e-5
epochs= 10
warmup_steps =2

batch_size=10

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=1)

model = BertForSequenceClassification.from_pretrained(bert_modell, num_labels=4,output_attentions=False,output_hidden_states=False)
model.cuda()

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)

loss_values = []


# results_txt=open('model_results.txt','a')
for epoch_i in range(0, epochs):
    print('======== Epoch {} / {} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_token_type_id = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device, dtype=torch.int64)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=b_token_type_id, attention_mask=b_input_mask, labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)

    print(" Average training loss: {0:.2f}".format(avg_train_loss))
    print("Running Validation --------")

    model.eval()

    predictions, true_scores = [], []
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_token_type_id, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=b_token_type_id,
                                 attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_scores.append(label_ids)

    accuracy(predictions, true_scores, len(dev_input_id))

    # results_txt.write('{}\t{}\t{}\n'.format(epoch_i, avg_train_loss,accuracy_score))

# results_txt.close()


