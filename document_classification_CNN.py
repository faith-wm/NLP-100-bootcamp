import re
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from nltk import word_tokenize
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# 80 turning words into numeric ids


newsCorpora='NewsAggregatorDataset/newsCorpora.csv'
pageSessions= 'NewsAggregatorDataset/2pageSessions.csv'

read_newsCorpora=pd.read_csv(newsCorpora, delimiter='\t',quoting=3,
                             names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

publishers = ['Reuters','Huffington Post','Businessweek','Contactmusic.com','Daily Mail']

filtered_Df=read_newsCorpora[read_newsCorpora['PUBLISHER'].isin(publishers)]
filtered_Df.sample(frac=10,replace=True).reset_index()





filtered_Df=filtered_Df[['CATEGORY','TITLE']]
train,valid_te= train_test_split(filtered_Df,test_size=0.2, shuffle=True, random_state=123,
                              stratify=filtered_Df['CATEGORY'])
valid,test= train_test_split(valid_te,test_size=0.5, shuffle=True, random_state=123,
                              stratify=valid_te['CATEGORY'])



categories = ['b', 't', 'e', 'm']
category_names = ['business', 'science and technology', 'entertainment', 'health']


def tokenize(x):
    x = re.sub(r'\s+', ' ', x)
    x= word_tokenize(x)
    x = [d for d in x]
    return x

def readDataset(data):
    text= data['TITLE'].values.tolist() #['hello there', 'welcome home']
    labels= data['CATEGORY'].values.tolist() #['b','e','t']

    dataset_labels = [categories.index(l) for l in labels]
    dataset_text=[tokenize(t) for t in text]

    return dataset_text, torch.tensor(dataset_labels, dtype=torch.long)

train_txt, train_label = readDataset(train)
valid_txt, valid_label = readDataset(valid)
test_txt, test_label = readDataset(test)

# print(train_txt[-1]),print(train_label)

from collections import Counter
counter = Counter([x for sent in train_txt for x in sent])  #word,count
train_vocab = [token for token,freq in counter.most_common() if freq>2]

print('all vocab',len(train_vocab))  #go back and remove punctuations
#convert word string into string of ID no.s
vocab_list = ['[UNK]'] + train_vocab
vocab_dict = {x:n for n,x in enumerate(vocab_list)}

print(len(vocab_dict))

def sent_to_ids(sent):
    return torch.tensor([vocab_dict[x if x in vocab_dict else '[UNK]'] for x in sent], dtype=torch.long)

def dataset_to_ids(dataset):
    return [sent_to_ids(x)for x in dataset]

train_ids=dataset_to_ids(train_txt)
valid_ids= dataset_to_ids(valid_txt)
test_ids=dataset_to_ids(test_txt)

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    def __init__(self, X, y):  #Specify the components of dataset
        self.X = X
        self.y = y

    def __len__(self):  # len(dataset)Specify the value to be returned with
        return len(self.y)

    def __getitem__(self, idx):  # dataset[idx]Specify the value to be returned with
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return [self.X[idx], self.y[idx]]


dataset_train = CreateDataset(train_ids,train_label)
dataset_valid = CreateDataset(valid_ids,valid_label)
dataset_test = CreateDataset(test_ids, test_label)

train_dataloader= DataLoader(dataset_train, batch_size=1, shuffle=True)
valid_dataloader= DataLoader(dataset_valid,batch_size=len(dataset_valid), shuffle=False)
test_dataloader=DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

# print(vars(test_dataloader))

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding=nn.Embedding(vocab_size,embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, x):
        # Initializing hidden state for first input using method defined below
        hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        # Passing in the input and hidden state into the model and obtaining outputs
        embeds=self.embedding(x)
        out, hidden = self.rnn(embeds, hidden)

        out = self.fc(out)
        return out



# model=RNN(vocab_size=len(vocab_dict), embed_dim=300, output_size=4, hidden_dim=5, n_layers=1)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     loss_train = 0.0
#     for i, (inputs, labels) in enumerate(train_dataloader):
#         optimizer.zero_grad()
#
#         outputs = model.forward(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()




class CNN_NLP(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits
import torch.optim as optim

def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    num_classes=2,
                    dropout=0.5,
                    learning_rate=0.01):
    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
       num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=2,
                        dropout=0.5)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer

loss_fn = nn.CrossEntropyLoss()


def train(model, optimizer, train_dataloader, val_dataloader=None, epochs=10):
    best_accuracy = 0
    for epoch_i in range(epochs):
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | { val_loss: ^ 10.6f} | {val_accuracy: ^ 9.2f} | {time_elapsed: ^ 9.2f}")

        print("\n")
        print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


cnn_rand, optimizer = initilize_model(vocab_size=len(vocab_dict),
                                      embed_dim=300,
                                      learning_rate=0.25,
                                      dropout=0.5)
train(cnn_rand, optimizer, train_dataloader, valid_dataloader, epochs=20)
