
# https://linuxtut.com/en/ae832a461bab8374a517/

import pandas as pd
from sklearn.model_selection import train_test_split
import string
import torch
import gensim
# import spacy
# nlp=spacy.load('en')



# 70. Generating Features through Word Vector Summation
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

print('training length {} valid length {} test length {}'.format(len(train),len(valid),len(test)))
print('category labels:\n train\n {} \nvalid\n {} \ntest\n {}'.format(train['CATEGORY'].value_counts(),
                                valid['CATEGORY'].value_counts(),test['CATEGORY'].value_counts()))




model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',binary=True)



def transform_w2v(text):
    table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    words = text.translate(table).split()

    vec = [model[word] for word in words if word in model]
    return torch.tensor(sum(vec)/len(vec))

X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])


category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

print(X_train.size()), print(y_train.size())


# 71. Building Single Layer Neural Network
from torch import nn

class SingeLayerNet(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.fc = nn.Linear(input_size, output_size, bias=False)
    nn.init.normal_(self.fc.weight)

  def forward(self, x):
    x = self.fc(x)
    return x

model = SingeLayerNet(300,4)
y_hat_1 = torch.softmax(model.forward(X_train[:1]), dim=-1)
print(y_hat_1)

Y_hat = torch.softmax(model.forward(X_train[:4]), dim=-1)
print(Y_hat)


# 72. Calculating loss and gradients
criterion = nn.CrossEntropyLoss()
y=model.forward(X_train[:1])
loss= criterion(y, y_train[:1])
model.zero_grad()
loss.backward()
print('loss: ', loss.item())
print('Slope:\n{}'.format(model.fc.weight.grad))


criterion = nn.CrossEntropyLoss()
y=model.forward(X_train[:4])
loss= criterion(y, y_train[:4])
model.zero_grad()
loss.backward()
print('loss: ', loss.item())
print('Slope:\n{}'.format(model.fc.weight.grad))


# 73. Learning with stochastic gradient descent
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):  # len(dataset)Specify the value to be returned with
        return len(self.y)

    def __getitem__(self, idx):  # dataset[idx]Specify the value to be returned with
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return [self.X[idx], self.y[idx]]


from torch.utils.data import DataLoader

dataset_train = CreateDataset(X_train, y_train)
dataset_valid = CreateDataset(X_valid, y_valid)
dataset_test = CreateDataset(X_test, y_test)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)


model = SingeLayerNet(300,4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    loss_train = 0.0
    for i, (inputs, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

    # Average loss calculation for each batch
    loss_train = loss_train / i

    # Validation
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(dataloader_valid))
        outputs = model.forward(inputs)
        loss_valid = criterion(outputs, labels)
    print('epoch: {}, loss_train: {}, loss_valid: {}'.
          format(epoch + 1,loss_train,loss_valid ))


# 74. Measuring accuracy
def calculate_accuracy(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return correct / total


acc_train = calculate_accuracy(model, dataloader_train)
acc_test = calculate_accuracy(model, dataloader_test)

print(f'Correct answer rate (learning data):{acc_train:.3f}')
print(f'Correct answer rate (evaluation data):{acc_test:.3f}')


# 75. Plotting loss and accuracy
def calculate_loss_and_accuracy(model, criterion, loader):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total

model = SingeLayerNet(300, 4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

num_epochs = 30
log_train = []
log_valid = []
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader_train:
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    print(
        f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')


from matplotlib import pyplot as plt
import numpy as np
# Visualization
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(np.array(log_train).T[0], label='train')
ax[0].plot(np.array(log_valid).T[0], label='valid')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend()
ax[1].plot(np.array(log_train).T[1], label='train')
ax[1].plot(np.array(log_valid).T[1], label='valid')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].legend()
plt.show()


# 76. Checkpoints

model = SingeLayerNet(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# Learning
num_epochs = 10
log_train = []
log_valid = []
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Calculation of loss and correct answer rate
    loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    # Save checkpoint
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
               f'checkpoint{epoch + 1}.pt')

    # Output log
    print(
        f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')

# 77. Mini - batches
import time


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs):
    # Creating a dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    # Learning
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # Record start time
        s_time = time.time()

        # Set to training mode
        model.train()
        for inputs, labels in dataloader_train:
            # Initialize gradient to zero
            optimizer.zero_grad()

            # Forward propagation+Error back propagation+Weight update
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Calculation of loss and correct answer rate
        loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # Save checkpoint
        torch.save(
            {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
            f'checkpoint{epoch + 1}.pt')

        # Record end time
        e_time = time.time()

        # Output log
        print(
            f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec')

    return {'train': log_train, 'valid': log_valid}