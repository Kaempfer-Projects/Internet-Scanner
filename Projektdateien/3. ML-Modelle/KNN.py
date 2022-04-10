#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from mlxtend.preprocessing import minmax_scaling

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.DataFrame()

#Input
# Daten Laden
file = "D:/Undersampling12000/balanced.csv"
df = pd.read_csv(
        file,
        header=0,
        dtype=float,
        sep=";",
        engine='python')


#sns.countplot(x='treffer', data=df)
#plt.xlabel('Internetscanner')
#plt.ylabel('Anzahl')

#plt.savefig('D:/Modelle/Bilder/Internetscanner.png', dpi=300, bbox_inches='tight')


#%%

#Encode Output Class
df['treffer'] = df['treffer'].astype('category')

#Trennung Labels (Input sind alle Spalten auser die letzte (Treffer) -> X, Output ist Treffer -> Y)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

#Split in Train- und Testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

#Set Model Parameters
EPOCHS = 7000
BATCH_SIZE = 256
LEARNING_RATE = 0.003

## train dataclass TrainData(Dataset):

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data


    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]


    def __len__(self):
        return len(self.X_data)


train_data = TrainData(torch.FloatTensor(X_train.to_numpy(dtype=np.float64)),
                       torch.FloatTensor(y_train.to_numpy(dtype=np.float64)))

#%%
## test data

class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data


    def __getitem__(self, index):
        return self.X_data[index]


    def __len__(self):
        return len(self.X_data)


test_data = TestData(torch.FloatTensor(X_test.to_numpy(dtype=np.float64)))

#Initialize Dataloader

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

#Define Neural Network

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 40.

        self.layer_1 = nn.Linear(41, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.bn3(x)
        x = self.layer_out(x)

        return x

##Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##Initialze Optimizer
model = BinaryClassification()
model.to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#loss and accuracy
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

#%%
model.train()
epoch_losses = []
epoch_accs = []
n_samples = len(train_loader)
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_losses.append(epoch_loss / n_samples)
    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / n_samples:.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    epoch_accs.append(epoch_acc / n_samples)

# save the model
torch.save(model.state_dict(), "model_weights.pt")
#%%

##Evaluation
y_pred_list = []
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

#confusion matrix
print(confusion_matrix(y_test, y_pred_list))

#Classification Report
print(classification_report(y_test, y_pred_list))


#################
################




# %%
plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('D:/Modelle/Bilder/losses-FFN.png', dpi=300, bbox_inches='tight')

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from mlxtend.preprocessing import minmax_scaling

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.DataFrame()

#Input
# Daten Laden
file = "D:/Undersampling5000/balanced.csv"
df = pd.read_csv(
        file,
        header=0,
        dtype=float,
        sep=";",
        engine='python')


sns.countplot(x='treffer', data=df)
#plt.xlabel('Internetscanner')
#plt.ylabel('Anzahl')

plt.savefig('D:/Modelle/Bilder/Internetscanner.png', dpi=300, bbox_inches='tight')


#%%

#Encode Output Class
df['treffer'] = df['treffer'].astype('category')

#Trennung Labels (Input sind alle Spalten auser die letzte (Treffer) -> X, Output ist Treffer -> Y)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

#Split in Train- und Testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

#Set Model Parameters
EPOCHS = 7000
BATCH_SIZE = 256
LEARNING_RATE = 0.0003

## train dataclass TrainData(Dataset):

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data


    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]


    def __len__(self):
        return len(self.X_data)


train_data = TrainData(torch.FloatTensor(X_train.to_numpy(dtype=np.float64)),
                       torch.FloatTensor(y_train.to_numpy(dtype=np.float64)))

#%%
## test data

class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data


    def __getitem__(self, index):
        return self.X_data[index]


    def __len__(self):
        return len(self.X_data)


test_data = TestData(torch.FloatTensor(X_test.to_numpy(dtype=np.float64)))

#Initialize Dataloader

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

#Define Neural Network

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 40.

        self.layer_1 = nn.Linear(41, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.bn3(x)
        x = self.layer_out(x)

        return x

##Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##Initialze Optimizer
model = BinaryClassification()
model.to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#loss and accuracy
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

#%%
model.train()
epoch_losses = []
epoch_accs = []
n_samples = len(train_loader)
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_losses.append(epoch_loss / n_samples)
    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / n_samples:.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    epoch_accs.append(epoch_acc / n_samples)

# save the model
torch.save(model.state_dict(), "model_weights.pt")
#%%

##Evaluation
y_pred_list = []
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

#confusion matrix
print(confusion_matrix(y_test, y_pred_list))
## Confusion Matrix wird leider nicht geprinted. Keine Ahnung wieso.

#Classification Report
print(classification_report(y_test, y_pred_list))


#################
################




# %%
plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('D:/Modelle/Bilder/losses-FFN.png', dpi=300, bbox_inches='tight')

plt.plot(epoch_accs)
plt.xlabel('Epoch')
plt.ylabel('Präzision')
plt.savefig('D:/Modelle/Bilder/acc-FFN.png', dpi=300, bbox_inches='tight')


#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from mlxtend.preprocessing import minmax_scaling

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.DataFrame()

#Input
# Daten Laden
file = "D:/Undersampling5000/balanced.csv"
df = pd.read_csv(
        file,
        header=0,
        dtype=float,
        sep=";",
        engine='python')


sns.countplot(x='treffer', data=df)
#plt.xlabel('Internetscanner')
#plt.ylabel('Anzahl')

plt.savefig('D:/Modelle/Bilder/Internetscanner.png', dpi=300, bbox_inches='tight')


#%%

#Encode Output Class
df['treffer'] = df['treffer'].astype('category')

#Trennung Labels (Input sind alle Spalten auser die letzte (Treffer) -> X, Output ist Treffer -> Y)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

#Split in Train- und Testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

#Set Model Parameters
EPOCHS = 7000
BATCH_SIZE = 256
LEARNING_RATE = 0.0003

## train dataclass TrainData(Dataset):

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data


    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]


    def __len__(self):
        return len(self.X_data)


train_data = TrainData(torch.FloatTensor(X_train.to_numpy(dtype=np.float64)),
                       torch.FloatTensor(y_train.to_numpy(dtype=np.float64)))

#%%
## test data

class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data


    def __getitem__(self, index):
        return self.X_data[index]


    def __len__(self):
        return len(self.X_data)


test_data = TestData(torch.FloatTensor(X_test.to_numpy(dtype=np.float64)))

#Initialize Dataloader

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

#Define Neural Network

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 40.

        self.layer_1 = nn.Linear(41, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.bn3(x)
        x = self.layer_out(x)

        return x

##Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##Initialze Optimizer
model = BinaryClassification()
model.to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#loss and accuracy
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

#%%
model.train()
epoch_losses = []
epoch_accs = []
n_samples = len(train_loader)
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_losses.append(epoch_loss / n_samples)
    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / n_samples:.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    epoch_accs.append(epoch_acc / n_samples)

# save the model
torch.save(model.state_dict(), "model_weights.pt")
#%%

##Evaluation
y_pred_list = []
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

#confusion matrix
print(confusion_matrix(y_test, y_pred_list))
## Confusion Matrix wird leider nicht geprinted. Keine Ahnung wieso.

#Classification Report
print(classification_report(y_test, y_pred_list))


#################
################




# %%
plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('D:/Modelle/Bilder/losses-FFN.png', dpi=300, bbox_inches='tight')

plt.plot(epoch_accs)
plt.xlabel('Epoch')
plt.ylabel('Präzision')
plt.savefig('D:/Modelle/Bilder/acc-FFN.png', dpi=300, bbox_inches='tight')


plt.plot(epoch_accs, epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Präzision')
plt.savefig('D:/Modelle/Bilder/test.png', dpi=300, bbox_inches='tight')


