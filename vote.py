from sklearn.neural_network import MLPClassifier
from pandas import read_csv
from pandas import DataFrame as df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
#from seaborn import kdeplot
from sklearn.metrics import confusion_matrix
from seaborn import set, heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


df = read_csv('/Users/nayanarora/Desktop/softComputing/MT1/Vote.csv')
def get_info_dataframe(dataframe):
    print(f"DATAFRAME GENERAL INFO - \n")
    print(dataframe.info(),"\n")
    print(f"DATAFRAME MISSING INFO - \n")
    print(dataframe.isnull().sum(),"\n")
    print(f"DATAFRAME SHAPE INFO - \n")
    print(dataframe.shape)
get_info_dataframe(df)

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]

# X = df.values[:,0:5]
# y = df.values[:,5]

# print(X)
# print(y)
# ensure all data are floating point values
#X = X.astype('float32')
# encode strings to integer
#y = LabelEncoder().fit_transform(y)
X = torch.tensor(X, dtype = torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1,1)
# split into train and test datasets
#X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.33, stratify = y_temp, random_state=42)

print(f"Number of training examples: {len(X_train)}")
print(f"Number of validation examples: {len(X_val)}")
print(f"Number of test examples: {len(X_test)}")
# determine the number of input features
#n_features = X.shape[1]

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
X_val = torch.Tensor(X_val)

y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)
y_val = torch.Tensor(y_val)

class voteClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(16,10)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(10,1)
        self.act_output = nn.Sigmoid()

    def forward(self,x):
        x = self.act1(self.hidden1(x))
        x = self.act_output(self.output(x))
        return x 

model = voteClassifier()
print(model)

loss_fn = nn.BCELoss() #binary cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

n_epochs = 100
batch_size = 10
# Check the dimensions of input data and weights
# print("Input data shape:", X_train.shape)
# print("act1 weight shape:", model.hidden1.weight.shape)
# print("output weight shape:", model.output.weight.shape)

def evaluate(model, X, y):
    for epoch in range(n_epochs):
        for i in range(0,len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            #y_pred = y_pred.view(-1)
            Ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, Ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f'Finished epoch {epoch}, latest loss {loss}')

    with torch.no_grad():
        y_pred = model(X)

    accuracy = (y_pred.round() == y).float().mean()
    print(f'Accuracy {accuracy*100}')

    # make class predictions with the model
    # predictions = (model(X) > 0.5).int()
    # for i in range(5):
    #     print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i],y[i]))
    return y_pred.round()



print(f"\n\n\nCalculating for Training Accuracy")
train_accuracy = evaluate(model, X_train, y_train)

print(f"\n\n\nCalculating for Validation Accuracy")
val_accuracy = evaluate(model, X_val, y_val)
    
print(f"\n\n\nCalculating for Test Accuracy")
test_accuracy = evaluate(model, X_test, y_test)



cm = confusion_matrix(y_test, test_accuracy)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust the font size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
            xticklabels=['Republican', 'Democrat'],
            yticklabels=['Republican', 'Democrat'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Test split')
plt.show()


cm = confusion_matrix(y_val, val_accuracy)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust the font size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
            xticklabels=['Republican', 'Democrat'],
            yticklabels=['Republican', 'Democrat'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Validation split')
plt.show()


cm = confusion_matrix(y_train, train_accuracy)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust the font size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
            xticklabels=['Republican', 'Democrat'],
            yticklabels=['Republican', 'Democrat'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Train split')
plt.show()