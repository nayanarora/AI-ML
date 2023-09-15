from sklearn.neural_network import MLPClassifier
from pandas import read_csv as rc
from pandas import DataFrame as df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
#from seaborn import kdeplot
from sklearn.metrics import confusion_matrix
from seaborn import set, heatmap

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

df = pd.read_csv('/Users/nayanarora/Desktop/softComputing/MT1/card.csv')
def get_info_dataframe(dataframe):
    print(f"DATAFRAME GENERAL INFO - \n")
    print(dataframe.info(),"\n")
    print(f"DATAFRAME MISSING INFO - \n")
    print(dataframe.isnull().sum(),"\n")
    print(f"DATAFRAME SHAPE INFO - \n")
    print(dataframe.shape)
get_info_dataframe(df)

# print(df.values[:,30])
# numArr = df.values[:,30]
# valid = (numArr == 0).astype(int)
# fraud = (numArr == 1).astype(int)
# print(valid)
# print(fraud)
# print(valid.size)
# print(fraud.size)
# print(len(valid))
# print(len(fraud))

# valid = valid.sample(n=492)

valid = df.values[:,-1]
flag = Counter(valid)
for i,j in flag.items():
    percentage = j/len(valid) *100
    print('Class=%d, Count=%d, Percentage=%.3f%%' % (i, j, percentage))

print("\n\n\n Check for outliers in the data")
print(df.iloc[:,29].describe())

X = df.values[:,1:30]
y = df.values[:,-1]

print(X.shape)
print(y.shape)
# split into input and output columns
#X, y = df.values[:, :-1], df.values[:, -1]

# X = df.values[:,0:5]
# y = df.values[:,5]

# # print(X)
# # print(y)
# # ensure all data are floating point values
# #X = X.astype('float32')
# # encode strings to integer
# #y = LabelEncoder().fit_transform(y)
scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)


X = torch.tensor(X, dtype = torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1,1)
# # split into train and test datasets
# #X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.33, stratify = y_temp, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Number of training examples: {len(X_train)}")
print(f"Number of validation examples: {len(X_val)}")
print(f"Number of test examples: {len(X_test)}")
# determine the number of input features
# #n_features = X.shape[1]

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
X_val = torch.FloatTensor(X_val)

y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)
y_val = torch.Tensor(y_val)

class fraudClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(fraudClassifier,self).__init__()
        self.input_layer    = nn.Linear(input_dim,24)
        self.hidden_layer1  = nn.Linear(24,12)
        self.output_layer   = nn.Linear(12,output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        out =  self.sigmoid(self.input_layer(x))
        out =  self.sigmoid(self.hidden_layer1(out))
        out =  self.output_layer(out)
        return out


input_dim  = 29
output_dim = 1
model = fraudClassifier(input_dim, output_dim)
print(model)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

def train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs,train_losses,test_losses):
    for epoch in range(num_epochs):
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        
        #forward feed
        output_train = model(X_train)

        #calculate the loss
        loss_train = criterion(output_train, y_train)
        

        #backward propagation: calculate gradients
        loss_train.backward()

        #update the weights
        optimizer.step()

        
        output_test = model(X_test)
        loss_test = criterion(output_test,y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")
num_epochs = 500
train_losses = np.zeros(num_epochs)

test_losses  = np.zeros(num_epochs)

train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs,train_losses,test_losses)


predictions_train = []
predictions_test =  []
with torch.no_grad():
    predictions_train = model(X_train)
    predictions_test = model(X_test)
    predictions_validation = model(X_val)
# Check how the predicted outputs look like and after taking argmax compare with y_train or y_test 
#predictions_train  
#y_train,y_test

def get_accuracy_multiclass(pred_arr,original_arr):
    if len(pred_arr)!=len(original_arr):
        return False
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred= []
    # we will get something like this in the pred_arr [32.1680,12.9350,-58.4877]
    # so will be taking the index of that argument which has the highest value here 32.1680 which corresponds to 0th index
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0
    #here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)
train_acc = get_accuracy_multiclass(predictions_train,y_train)
test_acc  = get_accuracy_multiclass(predictions_test,y_test)
validation_acc = get_accuracy_multiclass(predictions_validation,y_val)
print(f"Training Accuracy: {round(train_acc*100,3)}")
print(f"Test Accuracy: {round(test_acc*100,3)}")
print(f"Validation Accuracy: {round(validation_acc*100,3)}")
