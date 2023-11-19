import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('/Users/nayanarora/Desktop/softComputing/Assignment1/data/task4/Miles_Traveled.csv')
timeseries = df[["TRFVOLUSM227NFWA"]].values.astype('float32')
scaler = MinMaxScaler()
df['TRFVOLUSM227NFWA'] = scaler.fit_transform(df['TRFVOLUSM227NFWA'].values.reshape(-1, 1))
plt.plot(timeseries)
plt.show()
# train-test split for time series
train_size = int(len(timeseries) * 0.70)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 3
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

print("shape of the train dataset X and y\n")
print(X_train.shape, y_train.shape)
print("shape of the train dataset X and y\n")
print(X_test.shape, y_test.shape)

class miles_lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=2, batch_first=True)
        self.linear = nn.Linear(100, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = miles_lstm_model()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.MSELoss()
#loss_fn = nn.L1Loss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=15)

train_losses, test_losses = [], []
n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        train_losses.append(train_rmse.item())
        
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        test_losses.append(test_rmse.item())

    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

plt.figure(figsize=(10, 5))
plt.plot(range(0, n_epochs, 100), train_losses, label='Train RMSE')
plt.plot(range(0, n_epochs, 100), test_losses, label='Test RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Training and Test RMSE over Epochs')
plt.legend()
plt.grid(True)
plt.show()
# with torch.no_grad():
#     # shift train predictions for plotting
#     train_plot = np.ones_like(timeseries) * np.nan
#     y_pred = model(X_train)
#     y_pred = y_pred[:, -1, :]
#     train_plot[lookback:train_size] = model(X_train)[:, -1, :]
#     # shift test predictions for plotting
#     test_plot = np.ones_like(timeseries) * np.nan
#     test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# # plot
# plt.plot(timeseries, c='b', label = 'actual data')
# plt.plot(train_plot, c='r', label = 'training set')
# plt.plot(test_plot, c='g', label = 'test set')  
# plt.legend()
# plt.show()