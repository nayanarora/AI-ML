from numpy import unique
from numpy import argmax
from pandas import read_csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# load dataset
dataframe = read_csv('/Users/nayanarora/Desktop/softComputing/MT1/abalone.csv', header=None)
dataset = dataframe.values
# split into input (X) and output (y) variables
X, y = dataset[:, :-1], dataset[:, -1]
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]
# encode strings to integer
y = LabelEncoder().fit_transform(y)
n_class = len(unique(y))
# split data into train and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.33, stratify = y_temp, random_state=42)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)
# define the keras model
model = Sequential()
model.add(Dense(20, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(7, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(n_class, activation='softmax'))


# compile the keras model with a custom learning rate
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

# # compile the keras model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=500, batch_size=20, verbose=2)

# evaluate on test set
yhat = model.predict(X_test)
yhat = argmax(yhat, axis=-1).astype('int')
acc = accuracy_score(y_test, yhat)
print('Testing Accuracy: %.3f' % (acc*100))


yhat = model.predict(X_train)
yhat = argmax(yhat, axis=-1).astype('int')
acc = accuracy_score(y_train, yhat)
print('Training Accuracy: %.3f' % (acc*100))


yhat = model.predict(X_val)
yhat = argmax(yhat, axis=-1).astype('int')
acc = accuracy_score(y_val, yhat)
print('Validation Accuracy: %.3f' % (acc*100))