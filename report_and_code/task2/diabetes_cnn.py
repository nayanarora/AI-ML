from numpy import unique
from numpy import argmax
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# load dataset
dataframe = read_csv('/Users/nayanarora/Desktop/softComputing/Assignment1/data/task2/diabetes_binary.csv', header=None)
X = dataframe.iloc[:, :-1]  # All columns except the last one
y = dataframe.iloc[:, -1]   # The last column
n_features = X.shape[1]
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

# Define the scaler 
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the CNN model
model = Sequential()
model.add(Conv1D(input_shape=(n_features, 1), padding = 'same', filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# fit the keras model on the dataset
myModel = model.fit(X_train, y_train, epochs=50, batch_size=30, verbose=2)
# print(model.metrics_names)
accuracy_test = model.evaluate(X_test, y_test, verbose=0)[1]
accuracy_train = myModel.history['accuracy'][-1]  # Get the final training accuracy

print(f"Training Accuracy: {round(accuracy_train*100,3)}")
print(f"Test Accuracy: {round(accuracy_test*100,3)}")

# print('============================')
# # evaluate on test set
# yhat = model.predict(X_test)
# yhat = argmax(yhat, axis = -1).astype('int')
# print(yhat)
# print(y_test)
# acc = accuracy_score(y_test, yhat)
# print('Overall Accuracy: %.4f' % (acc*100))
