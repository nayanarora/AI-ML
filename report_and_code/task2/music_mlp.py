from numpy import unique
from numpy import argmax
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# load dataset
dataframe = read_csv('/Users/nayanarora/Desktop/softComputing/Assignment1/data/task2/music_genre.csv', header=None)
X = dataframe.iloc[:, :-1]  # All columns except the last one
y = dataframe.iloc[:, -1]   # The last column
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]
# encode strings to integer
y = LabelEncoder().fit_transform(y)
n_class = len(unique(y))

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

# Define the scaler 
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

# define the keras model
model = Sequential()
model.add(Dense(60, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(40, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(n_class, activation='softmax'))

# compile the keras model
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# fit the keras model on the dataset
myModel = model.fit(X_train, y_train, epochs=50, batch_size=40, verbose=2)

accuracy_test = model.evaluate(X_test, y_test, verbose=0)[1]
accuracy_train = myModel.history['accuracy'][-1]  # Get the final training accuracy

print(f"Training Accuracy: {round(accuracy_train*100,3)}")
print(f"Test Accuracy: {round(accuracy_test*100,3)}")

print('============================')
# evaluate on test set
yhat = model.predict(X_test)
yhat = argmax(yhat, axis=-1).astype('int')
acc = accuracy_score(y_test, yhat)
print('Overall Accuracy: %.4f' % (acc*100))
