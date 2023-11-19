from numpy import unique, argmax
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from keras.layers import Dropout

# Load the dataset
dataframe = read_csv('/Users/nayanarora/Desktop/softComputing/Assignment1/data/task2/music_genre.csv', header=None)
X = dataframe.iloc[:, :-1]  # All columns except the last one
y = dataframe.iloc[:, -1]   # The last column
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]

# Encode strings to integers
y = LabelEncoder().fit_transform(y)
n_class = len(unique(y))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Define the scaler
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

# Reshape input data for 1D convolution
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the CNN model
model = Sequential()
model.add(Conv1D(input_shape=(n_features, 1), filters=64, kernel_size=5, padding = 'same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(n_class, activation='softmax'))

# Compile the CNN model
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fit the CNN model on the dataset
myModel = model.fit(X_train, y_train, epochs=50, batch_size=40, verbose=2)

accuracy_test = model.evaluate(X_test, y_test, verbose=0)[1]
accuracy_train = myModel.history['accuracy'][-1]  # Get the final training accuracy

print(f"Training Accuracy: {round(accuracy_train*100,3)}")
print(f"Test Accuracy: {round(accuracy_test*100,3)}")

print('============================')

# # Evaluate on test set
# yhat = model.predict(X_test)
# yhat = argmax(yhat, axis=-1).astype('int')
# acc = accuracy_score(y_test, yhat)
# print('Overall Accuracy: %.4f' % (acc*100))
