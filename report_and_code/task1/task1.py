from numpy import unique
from numpy import argmax
from pandas import read_csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# load dataset
dataframe = read_csv('/Users/nayanarora/Desktop/softComputing/Assignment1/data/task1/dermatology.txt', header=None, delimiter=' ')

#dataset = dataframe.values
# split into input (X) and output (y) variables

# Split the data into input (X) and output (y) variables
X = dataframe.iloc[:, :-1]  # All columns except the last one
y = dataframe.iloc[:, -1]   # The last column

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
model.add(Dense(100, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(n_class, activation='softmax'))

# compile the keras model with a custom learning rate
learning_rate = 0.001
epochs = 100
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

loss_train =[]
loss_test = []
train_accuracy = []
test_accuracy = []
# Train the model and record loss values
for epoch in range(epochs):
    # Train for one epoch
    # fit the keras model on the dataset
    myModel = model.fit(X_train, y_train, epochs=1, batch_size=50, verbose=1)
    
    # Evaluate on the test dataset and record test loss and accuracy
    loss_test_val, accuracy_test_val = model.evaluate(X_test, y_test, verbose=0)

    # Append training and test loss to the lists
    loss_train.append(myModel.history['loss'][0])
    loss_test.append(loss_test_val)
    
    train_accuracy.append(myModel.history['accuracy'][0])
    test_accuracy.append(accuracy_test_val)
    # Print the loss and accuracy for each epoch
    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {myModel.history['loss'][0]:.4f} - Test Loss: {loss_test_val:.4f} - Training Accuracy: {round(myModel.history['accuracy'][0]*100,3)} - Test Accuracy: {round(accuracy_test_val*100,3)}")

# Plot the training and test loss over epochs
plt.figure(figsize=(10, 10))
plt.plot(loss_train, label='Training Loss')
plt.plot(loss_test, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Sparse_categorical_crossentropyLoss for Test and Train over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# evaluate on test set
y_pred = model.predict(X_test)
y_pred = argmax(y_pred, axis=-1)


acc = accuracy_score(y_test, y_pred)

print(f"Final Training Accuracy: {train_accuracy[-1]}")
print(f"Final Test Accuracy: {test_accuracy[-1]}")
#print('Testing Accuracy: %.3f' % (acc*100))

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)