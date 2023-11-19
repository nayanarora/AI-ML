##Using everything learned in the tutorials in week 5 and week 6 we use the VGG16BN and InceptionV3 pretrained models om the chest-xray dataset provided. 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader                   
import torchvision.datasets as datasets                   
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import pandas as pd
import torchvision.models as models 
import matplotlib.pyplot as plt
# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
  
# load the image
img_path = '/Users/nayanarora/Desktop/softComputing/tut4/chestX/test/normal/IM-0011-0001-0001.jpeg'
img = Image.open(img_path)
  
# convert PIL image to numpy array
img_np = np.array(img)
  
# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")

# Python code for converting PIL Image to
# PyTorch Tensor image and plot pixel values

# import necessary libraries
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# define custom transform function
transform = transforms.Compose([
	transforms.ToTensor()
])

# transform the pIL image to tensor
# image
img_tr = transform(img)

# Convert tensor image to numpy array
img_np = np.array(img_tr)

# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")

img_tr = transform(img)
  
# calculate mean and std
mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
  
# print mean and std
print("mean and std before normalize:")
print("Mean of the image:", mean)
print("Std of the image:", std)

# python code to normalize the image


from torchvision import transforms

# define custom transform
# here we are using our calculated
# mean & std
transform_norm = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
])

# get normalized image
img_normalized = transform_norm(img)

# convert normalized image to numpy
# array
img_np = np.array(img_normalized)

# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")

# Python Code to visualize normalized image

# get normalized image
img_normalized = transform_norm(img)

# convert tis image to numpy array
img_normalized = np.array(img_normalized)

# transpose from shape of (3,,) to shape of (,,3)
img_normalized = img_normalized.transpose(1, 2, 0)

# display the normalized image
plt.imshow(img_normalized)
plt.xticks([])
plt.yticks([])

# Python code to calculate mean and std
# of normalized image
  
# get normalized image
img_nor = transform_norm(img)
  
# cailculate mean and std
mean, std = img_nor.mean([1,2]), img_nor.std([1,2])
  
# print mean and std
print("Mean and Std of normalized image:")
print("Mean of the image:", mean)
print("Std of the image:", std)

variable = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.5 for _ in range(3)], [0.2 for _ in range(3)]),
    ]
)

Train_data = datasets.ImageFolder(root = '/Users/nayanarora/Desktop/softComputing/Assignment1/data/task3/ChestXray/train', transform =variable )
Train_loader = DataLoader(Train_data, batch_size = 12, shuffle= True, drop_last=True)
val_data = datasets.ImageFolder(root = '/Users/nayanarora/Desktop/softComputing/Assignment1/data/task3/ChestXray/test', transform =variable )
val_loader = DataLoader(val_data, batch_size = 4, shuffle= True, drop_last=True)
# test_data = datasets.ImageFolder(root = '', transform =variable )
# test_loader = DataLoader(test_data, batch_size = 12, shuffle= True, drop_last=True)

# Checking the images from the test dataset
examples = enumerate(Train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print('Shape of the Image: ',example_data.shape)
print('Shape of the label: ', example_targets.shape)
print(example_targets[0:6])

#checking the labels
class_name_train = Train_data.classes
print(class_name_train)
print(Train_data.class_to_idx)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    #print(example_targets[i])
    plt.xticks([])
    plt.yticks([])
fig

# Set Device
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    device = torch.device("mps")
    print("===============")
    print (x)
else:
    print ("MPS device not found.")
    device = 'cpu'
num_classes = 2

##CNN using vgg16_bn pretrained model
# class VGG16BNNet(nn.Module):
#     def __init__(self):
#         super(VGG16BNNet, self).__init__()
#         self.model = models.vgg16_bn(pretrained=True)
#         # Freeze model weights
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.model.classifier[6] = nn.Linear(4096, num_classes)

#     def forward(self, x):
#         return self.model(x)

# ##CNN using 
# class InceptionV3Net(nn.Module):
#     def __init__(self):
#         super(InceptionV3Net, self).__init__()
#         self.model = models.inception_v3(pretrained=True)
#         # Freeze model weights
#         for param in self.model.parameters():
#             param.requires_grad = False
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_ftrs, num_classes)

#     def forward(self, x):
#         return self.model(x)

class VGG16BNNet(nn.Module):
    def __init__(self):
        super(VGG16BNNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Freeze model weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the final fully connected layer for the number of classes
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)

class InceptionV3Net(nn.Module):
    def __init__(self):
        super(InceptionV3Net, self).__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        
        # Freeze model weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the final fully connected layer for the number of classes
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.model.num_classes = num_classes

    def forward(self, x):
        return self.model(x)
    

model_vgg16bn = VGG16BNNet().to(device)
model_inceptionv3 = InceptionV3Net().to(device)

# Setting the hyperparameters
learning_rate = 0.001
num_epochs = 5
channel_img = 3
feature_d = 64

criterion = nn.CrossEntropyLoss()
optimizer_vgg16bn = optim.Adam(model_vgg16bn.parameters(), lr=learning_rate)
optimizer_inceptionv3 = optim.Adam(model_inceptionv3.parameters(), lr=learning_rate)

# Training and validation loop for VGG16BN
num_total_steps = len(Train_loader.dataset)
valid_loss_min = np.Inf
step = 0
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []
accuracies = []
steps = []
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0 
    
    model_vgg16bn.train()
    for _ , (images, labels) in enumerate(Train_loader):
        images = images.to(device)
        #print(len(images))
        labels = labels.to(device)
        
        #forward
        outputs = model_vgg16bn(images)
        loss = criterion(outputs, labels)
        training_loss.append(loss.item())
        
        
        # backwards and optimizer
        optimizer_vgg16bn.zero_grad()
        loss.backward()
        optimizer_vgg16bn.step()
        
        # Calculating running training accuracies
        _, predictions = outputs.max(1)
        num_correct = (predictions == labels).sum()
        running_train_acc = float(num_correct)/float(images.shape[0])
        training_accuracy.append(running_train_acc)
        train_acc += running_train_acc
        train_loss += loss.item()
        
        avg_train_acc = train_acc / len(Train_loader)
        avg_train_loss = train_loss / len(Train_loader)
        
        model_vgg16bn.eval()

        ##Calculating the validation loss and validation accuracies. 

        with torch.no_grad():
            for _ , (images, labels) in enumerate(val_loader):
                images = images.to(device)
                #print(len(images))
                labels = labels.to(device)
                
                outputs = model_vgg16bn(images)
                loss = criterion(outputs, labels)
                validation_loss.append(loss)
                
                # Calculating running training accuracies
                _, predictions = outputs.max(1)
                num_correct = (predictions == labels).sum()
                running_val_acc = float(num_correct)/float(images.shape[0])
                validation_accuracy.append(running_val_acc)
                val_acc += running_val_acc
                val_loss += loss.item()
            
            avg_valid_acc = val_acc / len(val_loader)
            avg_valid_loss = val_loss / len(val_loader)
        step += 1
        steps.append(step)

    print("Epoch : {} VGG16 Train Loss : {:.6f} VGG16 Train Acc : {:.6f}".format(epoch+1,avg_train_loss,avg_train_acc))
    print("Epoch : {} VGG16 Valid Loss : {:.6f} VGG16 Valid Acc : {:.6f}".format(epoch+1,avg_valid_loss,avg_valid_acc))

plt.title('VGG 16 Training Loss')
plt.xlabel('Steps')
plt.ylabel('Losses')
plt.plot(steps, training_loss)
plt.show()

plt.title('VGG 16 Training accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(steps, training_accuracy)
plt.show()

plt.title('VGG 16 Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Losses')
plt.plot(steps, validation_loss)
plt.show()

plt.title('VGG 16 Validation accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(steps, validation_accuracy)
plt.show()

#============================

# Training and validation loop for InceptionV3NET
num_total_steps = len(Train_loader.dataset)
valid_loss_min = np.Inf
step = 0
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []
accuracies = []
steps = []
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0 
    
    model_inceptionv3.train()
    for _ , (images, labels) in enumerate(Train_loader):
        images = images.to(device)
        #print(len(images))
        labels = labels.to(device)
        
        #forward
        outputs = model_inceptionv3(images)
        loss = criterion(outputs, labels)
        training_loss.append(loss.item())
        
        
        # backwards and optimizer
        optimizer_inceptionv3.zero_grad()
        loss.backward()
        optimizer_inceptionv3.step()
        
        # Calculating running training accuracies
        _, predictions = outputs.max(1)
        num_correct = (predictions == labels).sum()
        running_train_acc = float(num_correct)/float(images.shape[0])
        training_accuracy.append(running_train_acc)
        train_acc += running_train_acc
        train_loss += loss.item()
        
        avg_train_acc = train_acc / len(Train_loader)
        avg_train_loss = train_loss / len(Train_loader)
        
        model_inceptionv3.eval()

        ##Calculating the validation loss and validation accuracies. 

        with torch.no_grad():
            for _ , (images, labels) in enumerate(val_loader):
                images = images.to(device)
                #print(len(images))
                labels = labels.to(device)
                
                outputs = model_inceptionv3(images)
                loss = criterion(outputs, labels)
                validation_loss.append(loss)
                
                # Calculating running training accuracies
                _, predictions = outputs.max(1)
                num_correct = (predictions == labels).sum()
                running_val_acc = float(num_correct)/float(images.shape[0])
                validation_accuracy.append(running_val_acc)
                val_acc += running_val_acc
                val_loss += loss.item()
            
            avg_valid_acc = val_acc / len(val_loader)
            avg_valid_loss = val_loss / len(val_loader)
        step += 1
        steps.append(step)

    print("Epoch : {} InceptionV3 Train Loss : {:.6f} InceptionV3 Train Acc : {:.6f}".format(epoch+1,avg_train_loss,avg_train_acc))
    print("Epoch : {} InceptionV3 Valid Loss : {:.6f} InceptionV3 Valid Acc : {:.6f}".format(epoch+1,avg_valid_loss,avg_valid_acc))

plt.title('InceptionV3 Training Loss')
plt.xlabel('Steps')
plt.ylabel('Losses')
plt.plot(steps, training_loss)
plt.show()

plt.title('InceptionV3 Training accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(steps, training_accuracy)
plt.show()

plt.title('InceptionV3 Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Losses')
plt.plot(steps, validation_loss)
plt.show()

plt.title('InceptionV3 Validation accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(steps, validation_accuracy)
plt.show()
