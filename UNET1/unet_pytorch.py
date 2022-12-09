# Importing bunch of libraries
import os
import sys
import time
import random
import warnings

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.model_selection import train_test_split

from model import UNet
import logging

seed = 42
random.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# Dataset 1: HGR
TRAIN_PATH = ['/content/drive/MyDrive/Datasets/finalimages']
MASK_PATH = ['/content/drive/MyDrive/Datasets/finalmasks']
train_ids = next(os.walk(TRAIN_PATH[0]))[2]
mask_ids = next(os.walk(MASK_PATH[0]))[2]
train_ids.sort()
mask_ids.sort()
TRAIN_PATH = TRAIN_PATH*len(train_ids)
MASK_PATH = MASK_PATH*len(train_ids)



#  Parameters to prepare training and test data.
# Specify image dimensions
IMG_WIDTH = 572
IMG_HEIGHT = 572
IMG_CHANNELS = 3
# Batch size for training.
batch_size = 3
# Control the size of split of training vs testing.
train_to_test_split_ratio = 0.8

# Flag to control if data preparation is required. It should be true when running the notebook for the first time or if the image dimensions are changed.
prepare_data = True

if prepare_data:
    # This creates two array of zeros (for the ground truth and mask data) to store the images in them. Note the images are
    # expected to be in channel foirst format.
    images = np.zeros((len(train_ids), IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
    labels = np.zeros((len(train_ids), 1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    g = list(range(0, len(train_ids)))
    np.random.shuffle(g)

    # Creates string arrays to store the path for every training image
    strs_original = ["" for x in range(len(train_ids))]
    strs_mask = ["" for x in range(len(train_ids))]
    pathmsk = MASK_PATH[0] + mask_ids[0]
    # Store images path in the corresponding arrays (one array for masks, one for the original ones)
    for n, id_ in tqdm.tqdm(enumerate(train_ids), total=len(train_ids)):
        strs_mask[n] = MASK_PATH[n] + mask_ids[n]
        strs_original[n] = TRAIN_PATH[n] + train_ids[n]

    # Read images from their paths and store them in arrays
    for n, id_ in tqdm.tqdm(enumerate(train_ids), total=len(train_ids)):
        #  Process image.
        path = strs_original[g[n]]
        img = np.asarray(imread(path)[:, :, :IMG_CHANNELS])
        # Resize the image to fixed dimension.
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant')
        # Make the image channel first. (NHWC -> NCHW)
        img = np.transpose(img, (2, 0, 1))
        images[n] = img

        #  Process masks.
        path = strs_mask[g[n]]
        mask = np.asarray(imread(path))
        if mask.ndim == 3:
            mask = mask[:, :, 1]
        # Resize the image to fixed dimension.
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant'), axis=-1)
        # Make binary lables.
        mask = mask > 0
        # Make the image channel first. (NHWC -> NCHW)
        mask = np.transpose(mask, (2, 0, 1))
        labels[n] = mask

#  Save and load the images and labels so that you dont have to run the above step everytime.
if prepare_data:
    print("Saved the shuffled images and labels locally.")
    np.save("images", images)
    np.save("labels", labels)
else:
    print("Loaded the shuffled images and labels from local path.")
    images = np.load('./images.npy')
    labels = np.load('./labels.npy')

    # Split the Training and Test dataset.
    random_state = 1  # To get reproducible results.
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels,
                                                                            train_size=train_to_test_split_ratio,
                                                                            random_state=random_state, shuffle=True)

    # Create tuple pair of training data.
    train_data = []
    for i in range(len(images_train)):
        train_data.append([images_train[i], labels_train[i]])

    # Create tuple pair of testing data.
    test_data = []
    for i in range(len(images_test)):
        test_data.append([images_test[i], labels_test[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=1)

# Iterate over the trainloader.
images, labels = next(iter(trainloader))
print("Image Tensor type: ",images.dtype)
print("Labels Tensor type: ",labels.dtype)
image = images[0].numpy()
image = np.transpose(image, (1,2,0))
imshow(image)
plt.show()
label = labels[0].numpy()
label = np.transpose(label, (1,2,0))
imshow(label)
plt.show()

# Iterate over the testloader.
images, labels = next(iter(testloader))
image = images[0].numpy()
image = np.transpose(image, (1,2,0))
imshow(image)
plt.show()
label = labels[0].numpy()
imshow(label[0])
plt.show()

model = UNet(input_channels=3)
model.to(device=device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 5
for epoch in range(num_epochs):  # loop over the dataset multiple times
    print("Epoch: ", epoch+1)
    loss_over_batches = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device=device, dtype=torch.float)
        labels = labels.to(device=device,dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        loss_over_batches.append(loss.item())
        if i % 100 == 0:    # print every 100 mini-batches
            print("Training loss: {}, Batches Processed: {}".format((np.sum(loss_over_batches) / len(loss_over_batches)),len(loss_over_batches)))

print('Finished Training')

# Saving the model
PATH = '../Models/unet_pytorch.pth'
torch.save(model.state_dict(), PATH)

# Load the saved model for inference.
PATH = '../Models/unet_pytorch.pth'
model = UNet(input_channels=3)
model.load_state_dict(torch.load(PATH))
model.to(device=device)
model.eval()

# Sanity check on random training samples
images, labels = next(iter(trainloader))
image_arr = images[0].numpy()
image_arr = np.transpose(image_arr, (1,2,0))
imshow(image_arr)
plt.show()

label = labels[0].numpy()
imshow(label[0])
plt.show()

images = images.to(device=device, dtype=torch.float)
prediction = model(images)
prediction = prediction[0].cpu().detach().numpy()
prediction = np.transpose(prediction, (1,2,0))
# prediction = prediction > 0.7
imshow(prediction)
plt.show()

# Sanity check on random testing samples
images, labels = next(iter(testloader))
image_arr = images[0].numpy()
image_arr = np.transpose(image_arr, (1,2,0))
imshow(image_arr)
plt.show()

label = labels[0].numpy()
imshow(label[0])
plt.show()

images = images.to(device=device, dtype=torch.float)
prediction = model(images)
prediction = prediction[0].cpu().detach().numpy()
prediction = np.transpose(prediction, (1,2,0))
# prediction = prediction > 0.7
imshow(prediction)
plt.show()
