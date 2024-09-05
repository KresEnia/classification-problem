# USAGE
# python unet_train.py

# import the necessary packages
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from src import config

from src.pyimagesearch.dataset import SegmentationDataset
from src import config
from src.pyimagesearch.model import UNet
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import sys
from src.utils import general_utils
from src.utils import generate_image_info
from print_color import print
import numpy as np
import matplotlib
matplotlib.use('Agg')


os.system('clear')
print("START",tag='INFO',tag_color='green',format='bold',color='white')


print('Generating image info...',tag='PREP',tag_color='blue',color='white',end='',flush=True)
image_info_df = generate_image_info.generate_image_info()
print('Done',color='white')


print('Calculating number of patches...',tag='PREP',tag_color='blue',color='white',end='',flush=True)
general_utils.generate_num_patches(image_info_df)
print('Done',color='white')

# Get training images that are annotated
print('Create training and testing dataframes...',tag='PREP',tag_color='blue',color='white',end='',flush=True)
training_df = image_info_df[(image_info_df['type']=='train') & (image_info_df[config.SEGMENTATION_TYPE]==True)]
testing_df = image_info_df[(image_info_df['type']=='test') & (image_info_df[config.SEGMENTATION_TYPE]==True)]
training_df = training_df.reset_index(drop=True)
testing_df = testing_df.reset_index(drop=True)

x = training_df.loc[:training_df.shape[0]-2,'num_patches'].cumsum().to_numpy()
training_df.loc[1:,'cumul_num_patches'] = x

x = testing_df.loc[:testing_df.shape[0]-2,'num_patches'].cumsum().to_numpy()
testing_df.loc[1:,'cumul_num_patches'] = x
print('Done',color='white')


print(training_df.head())
print('Setup image transforms...',tag='PREP',tag_color='blue',color='white',end='',flush=True)
# define transformations
transforms = transforms.Compose([transforms.ToTensor()])
print('Done',color='white')

# create the train and test datasets
print('Instantiate training and testing dataset classes...',tag='PREP',tag_color='blue',color='white',end='',flush=True)
train_ds = SegmentationDataset(image_df=training_df, transforms=transforms)
test_ds = SegmentationDataset(image_df=testing_df, transforms=transforms)
print('Done',color='white')

print(f'Data has {len(train_ds)} training patches and {len(test_ds)} testing patches',tag='INFO',tag_color='blue',color='white',flush=True)

print('Creating training and testing data loaders...',tag='PREP',tag_color='blue',color='white',end='',flush=True)
# create the training and test data loaders
train_loader = DataLoader(train_ds, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=4)
test_loader = DataLoader(test_ds, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=4)
print('Done',color='white')


print('Initializing UNet models...',tag='MODEL',tag_color='blue',color='white',end='',flush=True)
# initialize our UNet model
unet = UNet(
	nbClasses=config.NUM_CLASSES,
	outSize=list((np.array(config.PATCH_WINDOW)*config.PATCH_RESIZE).astype(int))
).to(config.DEVICE)
print('Done',color='white')



print('Initializing loss models...',tag='MODEL',tag_color='blue',color='white',end='',flush=True)
# initialize loss function and optimizer

if config.SEGMENTATION_TYPE  == 'compartment_annotation':
	weights = np.array(config.COMPARTMENT_ANNOTATION_WEIGHTS)
else:
	weights = np.array(config.CLASS_ANNOTATION_WEIGHTS)

weights = torch.from_numpy(weights)
weights = weights.to(config.DEVICE)

lossFunc = torch.nn.CrossEntropyLoss(weight=weights,reduction='sum')
opt = Adam(unet.parameters(), lr=config.INIT_LR)
print('Done',color='white')

# calculate steps per epoch for training and test set
trainSteps = len(train_ds) // config.BATCH_SIZE
testSteps = len(test_ds) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("Training the network...",tag='SEGM',tag_color='green',color='white')
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0

	# loop over the training set
	for (i, (x, y)) in enumerate(train_loader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)

		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far
		totalTrainLoss += loss
	
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()

		# loop over the validation set
		for (x, y) in test_loader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# make the predictions and calculate the validation loss
			pred = unet(x)
			pred_2= torch.sigmoid(pred)
			
			totalTestLoss += lossFunc(pred, y)
	
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))



# serialize the model to disk
if config.SEGMENTATION_TYPE.startswith('class'):
	torch.save(unet, config.CLASS_ANNOTATION_MODEL_PATH)
else:
	torch.save(unet, config.COMPARTMENT_ANNOTATION_MODEL_PATH)


# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

if config.SEGMENTATION_TYPE.startswith('class'):
	plt.savefig(config.CLASS_ANNOTATION_PLOT_PATH)
else:
	plt.savefig(config.COMPARTMENT_ANNOTATION_PLOT_PATH)
