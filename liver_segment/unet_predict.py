# USAGE
# python predict.py

# import the necessary packages
from src import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import openslide
import sys
from src import config
from src.utils.patch_generator import PatchGenerator
import matplotlib
from src.utils.patch_generator import PatchGenerator



def bm_2_rgb_mask(mask):
	shape = list(mask.shape)[:2]
	

def prepare_plot(orig_image,all_mask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
	
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(orig_image)
	ax[1].imshow(all_mask)
	
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Full mask")
	
	# set the layout of the figure and display it
	figure.tight_layout()
	plt.show()


def make_predictions(model, image_path,mask_path):

	# set model to evaluation mode
	model.eval()
	
	# turn off gradient tracking
	with torch.no_grad():
		# Get image dimensions
		image = openslide.OpenSlide(image_path)
		dimensions = list((np.array(image.level_dimensions[1])*config.PATCH_RESIZE).astype(int))

		# Create out mask
		out_dim = dimensions.copy()
		out_dim = out_dim[::-1]
		in_mask = np.zeros(out_dim,dtype=np.uint8)
		out_mask = np.zeros(out_dim,dtype=np.uint8)

		patch_generator = PatchGenerator(
			image_path=image_path,
			mask_path=mask_path
		)


		for idx in range(patch_generator.get_num_windows()):
			image,mask = patch_generator[idx]
			image = image.astype(np.float32)/255.0
			image = np.transpose(image, (2, 0, 1))
			image = np.expand_dims(image, 0)
			image = torch.from_numpy(image).to(config.DEVICE)

			pred_mask = model(image).squeeze()
			# pred_mask = torch.sigmoid(pred_mask)
			pred_mask = pred_mask.cpu().numpy()
			# pred_mask = 1.0*(pred_mask > config.THRESHOLD)
			pred_mask = np.transpose(pred_mask,(1,2,0))
			pred_mask = np.argmax(pred_mask,axis=2)
			mask = np.argmax(mask,axis=2)

			start_pixel = list((np.array(patch_generator.get_start_pixel(idx))*config.PATCH_RESIZE).astype(int))
			start_pixel = start_pixel[::-1]
			window = list((np.array(config.PATCH_WINDOW)*config.PATCH_RESIZE).astype(int))

			in_mask[
				start_pixel[0]:(start_pixel[0]+window[0]),
				start_pixel[1]:(start_pixel[1]+window[1]),
			] = mask
			out_mask[
				start_pixel[0]:(start_pixel[0]+window[0]),
				start_pixel[1]:(start_pixel[1]+window[1]),
			] = pred_mask
		return in_mask, out_mask
        


# load the image paths in our testing file and randomly select 10
# image paths
if __name__=="__main__":
	file_name = '383732.svs'
	full_image_path = config.TRAIN_BASE_PATH+'/'+file_name
	
	segmentation_type = 'compartment_annotation'
	annotation_path = getattr(config,f'{config.SEGMENTATION_TYPE.upper()}_OUT_BASE_PATH')
	annotation_file = [file for file in os.listdir(annotation_path) if file.startswith(file_name.split('.')[0])][0]
	full_annotation_path = annotation_path + '/' + annotation_file

	# load our model from disk and flash it to the current device
	unet = torch.load(config.COMPARTMENT_ANNOTATION_MODEL_PATH).to(config.DEVICE)

	compartment_in_mask, compartment_output = make_predictions(
		unet,
		full_image_path,
		full_annotation_path
	)
	
	segmentation_type = 'class_annotation'
	annotation_path = getattr(config,f'{config.SEGMENTATION_TYPE.upper()}_OUT_BASE_PATH')
	annotation_file = [file for file in os.listdir(annotation_path) if file.startswith(file_name.split('.')[0])][0]
	full_annotation_path = annotation_path + '/' + annotation_file

	# load our model from disk and flash it to the current device
	unet = torch.load(config.CLASS_ANNOTATION_MODEL_PATH).to(config.DEVICE)

	class_in_mask, class_output = make_predictions(
		unet,
		full_image_path,
		full_annotation_path
	)

	shape = class_output.shape
	out_mask_R = np.zeros(shape=shape)
	out_mask_G = np.zeros(shape=shape)
	out_mask_B = np.zeros(shape=shape)
	# epithelium benign in green color
	out_mask_G[
		(np.where((class_output==1) & (compartment_output==1)))
	] = 255
	out_mask_R[
		(np.where((class_output==1) & (compartment_output==1)))
	] = 0
	out_mask_B[
		(np.where((class_output==1) & (compartment_output==1)))
	] = 0

	# epithelium malignant as yellow color
	out_mask_G[
		(np.where((class_output==1) & (compartment_output==0)))
	] = 255
	out_mask_R[
		(np.where((class_output==1) & (compartment_output==0)))
	] = 0
	out_mask_B[
		(np.where((class_output==1) & (compartment_output==0)))
	] = 255
    
	# stroma as red
	out_mask_G[
		(np.where((class_output==2)))
	] = 0
	out_mask_R[
		(np.where((class_output==2)))
	] = 255
	out_mask_B[
		(np.where((class_output==2)))
	] = 0

	# excluded tissue as blue.	
	out_mask_G[
		(np.where((class_output==0)))
	] = 0
	out_mask_R[
		(np.where((class_output==0)))
	] = 0
	out_mask_B[
		(np.where((class_output==0)))
	] = 255

	out_mask = np.dstack((out_mask_R,out_mask_G,out_mask_B))


	patch_generator = PatchGenerator(
		image_path=full_image_path,
		mask_path=None
	)

	orig_image = patch_generator.get_image().read_region(
		(0,0),
		1,
		patch_generator.get_image().level_dimensions[1]
	)
	orig_image = np.array(orig_image)[:,:,:3]

	prepare_plot(orig_image,out_mask)

