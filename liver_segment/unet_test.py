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
# matplotlib.use('Agg')
# plt.style.use("ggplot")


def bm_2_rgb_mask(mask):
	shape = list(mask.shape)[:2]
	if config.SEGMENTATION_TYPE == 'class_annotation':
		out_mask_0 = np.zeros(shape=shape,dtype=np.uint8)
		out_mask_0[
			np.where(mask==0)
		]=255
		out_mask_1 = np.zeros(shape=shape,dtype=np.uint8)
		out_mask_1[
			np.where(mask==1)
		]=255
		out_mask_2 = np.zeros(shape=shape,dtype=np.uint8)
		out_mask_2[
			np.where(mask==2)
		]=255

		out_mask = np.dstack((out_mask_0,out_mask_1,out_mask_2))
		return out_mask
	else:
		out_mask_0 = np.zeros(shape=shape,dtype=np.uint8)
		out_mask_0[
			np.where(mask==0)
		]=255
		out_mask_1 = np.zeros(shape=shape,dtype=np.uint8)
		out_mask_1[
			np.where(mask==1)
		]=255
		out_mask_0[
			np.where(mask==1)
		]=255
		out_mask_2 = np.zeros(shape=shape,dtype=np.uint8)

		out_mask = np.dstack((out_mask_0,out_mask_1,out_mask_2))
		return out_mask


def prepare_plot(orig_image, in_mask, out_mask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(orig_image)
	ax[1].imshow(in_mask)
	ax[2].imshow(out_mask)
	
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	
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
		out_dim.append(3)
		in_mask = np.zeros(out_dim,dtype=np.uint8)
		out_mask = np.zeros(out_dim,dtype=np.uint8)

		patch_generator = PatchGenerator(
			image_path=image_path,
			mask_path=mask_path
		)

		for idx in range(patch_generator.get_num_windows()):
			image,mask = patch_generator[idx]
			# cv2.imshow("test",image)
			# cv2.waitKey(0)
			# continue
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
				:
			] = bm_2_rgb_mask(mask)
			out_mask[
				start_pixel[0]:(start_pixel[0]+window[0]),
				start_pixel[1]:(start_pixel[1]+window[1]),
				:
			] = bm_2_rgb_mask(pred_mask)
			


		orig_image = patch_generator.get_image().read_region(
			(0,0),
			1,
			patch_generator.get_image().level_dimensions[1]
		)
		orig_image = np.array(orig_image)[:,:,:3]
		orig_image = cv2.resize(orig_image,(0,0),fx=config.PATCH_RESIZE,fy=config.PATCH_RESIZE)

		# Resize everything for viewing
		k = 0.1		
		orig_image = cv2.resize(orig_image,(0,0),fx=k,fy=k,interpolation=cv2.INTER_NEAREST)
		in_mask = cv2.resize(in_mask,(0,0),fx=k,fy=k,interpolation=cv2.INTER_NEAREST)
		out_mask = cv2.resize(out_mask,(0,0),fx=k,fy=k,interpolation=cv2.INTER_NEAREST)
		prepare_plot(orig_image,in_mask,out_mask)



# load the image paths in our testing file and randomly select 10
# image paths
if __name__=="__main__":
	file_name = '383732.svs'
	full_image_path = config.TRAIN_BASE_PATH+'/'+file_name

	annotation_path = getattr(config,f'{config.SEGMENTATION_TYPE.upper()}_OUT_BASE_PATH')
	annotation_file = [file for file in os.listdir(annotation_path) if file.startswith(file_name.split('.')[0])][0]
	full_annotation_path = annotation_path + '/' + annotation_file

	# load our model from disk and flash it to the current device
	unet = torch.load(getattr(config,f'{config.SEGMENTATION_TYPE.upper()}_MODEL_PATH')).to(config.DEVICE)

	make_predictions(
		unet,
		full_image_path,
		full_annotation_path
	)