# import the necessary packages
from torch.utils.data import Dataset
import cv2
import os
from src.utils.patch_generator import PatchGenerator
from src import config

class SegmentationDataset(Dataset):
	_patch_generator = None
	_file_name = None

	def __init__(self, image_df, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_df = image_df
		self.transforms = transforms
	
	def __len__(self):
		# return the number of total samples contained in the dataset
		return self.image_df['num_patches'].sum()
	
	def __getitem__(self, idx):
		seg_type = config.SEGMENTATION_TYPE.upper()
		
		out_df = self.image_df[self.image_df['cumul_num_patches']<=idx]
		
		row = out_df.iloc[-1,:]
		if row['type'] == 'train':
			base_path = config.TRAIN_BASE_PATH
		else:
			base_path = config.TEST_BASE_PATH
        
		if self._file_name and row['file'] == self._file_name:
			patch_generator = self._patch_generator
			idx = idx - row['cumul_num_patches']
		else:
			image_path = os.path.join(base_path,row['file'])
			mask_path = os.path.join(
				getattr(config,f'{seg_type}_OUT_BASE_PATH'),
				row[f'{seg_type.lower()}_file']
			)
			idx = idx - row['cumul_num_patches']
			patch_generator = PatchGenerator(
				image_path=image_path,
				mask_path=mask_path,
				window=config.PATCH_WINDOW,
				slide=config.PATCH_SLIDER,
				randomize=True,
				rotate_flip=True
			)
			self._patch_generator = patch_generator
			self._file_name = row['file']
		image,mask = patch_generator[idx]
		image = image.copy()
		mask = mask.copy()

		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return (image, mask)