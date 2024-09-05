import os
from src.utils.patch_generator import  PatchGenerator
from src import config

def generate_num_patches(image_info_df):
    image_info_df['num_patches'] = 0

    for idx,row in image_info_df.iterrows():
        if row['type'] == 'train':
            base_path = config.TRAIN_BASE_PATH
        else:
            base_path = config.TEST_BASE_PATH
        
        image_path = os.path.join(base_path,row['file'])
        patches = PatchGenerator(
            image_path=image_path,
            mask_path=None,
            window=config.PATCH_WINDOW,
            slide=config.PATCH_SLIDER
        )
        image_info_df.loc[idx,'num_patches'] = patches.get_num_windows()
    image_info_df['cumul_num_patches'] = 0
