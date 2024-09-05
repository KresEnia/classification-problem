import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import config


def convert_class_annotation_masks():
    in_path = config.CLASS_ANNOTATION_BASE_PATH
    out_path = config.CLASS_ANNOTATION_OUT_BASE_PATH
    os.makedirs(out_path,exist_ok=True)
    files = os.listdir(in_path)

    for file in files:
        mask_path = os.path.join(
            in_path,
            file
        )
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask,(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_NEAREST)

        out_image_0 = np.zeros(shape=mask.shape[:2],dtype=np.uint8)
        out_image_0[(np.where((mask[:,:,0]==255) & (mask[:,:,1]==0) & (mask[:,:,2]==0)))] = 1

        out_image_1 = np.zeros(shape=mask.shape[:2],dtype=np.uint8)
        out_image_1[(np.where((mask[:,:,0]==0) & (mask[:,:,1]==255) & (mask[:,:,2]==0)))] = 1


        out_image_2 = np.zeros(shape=mask.shape[:2],dtype=np.uint8)
        out_image_2[(np.where((mask[:,:,0]==0) & (mask[:,:,1]==0) & (mask[:,:,2]==255)))] = 1

        # out_image_3 = np.zeros(shape=mask.shape[:2],dtype=np.uint8)
        # out_image_3[(np.where((mask[:,:,0]==128) & (mask[:,:,1]==128) & (mask[:,:,2]==128)))] = 1

        out_image_3 = np.zeros(shape=mask.shape[:2],dtype=np.uint8)
        out_image_3[(np.where((mask[:,:,0]==0) & (mask[:,:,1]==0) & (mask[:,:,2]==0)))] = 1
        
        out_mask_path = os.path.join(
            out_path,
            file
        )
        np.savez_compressed(out_mask_path,out_image_0,out_image_1,out_image_2,out_image_3)



def convert_compartment_annotation_masks():
    in_path = config.COMPARTMENT_ANNOTATION_BASE_PATH
    out_path = config.COMPARTMENT_ANNOTATION_OUT_BASE_PATH
    os.makedirs(out_path,exist_ok=True)
    files = os.listdir(in_path)

    for file in files:
        mask_path = os.path.join(
            in_path,
            file
        )
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask,(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_NEAREST)
        out_image_0 = np.zeros(shape=mask.shape[:2],dtype=np.uint8)
        out_image_0[(np.where((mask[:,:,0]==0) & (mask[:,:,1]==255) & (mask[:,:,2]==0)))] = 1


        out_image_1 = np.zeros(shape=mask.shape[:2],dtype=np.uint8)
        out_image_1[(np.where((mask[:,:,0]==0) & (mask[:,:,1]==255) & (mask[:,:,2]==255)))] = 1

        out_image_2 = np.zeros(shape=mask.shape[:2],dtype=np.uint8)
        out_image_2[(np.where((mask[:,:,0]==0) & (mask[:,:,1]==0) & (mask[:,:,2]==0)))] = 1

        out_mask_path = os.path.join(
            out_path,
            file
        )
        np.savez_compressed(out_mask_path,out_image_0,out_image_1,out_image_2)


if __name__=="__main__":
    convert_class_annotation_masks()
    # convert_compartment_annotation_masks()