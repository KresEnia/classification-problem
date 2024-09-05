import torch
import os

# Check if GPU exists to use
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# If GPU exists, copy to GPU memory
PIN_MEMORY = True if DEVICE == 'cuda' else False

# Segmentation parameters
NUM_CLASSES = 4 # 3
INIT_LR = 0.001
NUM_EPOCHS = 20
BATCH_SIZE= 8
SEGMENTATION_TYPE = "class_annotation"#"compartment_annotation"

# Patch window size and slider
# A slider that is smaller than the  window size
# means there will be an overlap between patches.
PATCH_WINDOW = (600,600)
PATCH_SLIDER = (600,600)
PATCH_RESIZE = 1.0

# IMAGE PATHS
TRAIN_BASE_PATH="/media/project_for_segmentation/images/train"
TEST_BASE_PATH="/media/project_for_segmentation/images/test"
CLASS_ANNOTATION_BASE_PATH="/media//project_for_segmentation/masks/class_annotations"
COMPARTMENT_ANNOTATION_BASE_PATH="/media/project_for_segmentation/masks/compartments_annotations"
CLASS_ANNOTATION_OUT_BASE_PATH="/media/project_for_segmentation/masks/class_annotations_out"
COMPARTMENT_ANNOTATION_OUT_BASE_PATH="/media/project_for_segmentation/masks/compartments_annotations_out"



BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

CLASS_ANNOTATION_MODEL_PATH = os.path.join(BASE_OUTPUT, "class_unet.pth")
CLASS_ANNOTATION_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "class_unet.png"])
CLASS_ANNOTATION_TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "class_unet.txt"])

COMPARTMENT_ANNOTATION_MODEL_PATH = os.path.join(BASE_OUTPUT, "compartment_unet.pth")
COMPARTMENT_ANNOTATION_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "compartment_unet.png"])
COMPARTMENT_ANNOTATION_TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "compartment_unet.txt"])

# Weights
COMPARTMENT_ANNOTATION_WEIGHTS = [0.5,0.5,0.25]
CLASS_ANNOTATION_WEIGHTS = [1,1,1,0.1]