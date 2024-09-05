import numpy as np
import openslide
from src import config
import cv2

class PatchGenerator:
    image = None
    mask = None
    window = None
    slide = None
    idx = None


    def __init__(self,image_path=None,mask_path=None,window=None,slide=None,randomize=False,rotate_flip=False):
        self.image_path = image_path
        self.mask_path = mask_path
        self.window = window or config.PATCH_WINDOW
        self.slide = slide or config.PATCH_SLIDER
        self.idx = 0
        self.randomize=randomize
        self.rotate_flip=rotate_flip


    @property
    def dimensions(self):
        return self.get_image().level_dimensions[1]

    def get_image(self):
        if self.image is not None:
            return self.image
        self.image = openslide.OpenSlide(self.image_path)
        return self.image

    def get_mask(self):
        if self.mask is not None:
            return self.mask
        if self.mask_path is None:
            return
        buffer = np.load(self.mask_path)
        self.mask = np.dstack(list(buffer.values()))
        return self.mask

    def __iter__(self):
        self.idx = 0
        return self
    
    def get_num_windows(self):
        num_windows_x = (np.floor((self.dimensions[0] - self.window[0])/self.slide[0])+1)
        num_windows_y = (np.floor((self.dimensions[1] - self.window[1])/self.slide[1])+1) 
        num_windows =  num_windows_x * num_windows_y
        return int(num_windows)

    def apply_rotate_flip(self,image,mask):
        # Check if we have to flip
        if np.random.choice([True,False]):
            image = np.flip(image,1)
            if mask is not None:
                mask = np.flip(mask,1)
        
        k = np.random.choice([0,1,2,3])
        if k!=0:
            image = np.rot90(image,k,(0,1))
            if mask is not None:
                mask = np.rot90(mask,k,(0,1))
        return image,mask

    def get_start_pixel(self,idx):
        num_windows_x = (np.floor((self.dimensions[0] - self.window[0])/self.slide[0])+1)
        start = (int(idx%num_windows_x*self.slide[0]),int(idx//num_windows_x*self.slide[1]))
        return start

    def __getitem__(self,idx):
        num_windows_x = (np.floor((self.dimensions[0] - self.window[0])/self.slide[0])+1)
        num_windows_y = (np.floor((self.dimensions[1] - self.window[1])/self.slide[1])+1) 
        num_windows =  num_windows_x * num_windows_y
        if idx >= num_windows: 
            raise IndexError
        
        if self.randomize:
            idx = np.random.randint(0,self.get_num_windows())
        
        start = (int(idx%num_windows_x*self.slide[0]),int(idx//num_windows_x*self.slide[1]))
        image = self.get_image_patch(start)
        mask = self.get_mask_patch(start)  

        if self.rotate_flip:
            image,mask=self.apply_rotate_flip(image,mask)

        return image,mask

    def get_image_patch(self,start):
        start = [start[0]*4,start[1]*4]
        image = np.array(
            self.get_image().read_region(
                start,
                1,
                self.window
            ),
            dtype=np.uint8
        )[:,:,:3]
        image = cv2.resize(image,(0,0),fx=config.PATCH_RESIZE,fy=config.PATCH_RESIZE,interpolation=cv2.INTER_NEAREST)
        return image

    def get_mask_patch(self,start):
        if self.get_mask() is None:
            return
        mask = self.get_mask()[
            start[1]:start[1]+self.window[1],
            start[0]:start[0]+self.window[0],
            :
        ]
        mask = cv2.resize(mask,(0,0),fx=config.PATCH_RESIZE,fy=config.PATCH_RESIZE,interpolation=cv2.INTER_NEAREST)
        return mask