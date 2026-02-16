from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np

'''
A simple way to structure this would be defining x,y,width and height.
But our objects are not uniform in any sense.
What we could do is, return a numpy arr of same h,w as img with each
pixel representing an object id.
'''
@dataclass
class SegmentationResult:
    segmentation_masks: np.ndarray




class Model(ABC):
    def __init__(self, model_name, model_path):
        self.model_name= model_name
        self.model_path = model_path 

    # model name, params, or any other helpful info
    @abstractmethod
    def get_model_specs(self)

    # different models require image to be loaded in different formats
    @abstractmethod
    def load_image(self, image_path)

    # all these models are used for segementing images.
    # they need to return the segmented mask in a shared format 
    # so it can be passed to the nextjs frontend easily. 
    @abstractmethod
    def segment(self)-> shared_format_object



