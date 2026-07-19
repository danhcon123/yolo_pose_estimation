from dataclasses import dataclass
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

@dataclass
class ImgSize:
    height: int
    width:int
    channel: int = 3
        
    def get_tuple(self)-> tuple:
        return (self.height, self.width, self.channel)


def letterbox(img: np.ndarray, new_size:ImgSize, fill_value: int=0) -> np.ndarray:        
    aspect_ratio = min(new_size.height / img.shape[1], new_size.width / img.shape[0])
    
    new_size_with_ar = int(img.shape[1] * aspect_ratio), int(img.shape[0] * aspect_ratio)
    
    resized_img = np.asarray(cv2.resize(img, new_size_with_ar))
    resized_h, resized_w, _ = resized_img.shape
    
    padded_img = np.full(new_size.get_tuple(), fill_value)
    center_x = new_size.width / 2
    center_y = new_size.height / 2
    
    x_range_start = int(center_x - (resized_w / 2))
    x_range_end = int(center_x + (resized_w / 2))
    
    y_range_start = int(center_y - (resized_h / 2))
    y_range_end = int(center_y + (resized_h / 2))
    
    padded_img[y_range_start: y_range_end, x_range_start: x_range_end, :] = resized_img
    return padded_img

image_path = os.path.join(
    "Beispielbilder",
    "laufer_poster.jpg"
    #"lhccoutinho-soccer-4275827_1280.jpg"
)
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_lb_mine = letterbox(np.asarray(img_rgb), ImgSize(640, 640))
plt.imshow(img_lb_mine)
plt.show()