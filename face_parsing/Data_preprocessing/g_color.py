import os
from PIL import Image
import glob
import numpy as np
from utils import make_folder
color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

folder_base = 'CelebAMask-HQ-mask'
folder_save = 'CelebAMask-HQ-mask-color'
img_num = 10

make_folder(folder_save)

for k in range(img_num):
    filename = os.path.join(folder_base, str(k) + '.png')
    if (os.path.exists(filename)):
        im_base = np.zeros((512, 512, 3))
        im = Image.open(filename)
        im = np.array(im)
        for idx, color in enumerate(color_list):
            im_base[im == idx] = color
    filename_save = os.path.join(folder_save, str(k) + '.png')
    result = Image.fromarray((im_base).astype(np.uint8))
    print (filename_save)
    result.save(filename_save)

