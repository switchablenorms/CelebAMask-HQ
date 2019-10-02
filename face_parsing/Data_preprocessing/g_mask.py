import os
import cv2
import glob
import numpy as np
from utils import make_folder
#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2	 
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = 'CelebAMaskHQ-mask-anno'
folder_save = 'CelebAMaskHQ-mask'
img_num = 30000

make_folder(folder_save)

for k in range(img_num):
	folder_num = k / 2000
	im_base = np.zeros((512, 512))
	for idx, label in enumerate(label_list):
		filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
		if (os.path.exists(filename)):
			print (label, idx+1)
			im=cv2.imread(filename)
			im = im[:, :, 0]
			im_base[im != 0] = (idx + 1)
	
	filename_save = os.path.join(folder_save, str(k) + '.png')
	print (filename_save)
	cv2.imwrite(filename_save, im_base)

