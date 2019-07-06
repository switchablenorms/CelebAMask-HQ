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
	d_1 = k / 10000
	d_2 = (k / 1000) % 10
	d_3 = (k / 100) % 10
	d_4 = (k / 10) % 10
	d_5 = k % 10 
	folder_num = k / 2000
				
	im_base = np.zeros((512, 512))
	for idx, label in enumerate(label_list):
		filename = os.path.join(folder_base, str(folder_num), str(d_1) + str(d_2) + str(d_3) + str(d_4) + str(d_5) + '_' + label + '.png')
		if (os.path.exists(filename)):
			print (label, idx+1)
			im=cv2.imread(filename)
			im = im[:, :, 0]
			for i in range(512):
				for j in range(512):
					if im[i][j] != 0:
						im_base[i][j] = (idx + 1)
	
	filename_save = os.path.join(folder_save, str(k) + '.png')
	print (filename_save)
	cv2.imwrite(filename_save, im_base)

