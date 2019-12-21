# Face parsing
A Pytorch implementation face parsing model trained by CelebAMask-HQ
## Dependencies
* Pytorch 0.4.1
* numpy
* Python3
* Pillow
* opencv-python
* tenseorboardX
## Preprocessing
* Move the mask folder, the image folder, and `CelebA-HQ-to-CelebA-mapping.txt` ( remove 1st line in advance ) under `./Data_preprocessing`
* Run `python g_mask.py`
* Run  `python g_partition.py` to split train set and test set.
## Training
* Run `bash run.sh #GPU_num`
## Well-trained model
* The model can be downloaded [here](https://drive.google.com/file/d/1o1m-eT38zNCIFldcRaoWcLvvBtY8S4W3/view?usp=sharing).
* The model (`model.pth`) should be put under `./models/parsenet`
* Mask labels are defined as following:

| Label list | | |
| ------------ | ------------- | ------------ |
| 0: 'background' | 1: 'skin' | 2: 'nose' |
| 3: 'eye_g' | 4: 'l_eye' | 5: 'r_eye' |
| 6: 'l_brow' | 7: 'r_brow' | 8: 'l_ear' |
| 9: 'r_ear' | 10: 'mouth' | 11: 'u_lip' |
| 12: 'l_lip' | 13: 'hair' | 14: 'hat' |
| 15: 'ear_r' | 16: 'neck_l' | 17: 'neck' |
| 18: 'cloth' | | |

* Overall Per-pixel Acc: 93.42 ( train and evaluate according to CelebA train/test split )
## Testing & Color visualization
* Run `bash run_test.sh #GPU_num`
* Results will be saved in `./test_results`
* Color visualized results will be saved in `./test_color_visualize`
* Another way for color visualization without using GPU: Run `python ./Data_preprocessing/g_color.py` 
