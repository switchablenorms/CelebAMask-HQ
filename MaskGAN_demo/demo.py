import os
import sys
import cv2
import time
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.base_dataset import BaseDataset, get_params, get_transform, normalize

from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
from ui_util.config import Config

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0), QColor(51, 51, 255), QColor(204, 0, 204), QColor(0, 255, 255), QColor(51, 255, 255), QColor(102, 51, 0), QColor(255, 0, 0), QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0)]

class Ex(QWidget, Ui_Form):
    def __init__(self, model, opt):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()
        self.model = model
        self.opt = opt

        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.img = None

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.mode, self.size)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.ref_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
 
        self.result_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.result_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(image)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(image)

    def open_mask(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:    
            mat_img = cv2.imread(fileName)
            self.mask = mat_img.copy()
            self.mask_m = mat_img       
            mat_img = mat_img.copy()
            image = QImage(mat_img, 512, 512, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    

            for i in range(512):
                for j in range(512):
                    r, g, b, a = image.pixelColor(i, j).getRgb()
                    image.setPixel(i, j, color_list[r].rgb()) 
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image)  
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items() 
            self.scene.addPixmap(self.image)

    def bg_mode(self):
        self.scene.mode = 0

    def skin_mode(self):
        self.scene.mode = 1

    def nose_mode(self):
        self.scene.mode = 2

    def eye_g_mode(self):
        self.scene.mode = 3

    def l_eye_mode(self):
        self.scene.mode = 4

    def r_eye_mode(self):
        self.scene.mode = 5

    def l_brow_mode(self):
        self.scene.mode = 6

    def r_brow_mode(self):
        self.scene.mode = 7

    def l_ear_mode(self):
        self.scene.mode = 8

    def r_ear_mode(self):
        self.scene.mode = 9

    def mouth_mode(self):
        self.scene.mode = 10

    def u_lip_mode(self):
        self.scene.mode = 11

    def l_lip_mode(self):
        self.scene.mode = 12

    def hair_mode(self):
        self.scene.mode = 13

    def hat_mode(self):
        self.scene.mode = 14

    def ear_r_mode(self):
        self.scene.mode = 15

    def neck_l_mode(self):
        self.scene.mode = 16

    def neck_mode(self):
        self.scene.mode = 17

    def cloth_mode(self):
        self.scene.mode = 18

    def increase(self):
        if self.scene.size < 15:
            self.scene.size += 1
    
    def decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1 

    def edit(self):
        for i in range(19):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)

        params = get_params(self.opt, (512,512))
        transform_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, normalize_mask=True)
        transform_image = get_transform(self.opt, params)

        mask = self.mask.copy()
        mask_m = self.mask_m.copy()

        mask = transform_mask(Image.fromarray(np.uint8(mask))) 
        mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
        img = transform_image(self.img)
    
        start_t = time.time()
        generated = model.inference(torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]), torch.FloatTensor([img.numpy()]))   
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        #save_image((generated.data[0] + 1) / 2,'./results/1.jpg')
        result = generated.permute(0, 2, 3, 1)
        result = result.cpu().numpy()
        result = (result + 1) * 127.5
        result = np.asarray(result[0,:,:,:], dtype=np.uint8)
        qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)

        #for i in range(512):
        #    for j in range(512):
        #       r, g, b, a = image.pixelColor(i, j).getRgb()
        #       image.setPixel(i, j, color_list[r].rgb()) 
        if len(self.result_scene.items())>0: 
            self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(QPixmap.fromImage(qim))

    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'],(color,color,color),sizes[idx])
        return mask

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                    QDir.currentPath())
            cv2.imwrite(fileName+'.jpg',self.output_img)

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.mask_m = self.mask.copy()
    
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    #model = Model(config)
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    model = create_model(opt)   
    app = QApplication(sys.argv)
    ex = Ex(model, opt)
    sys.exit(app.exec_())
