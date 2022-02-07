import torch
import os
import PIL

from utils import *
from parameter import *
from torchvision import transforms
import torch.nn as nn
from unet import unet

# python -u .\onnx_export_test.py --batch_size {bsz} --test_size {test_data_num} --imsize {resize} --version parsenet --train false

def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    
    return transform

#Make Exporter class
class Exporter(torch.nn.Module):
    def __init__(self, model):
        super(Exporter, self).__init__()
        self.trainmodel = model
        
        options = []
        # options.append(transforms.ToTensor())
        options.append(transforms.Resize((512,512), interpolation=PIL.Image.NEAREST))    
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(options)

    def forward(self, x):
        # #Preprocessing
        
        x = x[:,:,:3]
        x = x.to(torch.float32)
        x = x / 255.0
        x = x.permute(2, 0, 1)
        x = self.transform(x).unsqueeze(0)
        
        y = self.trainmodel(x)
        y = torch.argmax(y, 1).to(torch.uint8)
        mask = y == 10
        mask_n = y != 10
        y[mask] = 255
        y[mask_n] = 0
        #TODO : 치아에 해당하는 부분만 filtering, 나머지 0
        
        
        return y


class Tester(object):
    
    def __init__(self, config):
        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.test_label_path = config.test_label_path
        self.test_color_label_path = config.test_color_label_path
        self.test_image_path = config.test_image_path

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        self.build_model()

    def build_model(self):
        self.G = unet()
        if self.parallel:
            self.G = nn.DataParallel(self.G)

if __name__ == "__main__":
    config = get_parameters()
    tester = Tester(config)
    tester.G.load_state_dict(torch.load(os.path.join(tester.model_save_path, tester.model_name)))
    # tester.G.eval() 
    
    exporterModel = Exporter(tester.G)
    # exporterModel.to(device="cpu")
    exporterModel.eval()
    
    dummy_input = torch.zeros((512, 512, 3)).to(torch.uint8)
    
    file_path = "onnx/test.onnx"
    
    with torch.no_grad():
        torch.onnx.export(
            exporterModel, dummy_input, file_path,
            verbose = True,
            do_constant_folding = True,
            opset_version = 12,
            input_names = ["input"],
            output_names = ["output"],
            dynamic_axes={
                'input' : {0 : 'width', 1 : 'height', 2 : 'channels'},
                'output' : {0 : 'width', 1 : 'height'}
            }
        )