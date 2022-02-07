"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from models import create_model

import torch
import onnx
from onnx_tf.backend import prepare
import torchvision.transforms as transforms

#Make Exporter class
class Exporter(torch.nn.Module):
    def __init__(self, model):
        super(Exporter, self).__init__()
        self.trainmodel = model

        self.resize = transforms.Resize([256, 256])

    def forward(self, x):
        #Select Channel
        x = x[:,:,:,:3]
        outputResize = transforms.Resize([ x.size(1), x.size(2) ])
        #Preprocess
        x = x.permute(0, 3, 1, 2).to(torch.float32)        
        x = self.resize(x); # 256, 256
        x = x / 255.0
        x = x*2 -1


        output = self.trainmodel(x)
        output = (output+1) / 2
        output = output * 255.0
        output = output.to(torch.uint8)
        output = outputResize(output)
        output = output.permute((0, 2, 3, 1))
        output = torch.cat( [output, torch.zeros_like(output[:,:,:,:1])], dim=3 )

        return output


if __name__ == '__main__':


    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and pr

    # exit()

    ###################### complete loading model

    # expot onnx
    onnx_filename = "testtt"
    file_path = "onnx/" + onnx_filename + ".onnx"

    print("Export onnx file : ", file_path)

    if  hasattr(model.netG, "module"):
        model_A = model.netG.module
    else:
        model_A = model.netG

    exporterModel = Exporter(model_A)

    exporterModel.to(device="cpu")
    exporterModel.eval()
    dummy_input = torch.zeros((1, 600, 600, 3), dtype=torch.uint8)

    with torch.no_grad():
        torch.onnx.export(
            exporterModel, dummy_input, file_path,
            verbose = True,
            do_constant_folding = True,
            opset_version = 12,
            input_names = ["input"],
            output_names = ["output"],
            dynamic_axes={
                'input' : {1 : 'width', 2 : 'height', 3 : 'channels'},
                'output' : {1 : 'width', 2 : 'height'}
            }
        )
