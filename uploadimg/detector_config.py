from torchvision.models import detection

import torch
import torchvision

def create_frcnn_resnet():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load('uploadimg/fasterrcnn_resnet50_fpn_coco.pth'))
    return model

def create_frcnn_mobilenet():
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)
    model.load_state_dict(torch.load('uploadimg/fasterrcnn_mobilenet_v3_large_320_fpn.pth'))
    return model

def create_retinanet():
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load('uploadimg/retinanet_resnet50_fpn_coco.pth'))
    return model

def create_ssd_vgg16():
    model = torchvision.models.detection.ssd300_vgg16(pretrained=False)
    model.load_state_dict(torch.load('uploadimg/ssd300_vgg16_coco.pth'))
    return model



#this is the container for the model options that will be displayed to the user as dropdown box
MODEL_CHOICES = (
	("frcnn-resnet",'Faster R-CNN with ResNet50 backbone'),
	("frcnn-mobilenet",'Faster R-CNN with MobileNetv3 backbone'),
	("retinanet",'RetinaNet with ResNet50 backbone'),
    ("ssd-vgg16",'SSD with VGG16 backbone')
)

#here the code-word of a model needs to be assiosiated with a particular model
#from the torchivision pre-trained models farm
# MODELS = {
# 	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
# 	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
# 	"retinanet": detection.retinanet_resnet50_fpn,
#     "ssd-vgg16": detection.ssd300_vgg16,
# }

print("LOADING DETECTION MODELS")

frcnn_resnet=create_frcnn_resnet()
frcnn_mobilenet=create_frcnn_mobilenet()
retinanet=create_retinanet()
ssd_vgg16=create_ssd_vgg16()

MODELS = {
	"frcnn-resnet": frcnn_resnet,
	"frcnn-mobilenet": frcnn_mobilenet,
	"retinanet": retinanet,
    "ssd-vgg16": ssd_vgg16,
}

#be aware that when a new model is added it needs to be downloaded the first time it is picked