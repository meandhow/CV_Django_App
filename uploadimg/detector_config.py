from torchvision.models import detection

#this is the container for the model options that will be displayed to the user as dropdown box
MODEL_CHOICES = (
	("frcnn-resnet",'Faster R-CNN with ResNet50 backbone'),
	("frcnn-mobilenet",'Faster R-CNN with MobileNetv3 backbone'),
	("retinanet",'RetinaNet with ResNet50 backbone'),
    ("ssd-vgg16",'SSD with VGG16 backbone')
)

#here the code-word of a model needs to be assiosiated with a particular model
#from the torchivision pre-trained models farm
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn,
    "ssd-vgg16": detection.ssd300_vgg16,
}

#be aware that when a new model is added it needs to be downloaded the first time it is picked