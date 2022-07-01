from django.db import models
from torchvision.models import detection

from uploadimg.detector_config import MODEL_CHOICES

# COLOR_CHOICES = (
#     ('green','GREEN'),
#     ('blue', 'BLUE'),
#     ('red','RED'),
#     ('orange','ORANGE'),
#     ('black','BLACK'),
# )

# MODEL_CHOICES = (
# 	("frcnn-resnet",'Faster R-CNN with ResNet50 backbone'),
# 	("frcnn-mobilenet",'Faster R-CNN with MobileNetv3 backbone'),
# 	("retinanet",'RetinaNet with ResNet50 backbone')
# )

# model = MODELS[args["model"]](pretrained=True, progress=True, num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)

# class MyModel(models.Model):
#   color = models.CharField(max_length=6, choices=COLOR_CHOICES, default='green')


class Image(models.Model):
    title = models.CharField(max_length=200,default='car_detector')
    image = models.ImageField(upload_to='images')
    image_labeled = models.ImageField()
    detector_model = models.CharField(max_length=200, choices=MODEL_CHOICES)#, default='green'
    num_of_objects = models.CharField(max_length=200,default='failed')
    def __str__(self):
        return self.title
    class Meta:
        db_table = "myapp_image"