from django.shortcuts import render, redirect
from uploadimg.forms import *

# import the necessary packages
# from torchvision.models import detection
from torchvision.models import detection
import torchvision
from torchvision import transforms
import numpy as np
import argparse
import pickle
import torch
import cv2
from PIL import Image as PILImage
import random
import os
from uploadimg.detector_config import MODELS

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

# MODELS = {
# 	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
# 	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
# 	"retinanet": detection.retinanet_resnet50_fpn
# }


def detect_objects(img_path,model):

    source_img = PILImage.open(img_path).convert("RGB")
    orig = cv2.imread(img_path)

    # source_img = transform_image(source_img)

    my_transforms = transforms.Compose([transforms.ToTensor()])
    source_img = my_transforms(source_img).unsqueeze(0)

    detections = model(source_img)[0]
    num_of_cars=0

    rand_color=np.random.uniform(0, 255, size=(1, 3))[0]

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.7:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            if idx == 3:
                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                # display the prediction to our terminal
                label = "{}: {:.2f}%".format("car", confidence * 100)
                # label = "{}: {:.2f}%".format(idx, confidence * 100)
                print("[INFO] {}".format(label))
                # draw the bounding box and label on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                            rand_color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rand_color, 2)
                num_of_cars+=1
    # show the output image
    print(f'Total number of cars detected in the image: {num_of_cars}')

    path_elements=os.path.normpath(img_path).split(os.path.sep)

    filename=path_elements[-1].split('.')[0]+'_detection.'+path_elements[-1].split('.')[1]

    filename_with_path=os.path.join(path_elements[0], path_elements[1], filename)
    print(filename_with_path)
    # img_path=os.path.join(path_elements[1], path_elements[2], path_elements[3]) 
    cv2.imwrite(filename_with_path, orig)
    # cv2.waitKey(0)

    return num_of_cars


def index(request):
    """Process images uploaded by users"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.image_labeled = "failed_to_process"
            obj.save()
            blabla= obj.image.url
            path_elements=os.path.normpath(obj.image.url).split(os.path.sep)

            img_path=os.path.join(path_elements[1], path_elements[2], path_elements[3])
            # orig = cv2.imread(img_path)

            model = MODELS[obj.detector_model](pretrained=True)
            model.eval()
            num_of_cars=detect_objects(img_path,model)


            filename=path_elements[-1].split('.')[0]+'_detection.'+path_elements[-1].split('.')[1]

            detection_img_path=os.path.join(path_elements[2], filename)

            obj.image_labeled = detection_img_path
            obj.num_of_objects = str(num_of_cars)
            # "/images/whoop2"
            obj.save()

            img_obj = form.instance

            return render(request, 'index.html', {'form': form, 'img_obj': img_obj, 'num_of_objects': num_of_cars})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})



def display_images(request):
  
    if request.method == 'GET':
  
        # getting all the objects of hotel.
        images = Image.objects.all() 
        return render(request, 'display_images.html', {'images' : images})