import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['/home/dos/Desktop/workspace/object-counting-video/medium-traffic.jpg']  # batch of images

# Inference
results = model(imgs)
results.save()