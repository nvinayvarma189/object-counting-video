import cv2
import torch
from PIL import Image
from torchvision import transforms

model = torch.jit.load('yolov5s.torchscript.pt')
model.eval()

img = Image.open('medium-traffic.jpg').convert('RGB')
img = img.resize((640, 640))
img_tensor = transforms.ToTensor()(img).unsqueeze_(0)
img_tensor /= 255.0
preds = model(img_tensor)
out = non_max_suppression(out, 0.001, 0.6, labels=[], multi_label=True, agnostic=False)
print(preds[0].shape)