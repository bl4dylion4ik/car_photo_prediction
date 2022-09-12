import os

from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


CAR_IDX = [656, 627, 817, 511, 468, 751, 705, 757, 717, 734, 654, 675, 864, 609, 436]  # car classes in imagenet dataset
THRESH = 0.35
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


def process_img(filename: str):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def is_car_acc_prob(predictions: torch.Tensor, thresh: float):
    prob = np.array(torch.nn.functional.softmax(predictions), dtype=float)
    car_probs = prob[:, CAR_IDX]
    car_probs_acc = car_probs.sum(axis=1)
    return car_probs_acc > thresh


def removing_img(filenames: list):
    for filename in filenames:
        os.remove(filename)


def filter_img(photo_dir: str):
    unwanted_img = []
    for brand in os.listdir(photo_dir):
        brand_dir = os.path.join(photo_dir, brand)
        for model in os.listdir(brand_dir):
            model_brand_dir = os.path.join(brand_dir, model)
            for img in os.listdir(model_brand_dir):
                img_file = os.path.join(model_brand_dir, img)
                input_img = process_img(img_file)

                if torch.cuda.is_available():
                    input_img = input_img.to('cuda')
                    model.to('cuda')

                with torch.no_grad():
                    output = model(input_img)
                res = is_car_acc_prob(output, THRESH)
                if not res:
                    unwanted_img.append(img_file)
    return unwanted_img


if __name__ == '__main__':
    photo_dir = os.path.join(os.getcwd(), 'av-photos')
    removing_img(filter_img(photo_dir))
