import os
import pickle

from PIL import Image
import torch
from torchvision import transforms
import numpy as np

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CLASS_NUM = 576


def process_img(filename: str):
    """Process image to tensor"""
    input_image = Image.open(filename).convert('RGB')
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def predict_one_sample(model, inputs, device=DEVICE):
    """Predict for one sample"""
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit[0], dim=0).numpy()
    return probs


def predict_top_labels(model, filename, top_k, device=DEVICE):
    """Predict top k most probable labels"""
    input_img = process_img(filename)
    probabilities = predict_one_sample(model, input_img[0].unsqueeze(0), device=device)

    top_predict = {}
    label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
    top_prob, top_catid = torch.topk(probabilities, top_k)
    for i in range(top_prob.size(0)):
        top_predict[label_encoder.classes_[top_catid[i]]] = top_catid[i].item()
    return top_predict
