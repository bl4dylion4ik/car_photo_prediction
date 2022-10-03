import os
import io
import pickle

from PIL import Image
import torch
from torchvision import transforms

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CLASS_NUM = 567
MODEL = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
MODEL.classifier.fc = torch.nn.Linear(in_features=MODEL.classifier.fc.in_features, out_features=CLASS_NUM)
model_path = os.path.join(os.getcwd(), 'app', 'FN-EfficientNet_b18.tar')
checkpoint = torch.load(model_path, map_location=DEVICE)
MODEL.load_state_dict(checkpoint['model_state_dict'])
MODEL.to(DEVICE)


def process_img(image_bytes: bytes):
    """Process image to tensor"""
    input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def predict_one_sample(model, inputs, device=DEVICE) -> torch.Tensor:
    """Predict for one sample"""
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit[0], dim=0)
    return probs


def predict_top_labels(model, image_bytes, top_k, device=DEVICE) -> dict:
    """Predict top k most probable labels"""
    input_img = process_img(image_bytes)
    probabilities = predict_one_sample(model, input_img[0].unsqueeze(0), device=device)

    top_predict = {}
    encoder_path = os.path.join(os.getcwd(), 'app', 'label_encoder.pkl')
    label_encoder = pickle.load(open(encoder_path, 'rb'))
    top_prob, top_catid = torch.topk(probabilities, top_k)
    for i in range(top_prob.size(0)):
        top_predict[label_encoder.classes_[top_catid[i]]] = round(top_prob[i].item(), 3)
    return {'prediction': top_predict}
