## Car Model Classification by photos

### Introdusing: Transfer Learning and Prediction model
TBD
### Classifying Car Models

#### Data Preparation
I used dataset which i made by myself, using site av.by for extract and load image. You can also download [entire dataset](https://github.com/bl4dylion4ik/car_photo_prediction). Dataset contain about 400.000 not processed img.
Also, if you want to download [processed dataset](https://github.com/bl4dylion4ik/car_photo_prediction), with most popular brand, you can download it too.
Original dataset contain folder with brand and modelname, like:
```project
│
└───Audi
│   │   47623453.jpg
│   │   43244243.jpg
│   |   ...
|
└───BMW
|   │   23425253.jpg
|   │   76543523.jpg
|   |   ...
|
|   ...
```
Three sample images are shown below.
<div>
<img src="img_for_readme/1517175.jpg" width="270" height="210">
<img src="img_for_readme/30959382.jpg" width="270" height="210">
<img src="img_for_readme/41840134.jpg" width="270" height="210">
</div>


While checking through the data, i observed that the dataset contained many unwanted images, e.g., pictures of wing mirrors, door handles, GPS panels, or lights.

Examples of unwanted images can be seen below.

<div>
<img src="img_for_readme/47622533.jpg" width="270">
<img src="img_for_readme/57493534.jpg">
<img src="img_for_readme/65405577.jpg" width="270">
</div>

#### Filtering Unwanted Images Out of the Dataset

There are multiple possible approaches to filter non-car images out of the dataset:

- Process images manually
- Train another model to classify car/no-car
- Use a pretrained model

I decided to use a pretrained model since it is the most direct one, easy and outstanding pre trained models are easily available. I choose [MOBILENET V2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) in pytorch framework with the pretrained "imagenet" weights.

In a first step, i figure out the indices and classnames of the imagenet labels corresponding to car images.
```python
CAR_IDX = [656, 627, 817, 511, 468, 751, 705, 757, 717, 734, 654, 675, 864, 609, 436, 581]
```
Next i load pretrained model. Then i load images and preprocess them. 

```python
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
```

After prediction i filter unwanted image by using probability of prediction and threshold value.

```python
def is_car_acc_prob(predictions: torch.tensor, thresh: float):
    prob = np.array(torch.nn.functional.softmax(predictions), dtype=float)
    car_probs = prob[:, CAR_IDX]
    car_probs_acc = car_probs.sum(axis=1)
    return car_probs_acc > thresh
```

While tuning the prefiltering procedure, i observed the following:

- Many of the car images model classified as “beach wagons”. i thus decided to also consider the “beach wagon” index class in imagenet as one of the CAR_IDX.
- Images showing the front of a car are often assigned a high probability of “grille”, which is the grating at the front of a car used for cooling, so i decided to use “grille” index in CAR_IDX too.


#### Overview of the Final Datasets
TBD
