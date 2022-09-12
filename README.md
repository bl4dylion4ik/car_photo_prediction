## Car Model Classification by photos

### Introdusing: Transfer Learning and Prediction model

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
<img src="img_for_readme/47622533.jpg">
<img src="img_for_readme/57493534.jpg">
<img src="img_for_readme/65405577.jpg" width="270">
</div>

#### Filtering Unwanted Images Out of the Dataset

There are multiple possible approaches to filter non-car images out of the dataset:

- Process images manually
- Train another model to classify car/no-car
- Use a pretrained model
