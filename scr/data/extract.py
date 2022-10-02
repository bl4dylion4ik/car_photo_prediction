import json
import os

import requests
from PIL import Image
from PIL import UnidentifiedImageError


def extract_and_load_photos(data_dir: str, photo_dir: str):
    """
    Function scan input data dir, extract and load file from url in json in data dir
    :param data_dir: directory where located jsons about car in format ../Audi/A4
    :param photo_dir: directory where you want to save photos from advertisement
    :return:
    """
    for brand in os.listdir(data_dir):
        brand_dir = os.path.join(data_dir, brand)
        for data_id in os.listdir(brand_dir):
            with open(os.path.join(brand_dir, data_id), 'r') as f:
                data_json = json.load(f)
            model = str(data_json['properties'][1]['value'])
            brand_model_photo = os.path.join(os.path.join(photo_dir, brand), model)
            try:
                os.mkdir(brand_model_photo)
            except FileExistsError:
                pass

            for img_link in data_json['photos']:
                try:
                    img_data = requests.get(img_link['small']['url']).content
                    with open(f'{(os.path.join(brand_model_photo, str(img_link["id"])))}.jpg', 'wb') as handler:
                        handler.write(img_data)
                except:
                    continue


def check_and_save_valid_photos(photo_dir: str):
    """
    Function scan all images from directory and delete defective
    :param photo_dir: directory where you located photos
    :return:
    """
    for brand in os.listdir(photo_dir):
        brand_file = os.path.join(photo_dir, brand)
        for model in os.listdir(brand_file):
            model_file = os.path.join(brand_file, model)
            for img in os.listdir(model_file):
                img_file = os.path.join(model_file, img)

                try:
                    im = Image.open(img_file)
                except UnidentifiedImageError:
                    os.remove(img_file)


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'data')
    photos = os.path.join(os.getcwd(), 'av-photos')

    extract_and_load_photos(data_dir, photos)
    check_and_save_valid_photos(photos)
