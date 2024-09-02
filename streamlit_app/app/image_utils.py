import os
from PIL import Image

def save_image(image, path):
    image.save(path)

def ensure_images_dir_exists(directory="images"):
    if not os.path.exists(directory):
        os.makedirs(directory)
