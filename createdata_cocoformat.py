from PIL import Image
import cv2
import numpy as np
import random
import glob
import shutil
import os
import rstr
import string
import json

anno = {"info": {"description": "Passport Dataset", "url": "http://cocodataset.org", "version": "1.0", "year": 2017,
                 "contributor": "COCO Consortium", "date_created": "2017/09/01"},
        "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                      "name": "Attribution-NonCommercial-ShareAlike License"},
                     {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
                      "name": "Attribution-NonCommercial License"},
                     ],
        "images": [],
        "annotations": [],
        "categories": [{"supercategory": "idcard", "id": 1, "name": "passport"}, ]
        }


def rotate_image(img):
    """
    :param img: np.ndarray
    :return:
    """
    angle = random.randint(-20, 20)
    if angle < 0: angle = 360 + angle
    img = Image.fromarray(img)
    img = img.rotate(angle=angle, expand=True)
    return np.asarray(img)


def image_on_background(img):
    """

    :param img: ndarray
    :return: result: ndarray, img on background
            hull: convexhull
    """
    h_img, w_img = img.shape[:2]
    ratio = random.randint(130, 150) / 100
    h = int(h_img * ratio)
    w = int(w_img * ratio)

    result = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    x_draw = random.randint(0, w - w_img)
    y_draw = random.randint(0, h - h_img)
    result[y_draw:y_draw + h_img, x_draw:x_draw + w_img, :] = img

    mask[cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) != 0] = 255
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    background = cv2.imread(random.choice(glob.glob("background/*.jpg")))
    background = cv2.resize(background, (w, h))
    background[mask != 0] = 0
    result = result + background
    return result, mask, hull

anno_images = {"license": 1, "file_name": "", "coco_url": "", "height": 640, "width": 640,
               "date_captured": "2013-11-18 05:20:56", "flickr_url": "", "id": 233771}

anno_annotations = {"segmentation": [], "area": 0, "iscrowd": 0, "image_id": 554291, "bbox": [], "category_id": 17,
                    "id": 49229}


def create_data(folder_name,count):
    anno["images"] = []
    anno["annotations"] = []

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

    for i in range(count):
        img=cv2.imread(random.choice(glob.glob('real_passport/*.jpg')))
        rotated=rotate_image(img)
        passport,mask,hull=image_on_background(rotated)
        h,w=passport.shape[:2]
        new_img_name = "{:05d}".format(i) + ".jpg"

        anno_images_copy = anno_images.copy()
        anno_annotations_copy = anno_annotations.copy()

        anno_images_copy["file_name"] = new_img_name
        anno_images_copy["height"] = h
        anno_images_copy["width"] = w
        anno_images_copy["id"] = i
        anno_annotations_copy["segmentation"] = [hull.flatten().tolist()]
        anno_annotations_copy["area"] = cv2.contourArea(hull)
        anno_annotations_copy["image_id"] = i
        anno_annotations_copy["bbox"] = cv2.boundingRect(hull)
        # [top left x position, top left y position, width, height]
        anno_annotations_copy["category_id"] = 1  # passport
        anno_annotations_copy["id"] = i

        anno["images"].append(anno_images_copy)
        anno["annotations"].append(anno_annotations_copy)
        print(new_img_name)
        cv2.imwrite(folder_name + "/" + new_img_name, passport)

    with open(folder_name + ".json", 'w') as fp:
        json.dump(anno, fp)


create_data('train',80)
create_data('test',10)
create_data('valid',10)