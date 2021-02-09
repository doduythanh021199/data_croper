import io

import PIL
import tensorflow
import xml.etree.ElementTree as et
import dataset_util
import cv2
import os
import hashlib
import shutil
import numpy as np
from PIL import Image
from pycocotools import mask

folder = "data/"

class_ids = {'1': 'passport'}

if os.path.exists("tfrecord"):
    shutil.rmtree("tfrecord")

os.mkdir("tfrecord")
os.mkdir("tfrecord/train")
os.mkdir("tfrecord/test")
os.mkdir("tfrecord/valid")


def read_xml_make_tfrecord():
    num_data = 8
    for i in range(num_data):
        globals()['train_writer_{:05d}-of-{:05d}'.format(int(i), int(num_data))] = tensorflow.io.TFRecordWriter(
            'tfrecord/train/train.tfrecord-{:05d}-of-{:05d}'.format(
                int(i), int(num_data)))

    for i in range(int(num_data / 8)):
        globals()['test_writer_{:05d}-of-{:05d}'.format(int(i), int(num_data / 8))] = tensorflow.io.TFRecordWriter(
            'tfrecord/test/test.tfrecord-{:05d}-of-{:05d}'.format(
                int(i), int(num_data / 8)))
        globals()[
            'valid_writer_{:05d}-of-{:05d}'.format(int(i), int(num_data / 8))] = tensorflow.io.TFRecordWriter(
            'tfrecord/valid/valid.tfrecord-{:05d}-of-{:05d}'.format(
                int(i), int(num_data / 8)))

    length = len(os.listdir(folder))

    for number, img_name in enumerate(os.listdir(folder)):
        if img_name[-4:] != '.jpg': continue
        filename = img_name[:-4]
        img = cv2.imread(folder + filename + ".jpg")
        height, width = img.shape[:2]

        mask = cv2.imread('mask/' + filename + '.jpg', 0)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        cv2.imshow("asdas",mask)
        cv2.waitKey()
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        contours = sorted(contours, key=lambda x: len(x), reverse=True)

        x = [temp[0][0] for temp in contours[0]]
        y = [temp[0][1] for temp in contours[0]]
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        # cv2.circle(img,(xmin,ymin),5,(255,0,0),5)
        # cv2.circle(img, (xmax, ymax), 5, (255, 0, 0), 5)
        # cv2.imshow("asd",img)
        # cv2.waitKey()
        object_name = 'passport'
        pixel_val = 255
        with tensorflow.io.gfile.GFile(folder + filename + ".jpg", 'rb') as fid:
            encoded_image_data = fid.read()
        key = hashlib.sha256(encoded_image_data).hexdigest()

        with tensorflow.io.gfile.GFile('mask/' + filename + ".jpg", 'rb') as fid:
            encoded_mask_data = fid.read()

        encoded_mask = io.BytesIO(encoded_mask_data)
        mask = Image.open(encoded_mask)
        mask_np = np.asarray(mask.convert('L'))
        mask_remapped = (mask_np == pixel_val).astype(np.uint8)
        # print("mask",mask_remapped.shape)
        # cv2.imshow("asd",mask_remapped*255)
        # cv2.waitKey()
        mask_img = Image.fromarray(mask_remapped)
        output = io.BytesIO()
        mask_img.save(output, format='PNG')

        xmins = [xmin / width]
        xmaxs = [xmax / width]
        ymins = [ymin / height]
        ymaxs = [ymax / height]
        classes_text = [object_name.encode('utf8')]
        classes = [1]
        masks = [output.getvalue()]

        print(img_name)
        print(xmins)
        print(xmaxs)
        print(ymins)
        print(ymaxs)
        print(classes_text)
        print(classes)
        print(masks)
        example = tensorflow.train.Example(features=tensorflow.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(img_name.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(img_name.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/mask': dataset_util.bytes_list_feature(masks),
        }))
        if number < length * 0.8:
            globals()[
                'train_writer_{:05d}-of-{:05d}'.format(int(number / (length * 0.8) * num_data), int(num_data))].write(
                example.SerializeToString())

        elif number < length * 0.9:
            globals()[
                'valid_writer_{:05d}-of-{:05d}'.format(
                    int((number - length * 0.8) / (length * 0.1) * num_data / 8),
                    int(num_data / 8))].write(example.SerializeToString())
        elif number < length:

            globals()[
                'test_writer_{:05d}-of-{:05d}'.format(int((number - length * 0.9) / (length * 0.1) * num_data / 8),
                                                      int(num_data / 8))].write(example.SerializeToString())


read_xml_make_tfrecord()
