import os
import json
import numpy as np


def create_labels():
    path = os.path.join('datav5', 'annotations', '7_12_14_15_16_18_10.json')
    with open(path, 'r') as ff:
        data = json.loads(ff.read())
    images = data['images']
    annotations = data['annotations']
    print(len(images))
    for image in images:
        id = image['id']
        file_name = image['file_name']
        # file_path = os.path.join('data', 'images', '3', file_name)
        width = image["width"]
        height = image["height"]
        file_name = file_name.split('.')[0]
        bbox = np.array([ann['bbox'] for ann in annotations if ann['image_id'] == id])
        category_id = np.array([ann['category_id'] for ann in annotations if ann['image_id'] == id])

        # category_id=np.expand_dims(category_id, axis=1)
        target = np.array([[int(t - 1),
                            (x + w / 2) / width,
                            (y + h / 2) / height,
                            w / width,
                            h / height]
                           for (x, y, w, h), t in zip(bbox, category_id)]).astype(np.str)
        path_to_annotation = os.path.join('datav5', 'annotations', '7_12_14_15_16_18_10', file_name + '.txt')
        with open(path_to_annotation, 'w') as f:
            for box in target:
                f.write(' '.join(box))
                f.write('\n')
    # target = np.stack([bbox, category_id], axis=0)


# create_labels()

import shutil


def copy_images():
    folder_with_ann = '7_12_14_15_16_18_10'
    folders = ['7', '12', '14', '15', '16', '18', '10']
    path_images = os.path.join('..', '..', 'dataset', 'DONE_FOOTNOTE_DATASET')
    labels = [name.split('.')[0] + '.png' for name in
              os.listdir(os.path.join('datav5', 'annotations', folder_with_ann))]
    for folder in folders:
        images_names = os.listdir(os.path.join(path_images, folder, 'images'))
        for name in images_names:
            if name in labels:
                impath_from = os.path.join(path_images, folder, 'images', name)
                impath_to = os.path.join('datav5', 'data', 'zumen', folder_with_ann, 'images', name)
                shutil.copyfile(impath_from, impath_to)


# copy_images()
from numpy.random import shuffle


def train_val_data():
    labels = os.listdir(os.path.join('datav5', 'data', 'zumen', 'all', 'labels'))
    shuffle(labels)
    images = [name.split('.')[0] + '.png' for name in labels]
    n = int(0.8 * len(labels))
    images_train = images[:n]
    labels_train = labels[:n]
    images_test = images[n:]
    labels_test = images[n:]

    def coppy(files, folder_from, folder_to):
        folder_from = os.path.join('datav5', 'data', 'zumen', 'all', folder_from)
        folder_to = os.path.join('datav5', 'data', 'zumen', 'all', folder_to)
        for name in files:
            impath_from = os.path.join(folder_from, name)
            impath_to = os.path.join(folder_to, name)
            shutil.copyfile(impath_from, impath_to)

    coppy(images_train,
          os.path.join('images'),
          os.path.join('train', 'images'),
          )
    coppy(labels_train,
          os.path.join('labels'),
          os.path.join('train', 'labels'),
          )
    coppy(images_test,
          os.path.join('images'),
          os.path.join('test', 'images'),
          )
    coppy(labels_test,
          os.path.join('images'),
          os.path.join('test', 'labels'),
          )


train_val_data()
