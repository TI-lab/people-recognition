import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import datetime
import copy

from io import StringIO
from PIL import Image
from collections import defaultdict
from matplotlib import pyplot as plt

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2
cap = cv2.VideoCapture(0)

# This is needed since the notebook is stored in the object_detection folder.

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph.
# This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)

for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True
)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)
    ).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'test_images'

# PEP8 compliant, but super ugly :')
TEST_IMAGE_PATHS = [os.path.join(
                    PATH_TO_TEST_IMAGES_DIR,
                    'image{}.jpg'.format(i)
                    )
                    for i in range(1, 3)
                    ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def photo_crop(image, boxes, i):
    height, width, channels = image.shape
    x1 = int(boxes[0][i][1] * width)
    x2 = int(boxes[0][i][3] * width)

    y1 = int(boxes[0][i][0] * height)
    y2 = int(boxes[0][i][2] * height)

    return photo_crop_rect(image, x1, y1, x2, y2)


def photo_crop_rect(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Grab image
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Grab tensors
            image_tensor = detection_graph.get_tensor_by_name(
                'image_tensor:0'
            )
            boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0'
            )
            scores = detection_graph.get_tensor_by_name(
                'detection_scores:0'
            )
            classes = detection_graph.get_tensor_by_name(
                'detection_classes:0'
            )
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0'
            )

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )
            # Visualization of the results of a detection.
            image_copy = copy.copy(image_np)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8
            )

            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 1:
                    if scores[0][i] >= 0.8:
                        time_str = str(datetime.datetime.utcnow())
                        cv2.imwrite(
                            "/media/hu/pics/" + time_str + '.png',
                            photo_crop(
                                image_copy,
                                boxes,
                                i
                            )
                        )

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
