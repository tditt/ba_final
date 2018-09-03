import sys
import time

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pyscreenshot
import tensorflow as tf
from system_hotkey import SystemHotkey

from retinanet.keras_retinanet.models import load_model
from retinanet.keras_retinanet.utils.image import preprocess_image
from retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption


def _get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


print('initializing tensorflow session and neural net model...')
sess = _get_session()
keras.backend.tensorflow_backend.set_session(sess)
stay_looped = True
# model = load_model('final_models/inf_mobilenet.h5', backbone_name='mobilenet224_1.0')
model = load_model('trained_retinanet_models/inf/0826_resnet101_csv_62_withweights_FINALSETTINGS.h5', backbone_name='resnet101')
graph = tf.get_default_graph()
print('...ready!')


def main():
    global stay_looped
    hk = SystemHotkey()
    hk.register(('control', 'f8'), callback=lambda x: _capture_and_detect())
    hk.register(('control', 'f12'), callback=lambda y: _stop_loop())
    while stay_looped:
        time.sleep(1)
    sys.exit('exiting...')


def _stop_loop():
    global stay_looped
    stay_looped = False


def _capture_and_detect():
    global model
    global sess
    global graph
    # parameters
    score_threshold = 0.999
    max_detections = 500
    rectangle_threshold = 1.1
    box_minimum = 16
    cropped_image_pixels = 816
    max_radius_multiplier = 20
    # box_minimum = 1.05 * 2 * cropped_image_pixels / max_radius_multiplier


    print('capturing screen...')
    screenshot = pyscreenshot.grab(bbox=None, childprocess=None, backend=None)
    image = np.asarray(screenshot.convert('RGB'))
    screenshot.close()
    # crop the image (change values for different screen resolution than 1920x1080
    image = image[131:947, 551:1367]
    # image, scale = resize_image(image, min_side=816, max_side=816)
    image = image[:, :, ::-1].copy()
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    print('detecting craters...')
    start = time.time()
    with graph.as_default():
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("detection processing time: ", time.time() - start)

    valid_box_indices = []
    for b, box in enumerate(boxes[0]):
        width = box[2] - box[0]
        height = box[3] - box[1]
        ratio = 0
        smallest = 0
        if width > height:
            ratio = width / height
            smallest = height
        else:
            ratio = height / width
            smallest = width
        # filter out boxes that are too small or too rectangular
        if smallest < box_minimum or ratio > rectangle_threshold:
            continue
        valid_box_indices.append(b)

    valid_boxes = boxes[0][valid_box_indices]
    valid_scores = scores[0][valid_box_indices]
    valid_labels = labels[0][valid_box_indices]

    # correct boxes for image scale
    # valid_boxes /= scale

    # select indices which have a score above the threshold
    indices = np.where(valid_scores > score_threshold)[0]

    # select those scores
    valid_scores = valid_scores[indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-valid_scores)[:max_detections]

    # select detections
    image_boxes = valid_boxes[indices[scores_sort]]
    image_scores = valid_scores[scores_sort]
    image_labels = valid_labels[indices[scores_sort]]
    # if save_path is not None:
    #     draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
    #     draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)
    #
    #     cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
    #
    # # copy detections to all_detections
    # for label in range(generator.num_classes()):
    #     all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
    crater_count = 0
    for box, score, label in zip(image_boxes, image_scores, image_labels):
        crater_count += 1
        b = box.astype(int)
        draw_box(draw, b, color=(255, 215, 0))
        caption = "{:.3f}".format(score)
        draw_caption(draw, b, caption)
    print(str(crater_count), ' craters detected!')
    fig = plt.figure(figsize=(8, 8), dpi=102)
    fig.figimage(draw, xo=0, yo=0)
    plt.show()

if __name__ == '__main__':
    main()
