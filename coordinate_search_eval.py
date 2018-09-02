import annotation_util
import csv
import crater_search_voting_relative as cs_rel
import crater_search_voting_real as cs_real

from util import calculate_haversine_distance, approximate_visual_distance, calculate_radius, calculate_center
from retinanet.keras_retinanet.models import load_model
from retinanet.keras_retinanet.utils.image import preprocess_image, resize_image, read_image_bgr
from retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
import cv2
import keras
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from itertools import combinations as combos
from scipy.spatial.distance import cdist
import pandas as pd

# load tensorflow and weights
print('initializing tensorflow session and neural net model...')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)
stay_looped = True
# model = load_model('inf_experimental/inf_resnet152august25.h5', backbone_name='resnet152')
model = load_model('trained_retinanet_models/inf/0830_resnet50_csv_56_withweights_FINALSETTINGS_1248px.h5',
                   backbone_name='resnet50')
graph = tf.get_default_graph()
print('...ready!')

# parameters and definitions
score_threshold = 0.999
max_detections = 500
rectangle_threshold = 1.1
box_minimum = 35
debug_mode = False
ann_path = 'datasets/coordinate_extraction/'
annotations = ann_path + 'via_region_data.csv'
wmin = 1248
wmax = 28

# get annotation data
with annotation_util.open_for_csv(annotations) as file:
    ann_data = annotation_util._read_annotations(csv.reader(file, delimiter=','), True)


def evaluate():
    km_per_pixel = calculate_km_per_pixel()
    for i, (img, anns) in enumerate(ann_data.items()):
        if debug_mode:
            measure(anns, km_per_pixel[img])
            continue
        image = read_image_bgr(ann_path + img)
        image, scale = resize_image(image, min_side=1248, max_side=1248)
        for k, v in km_per_pixel.items():
            km_per_pixel[k] = v / scale
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
            if width > height and height != 0:
                ratio = width / height
                smallest = height
            elif width != 0:
                ratio = height / width
                smallest = width
            # filter out boxes that are too small or "too rectangular"
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
        fig = plt.figure(figsize=(12, 12), dpi=104)
        fig.figimage(draw, xo=0, yo=0)
        plt.show()
        center_lat = anns[0]['img_lat']
        center_lon = anns[0]['img_lon']
        img_width = image.shape[0]
        kmpp = km_per_pixel[img]
        check_results(cs_rel.extract_coordinates(image_boxes, wmax, wmin, 0.02, False), center_lat,
                      center_lon, img_width, kmpp)
        # check_results(cs_real.extract_coordinates(image_boxes, km_per_pixel[img], 0.1, 0.1), center_lat,center_lon, img_width, kmpp)


def check_results(results, center_lat, center_lon, img_width, kmpp):
    for k, v in results.items():
        lat, lon = v['coords']
        radius = v['radius']
        if is_crater_in_image(lat, lon, radius, center_lat, center_lon, img_width, kmpp):
            print('<(^.^)> yay! crater with id', k, ', coordinates: [', lat, lon, '], radius: ', radius, ' is legit!')


def is_crater_in_image(lat, lon, radius, center_lat, center_lon, width, kmpp):
    dist = approximate_visual_distance(calculate_haversine_distance([lat, lon], [center_lat, center_lon]))
    max_dist = ((width * kmpp / 2) * sqrt(2) - radius) * 1.3
    if dist < max_dist: return True
    return False


def calculate_km_per_pixel():
    hdf = pd.HDFStore('crater_database/crater_db.h5', 'r')
    crater_db = hdf.get('/db')
    km_per_pixel = {}
    for img, anns in ann_data.items():
        kmpp_img = 0
        count = 0
        for ann in anns:
            real_radius = approximate_visual_distance(crater_db.loc[crater_db.index[(ann['crater_id'])], 'radius'])
            pixel_radius = calculate_radius([ann['x1'], ann['y1'], ann['x2'], ann['y2']])
            kmpp_img += real_radius / pixel_radius
            count += 1
        if count != 0:
            kmpp_img /= count
        km_per_pixel[img] = kmpp_img
    hdf.close()
    return km_per_pixel


def measure(anns, kmpp):
    hdf = pd.HDFStore('crater_database/crater_db.h5', 'r')
    crater_db = hdf.get('/db')
    g = len(anns)
    tuples = combos(np.arange(g), 2)
    u = 0
    v = 0
    diff_dist = 0
    diff_rad = 0
    for a in range(g):
        box_a = [anns[a]['x1'], anns[a]['y1'], anns[a]['x2'], anns[a]['y2']]
        pixel_derived_radius = calculate_radius(box_a) * kmpp
        real_radius = crater_db.loc[crater_db.index[(anns[a]['crater_id'])], 'radius']
        d = real_radius / pixel_derived_radius
        diff_rad += d
        print('RADIUS DIFF: ', d)
        v += 1
    diff_rad /= v
    print('########################### RADIUS AVG DIFF: ', diff_rad)
    for a, b in tuples:
        u += 1
        print('#####################')
        print('distances between ', anns[a]['crater_id'], ' with ', anns[a]['lat'], anns[a]['lon'], 'and',
              anns[b]['crater_id'], ' with ', anns[b]['lat'], anns[b]['lon'])
        hav = calculate_haversine_distance([anns[a]['lat'], anns[a]['lon']], [anns[b]['lat'], anns[b]['lon']])
        appr_real = approximate_visual_distance(hav)
        print('haversine: ', hav)
        print('approx: ', appr_real)
        box_a = [anns[a]['x1'], anns[a]['y1'], anns[a]['x2'], anns[a]['y2']]
        box_b = [anns[b]['x1'], anns[b]['y1'], anns[b]['x2'], anns[b]['y2']]
        approx_dist_pixels = \
            cdist(np.array([calculate_center(box_a)]), np.array([calculate_center(box_b)]))[0][0]
        appr_img = approx_dist_pixels * kmpp
        print('approx pixel dist from bounding box: ', approx_dist_pixels, 'approx km dist from bounding box: ',
              appr_img)
        print('difference factor: ', appr_img / appr_real)
        diff_dist += appr_img / appr_real
    print('#####################', 'DIFF_AVG:', diff_dist / u, '####################')
    hdf.close()


# all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
#                                  save_path=save_path)
# all_annotations = _get_annotations(generator)

# get detections and find find overlaps with annotations by calculating iou

# take best overlap for each annotation

# compare crater candidate found for the respective detection with ground truth data in annotations
evaluate()
