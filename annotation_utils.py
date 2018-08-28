import csv
import sys
from shutil import copy2, move
from six import raise_from
import glob
from sklearn.model_selection import train_test_split
import random
from os import rename, getcwd


def combine_and_split_datasets(folder_list, destination_main, destination_test):
    """ Copies *png files from folder_list in to destination and outputs combined annotations dictionary"""
    annotations = {}
    filepaths = []
    for folder in folder_list:
        with open_for_csv(folder + '/via_region_data.csv') as csvfile:
            annotations = {**annotations, **_read_annotations(csv.reader(csvfile, delimiter=','))}
        filepaths += glob.glob(folder + '/*.png', recursive=True)
    for file in filepaths:
        if file.rpartition('\\')[2] in annotations: copy2(file, destination_main)
    filepaths = glob.glob(destination_main + '/*.png', recursive=True)
    random.shuffle(filepaths)
    train_filepaths, test_filepaths = train_test_split(filepaths, train_size=.8)
    train_annotations, test_annotations = {}, {}
    for i, file in enumerate(test_filepaths):
        file_name = file.rpartition('\\')[2]
        new_file_name = str(i).zfill(4) + '.PNG'
        test_annotations[new_file_name] = annotations.pop(file_name)
        move(file, destination_test + '/' + new_file_name)
        with open(destination_test + '/' + 'annotations.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            write_annotations(writer, test_annotations)
    for i, file in enumerate(train_filepaths):
        file_name = file.rpartition('\\')[2]
        new_file_name = str(i).zfill(4) + '.PNG'
        train_annotations[new_file_name] = annotations.pop(file_name)
        rename(file, destination_main + '/' + new_file_name)
        with open(destination_main + '/' + 'annotations.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            write_annotations(writer, train_annotations)


def write_annotations(writer, image_data):
    writer.writerow(['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes',
                     'region_attributes'])
    for image, boxes in image_data.items():
        region_count = len(boxes)
        for region_id, box in enumerate(boxes):
            x1 = int(box.get('x1'))
            y1 = int(box.get('y1'))
            x2 = int(box.get('x2'))
            y2 = int(box.get('y2'))
            line = '{"name":"rect","x":' + str(x1)
            line += ',"y":' + str(y1)
            line += ',"width":' + str(x2 - x1)
            line += ',"height":' + str(y2 - y1)
            line += '}'
            writer.writerow([image, '', '{}', region_count, region_id, line, '{}'])


def convert_annotations_to_yolo(csv_data_file='annotations.csv'):
    """ Converts annotations made with VIA-Tool into format required by YOLOv3"""
    try:
        with open_for_csv(csv_data_file) as file:
            data = _read_annotations(csv.reader(file, delimiter=','), False)
            with open("annotations.txt", "w") as annotations:
                for image_file_name, image_boxes in data.items():
                    line_to_write = 'data/train/' + image_file_name
                    for bounding_box in image_boxes:
                        x1, x2, y1, y2, class_id = bounding_box.values()
                        line_to_write += ' ' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(
                            class_id)
                    print(f'{line_to_write}', file=annotations)

    except ValueError as e:
        raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)


def _limit(v, v_min, v_max):
    return min(max(v, v_min), v_max)


# large parts of the code below are copied (with some custom modifications)
# from: https://github.com/fizyr/keras-retinanet (Apache License 2.0)

def parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def _read_annotations(csv_reader, has_real_world_coordinates):
    """ Read annotations from the csv_reader. """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1
        img_file, file_size, file_attributes, region_count, region_id, region_shape_attributes, region_attributes = row[
                                                                                                                    :7]
        if img_file == 'filename': continue
        region_shape_attributes_no_brackets = region_shape_attributes.replace("{", "").replace("}", "")
        (x1, y1, width, height) = ('', '', '', '')
        if not region_shape_attributes_no_brackets == '':
            shape_arr = region_shape_attributes_no_brackets.split(",")
            x1 = shape_arr[1].rpartition(":")[2]
            y1 = shape_arr[2].rpartition(":")[2]
            width = shape_arr[3].rpartition(":")[2]
            height = shape_arr[4].rpartition(":")[2]
        if img_file not in result:
            result[img_file] = []
        # class_name = 'crater'
        class_id = 0
        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, width, height) == ('', '', '', ''):
            continue

        x1 = parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        width = parse(width, int, 'line {}: malformed x2: {{}}'.format(line))
        height = parse(height, int, 'line {}: malformed y2: {{}}'.format(line))

        # Don't add boxes where longest side is less than 7 pixels or the shortest less than 4
        if max(width, height) < 7: continue
        if min(width, height) < 4: continue

        x2 = x1 + width
        y2 = y1 + height

        # Cut off boxes at image borders
        # x1 = _limit(x1, 0, 415)
        # x2 = _limit(x2, 0, 415)
        # y1 = _limit(y1, 0, 415)
        # y2 = _limit(y2, 0, 415)

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        # if class_name not in classes:
        #     raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        if not has_real_world_coordinates:
            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_id})
        else:
            region_attributes_no_brackets = region_attributes.replace("{", "").replace("}", "").replace('"','')
            if not region_attributes_no_brackets == '':
                data_arr = region_attributes_no_brackets.split(",")
                crater_id = data_arr[0].rpartition(":")[2]
                lat = data_arr[1].rpartition(":")[2]
                lon = data_arr[2].rpartition(":")[2]
                crater_id = parse(crater_id, int, 'line {}: malformed id: {{}}'.format(line))
                lat = parse(lat, float, 'line {}: malformed lat: {{}}'.format(line))
                lon = parse(lon, float, 'line {}: malformed lon: {{}}'.format(line))
                result[img_file].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_id, 'crater_id': crater_id, 'lat': lat, 'lon': lon})
    return result


# dest_main = getcwd() + '/dataset/train'
# dest_test = getcwd() + '/dataset/validate'
# folder_list = [getcwd() + '/resources/training/3',
#                getcwd() + '/resources/training/5',
#                getcwd() + '/resources/training/9',
#                getcwd() + '/resources/training_new/3',
#                # getcwd() + '/resources/training_new/5',
#                # getcwd() + '/resources/training_new/9',
#                getcwd() + '/resources/validation/3',
#                getcwd() + '/resources/validation/5',
#                getcwd() + '/resources/validation/9',
#                getcwd() + '/resources/training_north_pole/1',
#                getcwd() + '/resources/training_north_pole/3',
#                getcwd() + '/resources/training_south_pole/1',
#                getcwd() + '/resources/training_south_pole/3']
# combine_and_split_datasets(folder_list, dest_main, dest_test)
