import annotation_utils
import csv

# get annotation data
annotations = 'datasets/coordinate_extraction/via_region_data.csv'
with annotation_utils.open_for_csv(annotations) as file:
    data = annotation_utils._read_annotations(csv.reader(file, delimiter=','), True)

print('bla')

#
# all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
#                                  save_path=save_path)
# all_annotations = _get_annotations(generator)

# get detections and find find overlaps with annotations by calculating iou

# take best overlap for each annotation

# compare crater candidate found for the respective detection with ground truth data in annotations
