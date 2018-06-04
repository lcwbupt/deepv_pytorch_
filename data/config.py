# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

deepv = {
    'num_classes': 5,
    'lr_steps': (100000, 150000, 180000),
    'max_iter': 180000,
    'feature_maps': [[144,81], [72, 41], [36, 21], [18, 11], [9, 6], [5, 3], [3, 1]],
    'image_height': 648,
    'image_width': 1152,
    'min_dim': 648, 
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [50, 90, 130, 190, 250, [330, 410], 500],
    'max_sizes': [90, 130, 190, 250, 330, [410, 500], 700],
    'aspect_ratios': [[3], [2, 3], [2, 3], [2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'DEEPV',
}
