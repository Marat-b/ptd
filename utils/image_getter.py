import cv2
import numpy as np

from utils.ts_detection import TorchscriptDetection


def main(path_inference, image_path):
    td = TorchscriptDetection(path_inference, use_cuda=False)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    pr_boxes, scores, pr_classes, masks = td.detect(image)
    for bbox, pr_score, pr_class, image_mask in zip(pr_boxes, scores, pr_classes, masks):
        bbox = [int(x) for x in bbox]
        x0, y0, x1, y1 = bbox
        width_box = x1 - x0
        height_box = y1 - y0
        length = width_box if width_box > height_box else height_box
        delta_height = height_box / length
        delta_width = width_box / length
        roi = image[y0:y1, x0:x1]
        z_mask = np.zeros_like(roi)
        merged_mask = np.stack((image_mask, image_mask, image_mask), axis=2)
        z_mask[y0:y1, x0:x1] = merged_mask
        image2 = cv2.bitwise_and(roi, z_mask * 255)
        new_image = cv2.bitwise_or(cv2.bitwise_not(merged_mask), image2)