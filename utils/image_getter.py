import cv2
import numpy as np

from utils.cv2_imshow import cv2_imshow
from utils.ts_detection import TorchscriptDetection


def main(path_inference, image_path):
    td = TorchscriptDetection(path_inference, use_cuda=False)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (1024, 1024))
    cv2_imshow(image, 'image')
    pr_boxes, scores, pr_classes, masks = td.detect(image)
    for i, (bbox, pr_score, pr_class, image_mask) in enumerate(zip(pr_boxes, scores, pr_classes, masks)):
        bbox = [int(x) for x in bbox]
        x0, y0, x1, y1 = bbox
        width_box = x1 - x0
        height_box = y1 - y0
        print(f'width_box={ width_box}, height_box={height_box}')
        length = width_box if width_box > height_box else height_box
        delta_height = height_box / length
        delta_width = width_box / length
        roi = image[y0:y1, x0:x1]
        print(f'roi.shape={roi.shape}, image_mask.shape={image_mask.shape}')
        # cv2_imshow(roi, 'roi')
        z_mask = np.zeros_like(roi)
        # cv2_imshow(image_mask, 'image_mask')
        merged_mask = np.stack((image_mask, image_mask, image_mask), axis=2) * 255
        # merged_mask = cv2.merge((image_mask, image_mask, image_mask)) * 255
        # cv2_imshow(merged_mask, 'merged_mask')

        image2 = cv2.bitwise_and(roi, merged_mask)
        # cv2_imshow(image2, 'image2')
        new_image = cv2.bitwise_or(cv2.bitwise_not(merged_mask), image2)
        cv2_imshow(new_image, '{}. new_image'.format(i))

if __name__ == '__main__':
    main('../weights/potato_best2022082419_1024_11cl.ts',
         r'C:\softz\work\potato\in\images\20220814_151921.jpg')