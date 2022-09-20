

import torchvision  # need to put first
import cv2
import torch
import numpy as np
from tqdm import tqdm

from cv2_imshow import cv2_imshow

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

class TorchscriptDetection:
    def __init__(self, path_inference, use_cuda=True):
        self.device = 'cuda' if use_cuda else 'cpu'
        self.model = torch.jit.load(path_inference)
        self.model.to(self.device)
        self.color_mask = [
            (0, 255, 255),
            (192, 0, 0),
            (0, 192, 0),
            (0, 0, 192),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (192, 192, 0),
        ]
        self.class_names = ['strong', 'alternariosis', 'anthracnose', 'fomosis', 'fusarium', 'internalrot',
                            'necrosis', 'phytophthorosis', 'pinkrot', 'scab', 'wetrot']

    def detect(self, image):
        with torch.no_grad():
            out = self.model(
                torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
            )
        boxes = out[0].numpy()
        classes = out[1].numpy()
        scores = out[3].numpy()
        pr_masks = out[2].numpy()
        bbox, bbox_xcycwh, cls_conf, cls_ids, masks = [], [], [], [], []

        for (box, _class, score, pr_mask) in zip(boxes, classes, scores, pr_masks):
            # print(f'box={box}')
            x0, y0, x1, y1 = box
            # bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
            bbox.append(box)
            cls_conf.append(score)
            cls_ids.append(_class)
            new_mask = self._get_mask(pr_mask, box)
            # print(pr_mask)
            # print(pr_mask.dtype, pr_mask.shape)
            masks.append(new_mask)
            # print(new_mask.dtype, new_mask.shape)

        return np.array(bbox, dtype=np.float64), np.array(cls_conf), np.array(cls_ids), np.array(masks)

    def _get_mask(self, image_mask, box):
        """
        Get real mask from 28x28 mask
        Parameters:
        ----------
            image_mask:  np.array -  image mask (black-white)
        Returns:
        -------
            new_mask: np.array
        """

        width_box = int(box[2]) - int(box[0])
        height_box = int(box[3]) - int(box[1])
        # print(f'width_box={width_box}, height_box={height_box}')
        mask = image_mask[0]  # .astype('uint8')
        mask[mask > 0.9] = 1.0
        mask = mask.astype('uint8')
        new_mask = cv2.resize(mask, (width_box, height_box))
        return new_mask

    def visualize(self, image, confidence=0.1):
        pr_boxes, scores, pr_classes, masks = self.detect(image)
        # print(f'pred_boxes={pred_boxes}')
        # print(f'scores={scores}')
        # print(f'pred_classes={pred_classes}')
        for bbox, pr_score, pr_class, image_mask in zip(pr_boxes, scores, pr_classes, masks):
            if pr_score >= confidence:
                x0, y0, x1, y1 = bbox
                x0 = int(x0)
                y0 = int(y0)
                x1 = int(x1)
                y1 = int(y1)
                z_mask = np.zeros_like(image)
                # print(f'z_mask.shape={z_mask.shape}')
                image_mask_merged = cv2.merge((image_mask, image_mask, image_mask))
                z_mask[y0:y1, x0:x1] = image_mask_merged
                # c02_imshow(z_mask * 205, 'z_mask')
                image2 = cv2.bitwise_and(image, z_mask * 205)
                # cv2_imshow(image2, 'image2')
                z_image = image2 * np.array(self.color_mask[pr_class]).astype('uint8')
                # cv2_imshow(z_image, 'z_image')
                image_weighted = cv2.addWeighted(image2, 0.7, z_image, 0.3, 0.0)
                # cv2_imshow(image_weighted, 'image_weighted')
                mask_inverted = cv2.bitwise_not(z_mask * 255)
                # cv2_imshow(mask_inverted, 'mask_inverted')
                image = cv2.bitwise_or(cv2.bitwise_and(image, mask_inverted), image_weighted)
                # cv2_imshow(new_image[:, :, [2, 1, 0]], 'new_image')
                text = '{} - {:.2f}%'.format(self.class_names[pr_class], pr_score)
                t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(image, (x0, y0), (x1, y1), self.color_mask[pr_class], 3)
                cv2.rectangle(image, (x0, y0), (x0 + t_size[0] + 3, y0 + t_size[1] + 4), self.color_mask[pr_class], -1)
                cv2.putText(
                    image, text, (x0, y0 + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 0], 2
                )
        return image


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Video Detection & Visualize")
    parser.add_argument(
        "-w", "--weight",
        type=str,
        dest="weight_path",
        required=True,
        help="Path of weight file"
    )
    parser.add_argument(
        "-vi", "--video_input",
        type=str,
        dest="video_input_path",
        required=True,
        help="Path of input video file"
    )
    parser.add_argument(
        "-vo", "--video_output",
        type=str,
        dest="video_output_path",
        required=True,
        help="Path of output video file"
    )
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        dest="confidence",
        required=False,
        default=0.5,
        help="Confidence of prediction"
    )

    parser.add_argument(
        "-d", "--device",
        dest="cuda",
        default=True,
        help="Use CUDA (default)"
    )
    p_args = parser.parse_args()
    video_input_path = p_args.video_input_path
    video_output_path = p_args.video_output_path
    tsd = TorchscriptDetection(p_args.weight_path, use_cuda=eval(p_args.cuda))
    cap = cv2.VideoCapture(video_input_path)
    # img = cv2.imread(p_args.image_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer_rgb = cv2.VideoWriter(video_output_path, fourcc, num_frames, (im_width, im_height))

    # cv2_imshow(img, 'img')
    for i in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if ret:
            # img = cv2.resize(img, p_args.shape)
            # frame = frame[:, :, [2, 1, 0]]
            pred_boxes, scores, pred_classes, masks = tsd.detect(frame)
            # print(f'pred_boxes={pred_boxes}')
            # print(f'scores={scores}')
            # print(f'pred_classes={pred_classes}')

            img = tsd.visualize(frame, confidence=p_args.confidence)
            video_writer_rgb.write(img)
        if cv2.waitKey(1) == ord('q'):
            break
    # cv2_imshow(img, 'img')
    video_writer_rgb.release()

    cv2.destroyAllWindows()
