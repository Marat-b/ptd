import torchvision  # need to put first
import cv2
import torch
import numpy as np

from cv2_imshow import cv2_imshow


class TorchscriptDetection:
    def __init__(self, path_inference, use_cuda=True):
        self.device = 'cuda' if use_cuda else 'cpu'
        self.model = torch.jit.load(path_inference)
        self.model.to(self.device)

    def detect(self, image):
        with torch.no_grad():
            out = self.model(
                     torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
            )
        boxes = out[0].numpy()
        classes = out[1].numpy()
        scores = out[3].numpy()
        # pr_masks = self._do_mask(out, image)
        pr_masks = out[2].numpy()
        # pr_masks = out.pred_masks.cpu().numpy()
        # pr_masks = pr_masks.astype(np.uint8)
        # pr_masks[pr_masks > 0] = 255
        bbox_xcycwh, cls_conf, cls_ids, masks = [], [], [], []

        for (box, _class, score, pr_mask) in zip(boxes, classes, scores, pr_masks):
            # print(box)
            x0, y0, x1, y1 = box
            bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
            cls_conf.append(score)
            cls_ids.append(_class)
            # masks.append(pr_mask)
            new_mask = self._get_mask(pr_mask, box)
            # print(pr_mask)
            print(pr_mask.dtype, pr_mask.shape)
            # new_mask = pr_mask.transpose((1, 2, 0))
            # new_mask[new_mask > 0.9] = 1.0
            # new_mask = new_mask.astype('uint8') * 255
            # new_mask = cv2.resize(new_mask, (length, length))
            masks.append(new_mask)
            print(new_mask.dtype, new_mask.shape)
            # cv2.imshow('new_mask', new_mask)
            # cv2.waitKey(0)

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids), np.array(masks)

    # def _do_mask(self, out, image):
    #     N = len(out[2])
    #     chunks = torch.chunk(torch.arange(N, device='cpu'), N)
    #     img_masks = torch.zeros(
    #         N, image.shape[0], image.shape[1], device='cpu', dtype=torch.bool
    #     )
    #     # masks = out.pred_masks[:, 0, :, :]  # shape was [#instances, 1, 28, 28] --> [#instances, 28, 28]
    #     for inds in chunks:
    #         masks_chunk, spatial_inds = _do_paste_mask(
    #             out[2][inds, :, :, :], out[0].tensor[inds], image.shape[0], image.shape[1],
    #             skip_empty=True
    #         )
    #
    #         masks_chunk = (masks_chunk >= 0.5).to(dtype=torch.bool)
    #         img_masks[(inds,) + spatial_inds] = masks_chunk
    #         # print(img_masks.shape)
    #     # print(np.array(np.uint8(img_masks)*255))
    #     return np.array(np.uint8(img_masks)*255)
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
        def calculate_distance(point_a, point_b):
            dist = np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
            return dist
        width_box = box[2] - box[0]
        height_box = box[3] - box[1]
        print(f'width_box={width_box}, height_box={height_box}')
        # box_len = width if width > height else height
        mask = image_mask[0]  # .astype('uint8')
        mask[mask > 0.9] = 1.0
        mask = mask.astype('uint8')
        # f = np.argwhere(mask > 0)
        # f_max = np.argmax(f, axis=0)
        # bottom = f[f_max[0]]
        # right = f[f_max[1]]
        # f_min = np.argmin(f, axis=0)
        # top = np.array([f[f_min[0]][0], bottom[1]])
        # left = np.array([right[0], f[f_min[1]][1]])
        # width = calculate_distance(left, right)
        # height = calculate_distance(top, bottom)
        # length = width if width > height else height
        # return int((box_len * length) / 28)
        new_mask = cv2.resize(mask * 255,(int(width_box), int(height_box)))
        return new_mask

    def visualize(self, image, mask, bbox,):
        # bbox = cxcywh
        pass

if __name__ == '__main__':
    tsd = TorchscriptDetection('../weights/potato_best20220715.ts', use_cuda=False)
    img = cv2.imread('../images/300000000.jpg')
    cv2_imshow(img, 'img')
    pred_boxes, scores, pred_classes, masks = tsd.detect(img)
    print(f'pred_boxes={pred_boxes}')
    print(f'scores={scores}')
    print(f'pred_classes={pred_classes}')
    for mask in masks:
        print(f'mask.shape={mask.shape}')
        cv2_imshow(mask, 'mask')
        cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
    # if len(pred_masks) > 0:
    #     print(f'pred_masks={pred_masks[0].shape}')
    # im = tsd.detect2(img)
    # print(im)
    # cv2.imshow('im', im)
    # cv2.waitKey(1000)
