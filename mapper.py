import torch
import copy
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T


class MyMapper:
    def __init__(self, cfg, is_train: bool = True):
        if is_train:
            aug_list = [
                # T.ResizeShortestEdge([800, 800], sample_style='range'),
                T.RandomBrightness(0.8, 1.2),
                # T.RandomRotation([0.5, 1]),
                T.RandomContrast(0.8, 1.2),
                T.RandomSaturation(0.8, 1.2),
                # T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
            ]
        else:
            aug_list = [T.ResizeShortestEdge(800, sample_style='choice')]

        self.augmentations = T.AugmentationList(aug_list)
        self.is_train = is_train
        mode = "training" if is_train else "inference"
        print(f"[MyDatasetMapper] Augmentations used in {mode}: {self.augmentations}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict