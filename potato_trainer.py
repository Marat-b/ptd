from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances


class PotatoTrainer:
    def __init__(self):
        self.base_lr = base_lr
        self.cfg_path = None
        self.dataset_test = None
        self.dataset_train = None
        self.eval_period = None
        self.max_iter = max_iter
        self.num_classes = num_classes
        self.train_coco_file_path = train_coco_file_path
        self.train_images_path = train_images_path
        self.validate_coco_file_path = validate_coco_file_path
        self.validate_images_path = validate_images_path
        self.weights =


    def _load_cfg(self):
        cfg = get_cfg()  # obtain detectron2's default config
        cfg.merge_from_file(self.cfg_path)  # load values from a file
        cfg.DATASETS.TRAIN = ('train_instances',)
        cfg.DATASETS.TEST = ('validate_instances',)
        cfg.MODEL.WEIGHTS = self.weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.SOLVER.BASE_LR = self.base_lr
        cfg.SOLVER.MAX_ITER = self.max_iter
        # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 200
        # cfg.SOLVER.WARMUP_ITERS = 200
        cfg.TEST.EVAL_PERIOD = self.eval_period

    def _load_datasets(self):
        register_coco_instances('train_instances', {}, self.train_coco_file_path,  self.train_images_path)
        register_coco_instances('validate_instances', {}, self.validate_coco_file_path, self.validate_images_path)

