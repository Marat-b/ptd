import datetime
import os
import shutil
import sys

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import PeriodicCheckpointer, default_writers
from detectron2.evaluation import inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.utils import comm
from detectron2.utils.events import EventStorage
from detectron2.export import TracingAdapter
from tools.plain_train_net import get_evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from madgrad import MADGRAD

from mapper import MyMapper
from utils.custom_filter import CustomFilter


class PotatoTrainer:
    def __init__(self
                 ):
        # self.base_lr = base_lr
        self.cfg = None
        self.cfg_path = './config_potato.yml'
        self.count_iteration = 2000
        # self.dataset_test = None
        # self.dataset_train = None
        self.eval_period = 20000
        # self.max_iter = 1
        self.num_classes = 11
        self.output_folder = None
        self.patience = 2
        self.train_coco_file_path = None
        self.train_images_path = None
        self.validate_coco_file_path = None
        self.validate_images_path = None
        # self.weights =

    def _load_cfg(self):
        self.cfg = get_cfg()  # obtain detectron2's default config
        # self.cfg.merge_from_file(self.cfg_path)  # load values from a file
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        # self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.DATASETS.TRAIN = ('train_instances',)
        self.cfg.DATASETS.TEST = ('validate_instances',)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )  # self.weights
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.SOLVER.GAMMA = 0.8
        self.cfg.SOLVER.WEIGHT_DECAY = 0  # for MADGRAD
        self.cfg.SOLVER.MOMENTUM = 0  # for MADGRAD
        self.cfg.SOLVER.BASE_LR = 0.000001  # self.base_lr
        self.cfg.SOLVER.MAX_ITER = 600000  # elf.max_iter
        self.cfg.SOLVER.WARMUP_FACTOR = 1.0 / 200
        self.cfg.SOLVER.WARMUP_ITERS = 1000
        self.cfg.TEST.EVAL_PERIOD = self.eval_period
        self.cfg.SOLVER.WARMUP_METHOD = "linear"
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def _load_cfg_resuming(self, max_iter: int, lr: float):
        self.cfg = get_cfg()  # obtain detectron2's default config
        # self.cfg.merge_from_file(self.cfg_path)  # load values from a file
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        # self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.DATASETS.TRAIN = ('train_instances',)
        self.cfg.DATASETS.TEST = ('validate_instances',)
        self.cfg.MODEL.WEIGHTS = './output/potato_model_current.pth'
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.SOLVER.GAMMA = 0.8
        self.cfg.SOLVER.WEIGHT_DECAY = 0
        self.cfg.SOLVER.MOMENTUM = 0
        self.cfg.SOLVER.BASE_LR = lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.WARMUP_FACTOR = 1.0 / 200
        self.cfg.SOLVER.WARMUP_ITERS = 1000
        self.cfg.TEST.EVAL_PERIOD = self.eval_period
        self.cfg.SOLVER.WARMUP_METHOD = "linear"
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def _load_datasets(self,
                       train_coco_file_path,
                       train_images_path,
                       validate_coco_file_path,
                       validate_images_path,
                       ):
        register_coco_instances('train_instances', {}, train_coco_file_path, train_images_path)
        register_coco_instances('validate_instances', {}, validate_coco_file_path, validate_images_path)

    def _save_torchscript(self):
        print('Save torch_script file...')
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = './output/best_mAP.pth'
        model = build_model(self.cfg)
        DetectionCheckpointer(model).resume_or_load(self.cfg.MODEL.WEIGHTS)
        model.eval()
        image = torch.randint(255, size=(3, 512, 512))
        inputs = [{"image": image}]  # remove other unused keys
        if isinstance(model, GeneralizedRCNN):
            print('inference is Not None')

            def inference(mdl, inpts):
                # use do_postprocess=False so it returns ROI mask
                inst = mdl.inference(inpts, do_postprocess=False)[0]
                return [{"instances": inst}]
        else:
            print('inference is None')
            inference = None  # assume that we just call the model directly
        traceable_model = TracingAdapter(model, inputs, inference)
        ################ torchscript ######################
        ts_model = torch.jit.trace(traceable_model, (image,))
        torch.jit.save(ts_model, './output/potato_model.ts')

    def train_model(self, resume=False, output_folder='./'):
        def get_ymd():
            now = datetime.datetime.now()
            year = now.year
            month = str(now.month)
            day = str(now.day)
            hour = str(now.hour)
            if len(month) != 2:
                month = '0' + month
            if len(day) != 2:
                day = '0' + day
            if len(hour) != 2:
                hour = '0' + hour
            return '{}{}{}{}'.format(year, month, day, hour)

        count_iteration = self.count_iteration
        logger = logging.getLogger("detectron2")
        # handler = logging.StreamHandler(stream=sys.stdout)
        # logger.addHandler(handler)
        # logger.addFilter(CustomFilter())
        # Data loader
        dataset_names = [d_names for d_names in self.cfg.DATASETS.TEST]
        print(f'dataset_names={dataset_names}')
        train_loader = build_detection_train_loader(self.cfg, mapper=MyMapper(self.cfg, is_train=True))
        dataset_name = self.cfg.DATASETS.TEST[0]
        valid_loader = build_detection_test_loader(self.cfg, dataset_name, mapper=MyMapper(self.cfg, is_train=False))

        # Model, optimizer, scheduler
        model = build_model(self.cfg)
        # On  sparse  problems  both  weight_decay and momentum  should  be  set  to 0.
        optimizer = MADGRAD(
            model.parameters(), lr=self.cfg.SOLVER.BASE_LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            momentum=self.cfg.SOLVER.MOMENTUM
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=self.cfg.SOLVER.GAMMA, patience=self.patience, verbose=True, eps=1e-8
        )

        # Checkpoint
        checkpointer = DetectionCheckpointer(model, self.cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
        start_iter = (checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        max_iter = self.cfg.SOLVER.MAX_ITER
        periodic_checkpointer = PeriodicCheckpointer(checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

        # Use a writer to record loss and common metrics
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        writers = default_writers(self.cfg.OUTPUT_DIR, max_iter)
        # writers.append(WAndBWriter())

        # Help to record validation results
        best_mAP = 0
        valid_AP = []
        # Train loop
        model.train()
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(train_loader, range(start_iter, max_iter)):
                storage.iter = iteration
                loss_dict = model(data)  # Four types of loss obtained from the faster rcnn model
                losses = sum(loss_dict.values())
                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # store the loss and lr
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                if iteration % count_iteration == 0:
                    print(f'iter={iteration}, losses_reduced={losses_reduced}, lr={optimizer.param_groups[0]["lr"]}')
                    logger.info(
                        f'iter={iteration}, losses_reduced={losses_reduced}, lr={optimizer.param_groups[0]["lr"]}'
                    )

                # validation
                if (self.cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % self.cfg.TEST.EVAL_PERIOD == 0):
                    print("save potato_model_current")
                    checkpointer.save("potato_model_current")
                    shutil.copy(
                        './output/potato_model_current.pth',
                        os.path.join(output_folder, 'potato_model_current{}.pth'.format(get_ymd()))
                    )
                    # out_folder = os.path.join(self.cfg.OUTPUT_DIR, "inference")
                    # for dataset_name in dataset_names:
                    #     valid_loader = build_detection_test_loader(
                    #         self.cfg, dataset_name, mapper=MyMapper(self.cfg, is_train=False)
                    #     )
                    evaluator = get_evaluator(
                        self.cfg, dataset_name, os.path.join(
                            self.cfg.OUTPUT_DIR,
                            "inference"
                        )
                    )
                    eval_stat = inference_on_dataset(model, valid_loader, evaluator)
                    mAP = list(eval_stat.values())[0]['AP']
                    print(f'mAP={mAP}')
                    if mAP > best_mAP:
                        best_mAP = mAP
                        checkpointer.save("best_mAP")
                        print(f'Save Best Score: {best_mAP:.4f}')
                        logger.info(f'Save Best Score: {best_mAP:.4f}')
                        shutil.copy(
                            './output/best_mAP.pth',
                            os.path.join(output_folder, 'potato_model_best{}.pth'.format(get_ymd()))
                        )
                        shutil.copy(
                            './output/metrics.json',
                            os.path.join(output_folder, 'metrics{}.json'.format(get_ymd()))
                        )
                    scheduler.step(mAP)
                    # wandb.log({'eval-mAP': mAP,
                    #            'eval-AP50': list(eval_stat.values())[0]['AP50'],
                    #            'eval-AP75': list(eval_stat.values())[0]['AP75'],
                    #            'eval-APs': list(eval_stat.values())[0]['APs'],
                    #            'eval-APm': list(eval_stat.values())[0]['APm'],
                    #            'eval-APl': list(eval_stat.values())[0]['APl'],
                    #            'eval-belt': list(eval_stat.values())[0]['AP-belt'],
                    #            'eval-boot': list(eval_stat.values())[0]['AP-boot'],
                    #            'eval-cowboy_hat': list(eval_stat.values())[0]['AP-cowboy_hat'],
                    #            'eval-jacket': list(eval_stat.values())[0]['AP-jacket'],
                    #            'eval-sunglasses': list(eval_stat.values())[0]['AP-sunglasses'],
                    #            'global_step': storage.iter,
                    #           })
                    valid_AP.append(list(eval_stat.values())[0])
                    for writer in writers:
                        writer.write()
        # return pd.DataFrame.from_dict(valid_AP)

    def main(self, args):
        self.output_folder = args.output_folder
        self.train_coco_file_path = args.train_coco_file_path
        self.train_images_path = args.train_images_path
        self.validate_coco_file_path = args.validate_coco_file_path
        self.validate_images_path = args.validate_images_path
        resume = args.resume
        self._load_datasets(
            self.train_coco_file_path,
            self.train_images_path,
            self.validate_coco_file_path,
            self.validate_images_path
        )
        if resume:
            max_iter = input('Input max iteration (max_iter):')
            lr = input('Input learning rate (lr):')
            self._load_cfg_resuming(max_iter=int(max_iter), lr=float(lr))
        else:
            self._load_cfg()
        self.train_model(output_folder=self.output_folder, resume=resume)
        self._save_torchscript()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trainer Composition")
    parser.add_argument(
        "--train_coco_dir",
        type=str,
        dest="train_coco_file_path",
        required=True,
        help="Path of json file of COCO format"
    )
    parser.add_argument(
        "--train_images_dir",
        type=str,
        dest="train_images_path",
        required=True,
        help=""
    )
    parser.add_argument(
        "--validate_coco_dir",
        type=str,
        dest="validate_coco_file_path",
        required=True,
        help=""
    )
    parser.add_argument(
        "--validate_images_dir",
        type=str,
        dest="validate_images_path",
        required=True,
        help=""
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        dest="output_folder",
        required=False,
        default="./",
        help=""
    )
    parser.add_argument(
        "--resume",
        type=bool,
        dest="resume",
        default=False,
        required=False,
        help=""
    )
    args = parser.parse_args()
    pt = PotatoTrainer()
    pt.main(args)
