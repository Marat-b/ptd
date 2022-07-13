import os
import shutil

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import PeriodicCheckpointer, default_writers
from detectron2.evaluation import inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.utils import comm
from detectron2.utils.events import EventStorage
from tools.plain_train_net import get_evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from madgrad import MADGRAD

from mapper import MyMapper


class PotatoTrainer:
    def __init__(self
                 ):
        # self.base_lr = base_lr
        self.cfg = None
        self.cfg_path = './config_potato.yml'
        # self.dataset_test = None
        # self.dataset_train = None
        self.eval_period = None
        # self.max_iter = 1
        # self.num_classes = num_classes
        self.output_folder = None
        self.train_coco_file_path = None
        self.train_images_path = None
        self.validate_coco_file_path = None
        self.validate_images_path = None
        # self.weights =

    def _load_cfg(self):
        self.cfg = get_cfg()  # obtain detectron2's default config
        # self.cfg.merge_from_file(self.cfg_path)  # load values from a file
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        # self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.DATASETS.TRAIN = ('train_instances',)
        self.cfg.DATASETS.TEST = ('validate_instances',)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml") #self.weights
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 #self.num_classes
        self.cfg.SOLVER.BASE_LR = 0.0001 #self.base_lr
        self.cfg.SOLVER.MAX_ITER = 20 #elf.max_iter
        # self.cfg.SOLVER.WARMUP_FACTOR = 1.0 / 200
        # self.cfg.SOLVER.WARMUP_ITERS = 200
        self.cfg.TEST.EVAL_PERIOD = 10 #self.eval_period
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def _load_datasets(self,
                       train_coco_file_path,
                       train_images_path,
                       validate_coco_file_path,
                       validate_images_path,
                       ):
        register_coco_instances('train_instances', {}, train_coco_file_path, train_images_path)
        register_coco_instances('validate_instances', {}, validate_coco_file_path, validate_images_path)

    def train_model(self, resume=False, output_folder='./'):
        logger = logging.getLogger("detectron2")
        # Data loader
        dataset_names = [d_names for d_names in self.cfg.DATASETS.TEST]
        print(f'dataset_names={dataset_names}')
        train_loader = build_detection_train_loader(self.cfg, mapper=MyMapper(self.cfg, is_train=True))
        #     dataset_name = self.cfg.DATASETS.TEST[0]
        #     valid_loader = build_detection_test_loader(self.cfg, dataset_name, mapper=MyMapper(self.cfg,
        #     is_train=False))

        # Model, optimizer, scheduler
        model = build_model(self.cfg)
        optimizer = MADGRAD(
            model.parameters(), lr=self.cfg.SOLVER.BASE_LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            momentum=self.cfg.SOLVER.MOMENTUM
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=self.cfg.SOLVER.GAMMA, patience=2, verbose=True, eps=1e-8
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
                # print(f'losses_reduced={losses_reduced}, lr={optimizer.param_groups[0]["lr"]}')
                if iteration % 10 == 0:
                    logger.info(
                        f'iter={iteration}, losses_reduced={losses_reduced}, lr={optimizer.param_groups[0]["lr"]}'
                    )

                # validation
                if (self.cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % self.cfg.TEST.EVAL_PERIOD == 0):
                    checkpointer.save("potato_model_current")
                    shutil.copy(
                        './output/potato_model_current.pth',
                        os.path.join(output_folder, 'potato_model_current.pth')
                    )
                    output_folder = os.path.join(self.cfg.OUTPUT_DIR, "inference")
                    for dataset_name in dataset_names:
                        valid_loader = build_detection_test_loader(
                            self.cfg, dataset_name, mapper=MyMapper(self.cfg, is_train=False)
                        )
                        evaluator = get_evaluator(
                            self.cfg, dataset_name, os.path.join(
                                self.cfg.OUTPUT_DIR,
                                "inference"
                                )
                            )
                        eval_stat = inference_on_dataset(model, valid_loader, evaluator)
                        mAP = list(eval_stat.values())[0]['AP']
                        if mAP > best_mAP:
                            best_mAP = mAP
                            checkpointer.save("best_mAP")
                            logger.info(f'Save Best Score: {best_mAP:.4f}')
                            shutil.copy(
                                './output/best_mAP.pth',
                                os.path.join(output_folder, 'potato_model_best.pth')
                            )
                            shutil.copy(
                                './output/metrics.json',
                                os.path.join(output_folder, 'metrics.json')
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
        self._load_datasets(self.train_coco_file_path,
                            self.train_images_path,
                            self.validate_coco_file_path,
                            self.validate_images_path
                            )
        self._load_cfg()
        self.train_model(output_folder=self.output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trainer Composition")
    parser.add_argument("--train_coco_dir",
                        type=str,
                        dest="train_coco_file_path",
                        required=True,
                        help="Path of json file of COCO format")
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
        required=True,
        help=""
    )
    args = parser.parse_args()
    pt = PotatoTrainer()
    pt.main(args)
