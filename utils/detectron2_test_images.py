import argparse
# from multiprocessing import freeze_support
import pathlib
from os import listdir
from os.path import isfile, join

import cv2
import detectron2
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog
# from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
# from detectron2.export import TracingAdapter
from detectron2.config import get_cfg
from detectron2 import model_zoo
# from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def visualize(image, cfg, predictor):
  outputs = predictor(image)
  v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("potato_dataset_test")) #cfg.DATASETS.TRAIN[0]
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  out_image = out.get_image()[:, :, ::-1]
  return out_image

def main(args):
    # print(args)
    weight_path = args.weight_path
    test_coco_file_path = args.test_coco_file_path
    test_images_path = args.test_images_path
    # output_path = args.output_path
    # new_shape = args.shape
    classes = args.classes
    score = args.score
    if args.cuda:
        device = 'cuda'
    else:
        device ='cpu'

    # register_coco_instances(
    #     "potato_dataset_test", {},
    #     test_coco_file_path,
    #     test_images_path
    # )

    # img = cv2.imread('./images/000000000.jpg')

    cfg = get_cfg()
    # cfg = add_export_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = classes
    # cfg.DATASETS.TEST = ('potato_dataset_test',)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device
    # model = build_model(cfg)
    # DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    # model.eval()
    predictor = DefaultPredictor(cfg)
    detectron2.data.datasets.load_coco_json(
         test_coco_file_path,
         test_images_path, dataset_name="potato_dataset_test"
        )
    return cfg, predictor

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Video Detection & Visualize")
    parser.add_argument(
        "-tc", "--test_coco_dir",
        type=str,
        dest="test_coco_file_path",
        required=True,
        help="Path of json file of COCO format"
    )
    parser.add_argument(
        "-ti", "--test_images_dir",
        type=str,
        dest="test_images_path",
        required=True,
        help=""
    )
    parser.add_argument(
        "-w", "--weight",
        type=str,
        dest="weight_path",
        required=True,
        help="Path of weight file"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Path of input video file"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Path of output video file"
    )
    parser.add_argument(
        "-s", "--score",
        type=float,
        dest="score",
        required=False,
        default=0.5,
        help="Confidence of prediction"
    )
    parser.add_argument(
        "-c", "--classes", default=1,
        type=int,
        dest="classes",
        help="Number of classes"
    )
    parser.add_argument(
        "-cpu", "--cpu",
        dest="cuda",
        default=True,
        action="store_false",
        help="Use CUDA (default)"
    )
    p_args = parser.parse_args()
    input_dir = p_args.input_dir
    output_dir = p_args.output_dir
    cfg, predictor = main(p_args)
    # tsd = TorchscriptDetection(p_args.weight_path, use_cuda=eval(p_args.cuda))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and
             join(input_dir, f).split('.')[1] == 'jpg']

    for file in tqdm(files):
        # print('---------------------------')
        im1 = cv2.imread(join(input_dir, file), cv2.IMREAD_COLOR)
        # img = cv2.resize(img, p_args.shape)
        # frame = frame[:, :, [2, 1, 0]]
        # pred_boxes, scores, pred_classes, masks = tsd.detect(im1)
        # print(f'pred_boxes={pred_boxes}')
        # print(f'scores={scores}')
        # print(f'pred_classes={pred_classes}')

        img = visualize(im1, cfg, predictor)
        cv2.imwrite(join(output_dir, file), img)

    cv2.destroyAllWindows()
