import argparse
# from multiprocessing import freeze_support

import cv2
import detectron2
from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
# from detectron2.export import TracingAdapter
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import GeneralizedRCNN, build_model
# import cv2
# import numpy as np
# import torch
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
    video_input_path = p_args.video_input_path
    video_output_path = p_args.video_output_path
    cfg, predictor = main(p_args)
    # tsd = TorchscriptDetection(p_args.weight_path, use_cuda=eval(p_args.cuda))
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
            # pred_boxes, scores, pred_classes, masks = tsd.detect(frame)
            # print(f'pred_boxes={pred_boxes}')
            # print(f'scores={scores}')
            # print(f'pred_classes={pred_classes}')
            # frame = cv2.resize(frame, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
            img = visualize(frame, cfg, predictor)
            # img = cv2.resize(img, None,fx=0.25, fy=0.25)
            video_writer_rgb.write(img)
            # cv2.imshow('img', cv2.resize(img, None, fx=0.3, fy=0.3))
        # if cv2.waitKey(1) == ord('q'):
        #     break

    video_writer_rgb.release()

    cv2.destroyAllWindows()
