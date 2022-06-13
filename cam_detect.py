import cv2
import torch
import random
import numpy as np
import pickle as pkl
import torchvision.transforms as transforms

from models import load_model
from utils.utils import load_classes, rescale_boxes, non_max_suppression
from utils.transforms import Resize, DEFAULT_TRANSFORMS


def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()


def write(img, detections, classes, colors):
    for x1, y1, x2, y2, conf, cls_pred in detections:
        # print('detections:', detections)
        # print('x1:', x1)
        c1 = [int(x1), int(y1)]
        c2 = [int(x2), int(y2)]
        # print('classes:', classes)
        # print('int(cls_pred):', int(cls_pred))
        try:
            label = classes[int(cls_pred)]
            color = random.choice(colors)
            cv2.rectangle(img, c1, c2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        except Exception as e:
            print('err:', repr(e))
    
    return img


def run():
    model = "config/yolov3.cfg"
    weights = "checkpoints3/yolov3_ckpt_100.pth"
    classes = "data/custom/classes.names"
    # weights = "yolov3.weights"
    # classes = "data/coco.names"
    img_size = 416
    conf_thres  = 0.3
    nms_thres  = 0.15

    # Extract class names from file
    classes = load_classes(classes)  # List of class names
    colors = pkl.load(open("pallete", "rb"))

    model = load_model(model, weights)

    cap = cv2.VideoCapture(0)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:

            detections = detect_image(
                model=model,
                image=frame,
                img_size=img_size,
                conf_thres=conf_thres,
                nms_thres=nms_thres
            )
            
            # print('detections.shape:', detections.shape)
            # print('detections:', detections)

            if detections.shape[0] == 0:
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            else:
                img = write(
                    img=frame,
                    detections=detections,
                    classes=classes,
                    colors=colors
                )
                cv2.imshow("frame", img)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
        else:
            break


if __name__ == '__main__':
    run()
