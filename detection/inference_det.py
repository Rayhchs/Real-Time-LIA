import cv2
import numpy as np
import tensorflow as tf
from detection.utils.prior_box import PriorBox
from detection.utils.py_cpu_nms import py_cpu_nms
from detection.utils.box_utils import decode
from detection.utils.config import  cfg_blaze


class Inference_det():
    def __init__(self, det_model_path):

        self.interpreter = tf.lite.Interpreter(model_path=det_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def img_preprocessing(self, image):

        # image preprocessing
        img = np.float32(image)
        target_size = 128
        im_size_max = np.max(img.shape[0:2])
        height, width, _ = img.shape
        image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
        image_t[:, :] = (104, 117, 123)

        image_t[0:0 + height, 0:0 + width] = img
        img = cv2.resize(image_t, (target_size, target_size))

        resize = float(target_size) / float(im_size_max)

        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img, resize

    def inference(self, image):

        processed_img, resize = self.img_preprocessing(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_img)  
        self.interpreter.invoke()
        conf = self.interpreter.get_tensor(self.output_details[0]['index'])
        landms = self.interpreter.get_tensor(self.output_details[1]['index'])
        loc = self.interpreter.get_tensor(self.output_details[2]['index'])

        priorbox = PriorBox(cfg_blaze, image_size=(128, 128))
        prior_data = priorbox.forward()

        boxes = decode(loc.squeeze(0), prior_data, cfg_blaze['variance'])
        np.clip(boxes,1, 0.00001)
        scale = np.array([128,128,128,128])
        boxes = boxes * scale / resize
        boxes = np.nan_to_num(boxes)
        scores = conf.squeeze(0)[:, 1]

        # ignore low scores
        inds = np.where(scores > 0.5)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        d = dets[0]

        return [int(d[0]), int(d[1]), int(d[2]), int(d[3])]
