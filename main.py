import cv2
import torch
import torch.nn as nn
import argparse
import numpy as np
import tensorflow as tf
from lia.inference_lia import Inference_LIA
from detection.inference_det import Inference_det


class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 16)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        sys.exit("Invalid Camera") if not self.cap.isOpened() else print("Valid Camera")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            sys.exit("Can't receive frame (stream end?). Exiting ...")
        return frame

    def release_cam(self):
            self.cap.release()


def main(args):
    cam = Camera()
    det = Inference_det(args.det_model_path)
    lia = Inference_LIA(args, det)
    start = False
    recon_img = None
    record = False
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('zedv6.mp4', fourcc, 8, (400, 400), True)

    while True:
        keycode = cv2.waitKey(1)
        frame = cam.get_frame()

        if start:
            frame = lia.crop_image(frame, roi)
            recon_img = lia.inference(frame, start_img)
            recon_img = cv2.cvtColor(recon_img, cv2.COLOR_BGR2RGB)

        if keycode == ord('s'):
            start_img = frame
            roi = det.inference(start_img)
            start_img = lia.crop_image(start_img, roi)
            start = True
        elif keycode == ord('q'):
            break
        elif keycode == ord('r'):
            record = True

        if record and start:
            out.write(recon_img)

        if recon_img is not None:
            # test = lia.crop_image(frame, roi)
            # print(test.shape)
            # cv2.imshow("live", test)
            cv2.imshow("live", recon_img)
        else:
            cv2.imshow("live", frame)

    out.release()


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--lia_model_path", type=str, default='./lia/vox.pt')
    parser.add_argument("--det_model_path", type=str, default='./detection/Det.tflite')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='macron.png')

    args = parser.parse_args()
    main(args)