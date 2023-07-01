import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import cv2
import os
from PIL import Image
# from pathlib import Path
from tqdm import tqdm


def load_image(filename):
    return Image.open(filename).convert('RGB')


def img_preprocessing(img, size):

    img = np.asarray(img)
    img = cv2.resize(img, (size, size))
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256
    img = torch.from_numpy(img/255.0).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


class Inference_LIA(nn.Module):

    def __init__(self, args):
        super(Inference_LIA, self).__init__()

        model_path = args.model_path
        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()
        self.img_source = load_image(args.source_path)
        self.img_source = img_preprocessing(self.img_source, args.size).cuda()

    def inference(self, image, image_start):

        with torch.no_grad():

            h_start = self.gen.enc.enc_motion(img_preprocessing(image_start, args.size).cuda())
            img_target = img_preprocessing(image, args.size).cuda()
            img_recon = self.gen(self.img_source, img_target, h_start)
            img_recon = img_recon.clamp(-1, 1).squeeze(0).cpu()
            img_recon = ((img_recon - img_recon.min()) / (img_recon.max() - img_recon.min()) * 255).type('torch.ByteTensor')
            img_recon = img_recon.numpy()
            img_recon = np.transpose(img_recon, (1, 2, 0))

        return img_recon


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
    lia = Inference_LIA(args)
    start = False
    recon_img = None
    record = False
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('zedv6.mp4', fourcc, 8, (256, 256), True)

    while True:
        keycode = cv2.waitKey(1)
        frame = cam.get_frame()

        if start:
            recon_img = lia.inference(frame, start_img)
            recon_img = cv2.cvtColor(recon_img, cv2.COLOR_BGR2RGB)

        if keycode == ord('s'):
            start_img = frame
            start = True
        if keycode == ord('q'):
            break
        if keycode == ord('r'):
            record = True

        if record:
            out.write(recon_img)

        if recon_img is not None:
            # cv2.imshow("live", frame)
            cv2.imshow("live", recon_img)
        else:
            cv2.imshow("live", frame)

    out.release()



if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model_path", type=str, default='./models/vox.pt')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='GE.jpg')

    args = parser.parse_args()
    main(args)