import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from lia.networks.generator import Generator


class Inference_LIA(nn.Module):

    def __init__(self, args, det_model=None):
        super(Inference_LIA, self).__init__()

        self.args = args
        model_path = args.lia_model_path
        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()
        self.img_source = self.load_image(args.source_path)
        if det_model is not None:
            roi = det_model.inference(self.img_source)
            self.img_source = self.crop_image(self.img_source, roi)
        self.img_source = self.img_preprocessing(self.img_source, args.size).cuda()

    def crop_image(self, image, roi):
        m, n, _ = image.shape
        length = (roi[2] - roi[0])/2 if (roi[2] - roi[0]) < (roi[3] - roi[1]) else (roi[3] - roi[1])/2
        min_len = min([roi[0], roi[1], n - roi[2], m - roi[3]])
        length = int(min_len) if min_len < length else int(length)
        xc, yc = int(roi[0]+(roi[2] - roi[0])/2), int(roi[1]+(roi[3] - roi[1])/2)
        print(image[yc-length*2:yc+length*2, xc-length*2:xc+length*2,:].shape)
        return image[yc-length*2:yc+length*2, xc-length*2:xc+length*2, :]

    def load_image(self, filename):
        img = Image.open(filename).convert('RGB')
        return np.asarray(img)

    def img_preprocessing(self, img, size):

        img = cv2.resize(img, (size, size))
        img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256
        img = torch.from_numpy(img/255.0).unsqueeze(0).float()  # [0, 1]
        imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

        return imgs_norm

    def inference(self, image, image_start):

        with torch.no_grad():

            h_start = self.gen.enc.enc_motion(self.img_preprocessing(image_start, self.args.size).cuda())
            img_target = self.img_preprocessing(image, self.args.size).cuda()
            img_recon = self.gen(self.img_source, img_target, h_start)
            img_recon = img_recon.clamp(-1, 1).squeeze(0).cpu()
            img_recon = ((img_recon - img_recon.min()) / (img_recon.max() - img_recon.min()) * 255).type('torch.ByteTensor')
            img_recon = img_recon.numpy()
            img_recon = np.transpose(img_recon, (1, 2, 0))

        return img_recon