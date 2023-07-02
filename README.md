# Real Time LIA
This repository implements real-time image reenactment using the technique called ["Latent Image Animator: Learning to Animate Images via Latent Space Navigation."](https://arxiv.org/abs/2203.09043) By integrating face detection and face reenactment, this project has the capability to process various types of face images as input. It enables the user to control the facial expressions of a real person based on user's face in real-time. It is worth noting that for optimal results, it is recommended to use face images with a size larger than 256x256.

## Getting Started
* Clone this repository

   ```bash
   git https://github.com/Rayhchs/Real-Time-LIA.git
   cd Real-Time-LIA
   ```

* Download models
You can download LIA model from [here](https://drive.google.com/drive/folders/1N4QcnqUQwKUZivFV-YeBuPyH4pGJHooc). In this repository, we only use **vox.pt**.
Afterward, you will put the **vox.pt** to **./lia**.

* Usage

   ```bash
   python main.py --source_path [path-to-your-source-image]
   ```
   
   - press 's' to start reenactment
   - press 'r' to record video
   - press 'q' to quit

## Acknowledge
Code and pretrain weights heavily borrows from [LIA](https://github.com/wyhsirius/LIA) and [blazeface]([https://github.com/ipazc/mtcnn](https://github.com/zineos/blazeface)). Thanks for the excellent work!

Latent Image Animator: Learning to Animate Images via Latent Space Navigation:
```bibtex
@inproceedings{
wang2022latent,
title={Latent Image Animator: Learning to Animate Images via Latent Space Navigation},
author={Yaohui Wang and Di Yang and Francois Bremond and Antitza Dantcheva},
booktitle={International Conference on Learning Representations},
year={2022}
}
```
