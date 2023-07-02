# Real Time LIA
This repository conducted real time image reenactment using ["Latent Image Animator: Learning to Animate Images via Latent Space Navigation"](https://github.com/wyhsirius/LIA). By combining face detection and face reenactment, the project can accept any type of face image as input. Combining with face detection and face reenactment, any kind of face image can be inputed. However, image size should be larger than 256x256. This project allows you to control a real person's face based on your expression. 

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
