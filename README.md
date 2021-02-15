# GAN_Attribution

This is the implementation for GAN attribution via Latent Recovery


### [Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints](https://arxiv.org/pdf/1811.08180.pdf)
[Ning Yu](https://sites.google.com/site/ningy1991/), [Larry Davis](http://users.umiacs.umd.edu/~lsd/), [Mario Fritz](https://cispa.saarland/group/fritz/)<br>
ICCV 2019

### [paper](https://arxiv.org/pdf/1811.08180.pdf) 


## GAN fingerprints demo
<img src='classifier_visNet/demo/demo.gif' width=800>

- The display window is refreshed every 5 seconds. Each refresh represents a testing case.
- The left two images are the testing input and its image fingerprint.
- The 1st row lists the learned model fingerprints of 5 sources, which are unchanged w.r.t refresh.
- The 2nd row lists the image fingerprint response to each model fingerprint.
- The 3rd row is a barplot of the response intensity w.r.t. each model fingerprint, the higher the stronger correlation, which also evidences for classification. The plot locates at the column corresponding to the predicted label.
- The model fingerprint title in green and the green barplot indicate the label ground truth.
- If there exists a red barplot, it means an incorrect classification. The red barplot indicates the predicted label, which is different from the label ground truth in green.

## Abstract
Recent advances in Generative Adversarial Networks (GANs) have shown increasing success in generating photorealistic images. But they also raise challenges to visual forensics and model attribution. We present the first study of learning GAN fingerprints towards image attribution and using them to classify an image as real or GAN-generated. For GAN-generated images, we further identify their sources. Our experiments show that:
- **Existence**: GANs carry distinct model fingerprints and leave stable fingerprints in their generated images, which support image attribution;
- **Uniqueness**: Even minor differences in GAN training can result in different fingerprints, which enables fine-grained model authentication;
- **Persistence**: Fingerprints persist across different image frequencies and patches and are not biased by GAN artifacts;
- **Immunizability**: Fingerprint finetuning is effective in defending against five types of image perturbation attacks;
- **Superiority**: Comparisons also show our learned fingerprints consistently outperform several baselines in a variety of setups.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA 10.0 + CuDNN 7.5
- Python 3.6
- tensorflow-gpu 1.12
- Other Python dependencies: numpy, scipy, Pillow, lmdb, opencv-python, cryptography, h5py, six, chainer, cython, cupy, pyyaml

## Datasets
To train GANs and our classifiers, we consider two real-world datasets:
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) aligned and cropped face dataset. We crop each image centered at (x,y) = (89,121) with size 128x128 before training.
- [LSUN](https://github.com/fyu/lsun) bedroom scene dataset. We select the first 200k images, center-crop them to square size according to the shorter side length, and resize them to 128x128 before training.
  
## GAN sources
For each dataset, we pre-train four GAN sources: ProGAN
- [ProGAN](https://github.com/tkarras/progressive_growing_of_gans)
  - **Data preparation**. Run, e.g.,
    ```
    python3 dataset_tool.py \
    create_from_images \
    datasets/celeba_align_png_cropped/ \
    ../celeba_align_png_cropped/
    ```
    where `datasets/celeba_align_png_cropped/` is the output directory containing the prepared data format that enables efficient streaming, and `../celeba_align_png_cropped/` is the training dataset directory containing 128x128 png images.
  - **Training**. Run, e.g.,
    ```
    python3 run.py \
    --app train \
    --training_data_dir datasets/celeba_align_png_cropped/ \
    --out_model_dir models/celeba_align_png_cropped_seed_v0/ \
    --training_seed 0
    ```
    where
    - `training_data_dir`: The prepared training dataset directory that can be efficiently called by the code.
    - `out_model_dir`: The output directory containing trained models, training configureation, training log, and training snapshots.
    - `training_seed`: The random seed that differentiates training instances.
  - **Pre-trained models**. Download our pre-trained models [here](https://drive.google.com/drive/folders/1E4Bm8xshBTDPBU3Nh8x6ASFduLZZmtVI?usp=sharing) and put them at `ProGAN/models/`. The models named with `_seed_v%d` are only different in random seeds from each other.
  - **Generation**. With a pre-trained model, generate images of size 128x128 by running, e.g.,
    ```
    python3 run.py \
    --app gen \
    --model_path models/celeba_align_png_cropped.pkl \
    --out_image_dir gen/celeba_align_png_cropped/ \
    --num_pngs 10000 \
    --gen_seed 0
    ```
    where
    - `model_path`: The pre-trained GAN model.
    - `out_image_dir`: The outpupt directory containing generated images.
    - `num_pngs`: The number of generated images.
    - `gen_seed`: The random seed that differentiates generation instances.

## GAN classifier
Given images of size 128x128 from real dataset or generated by different GANs, we train a classifier to attribute their sources. The code is modified from [ProGAN](https://github.com/tkarras/progressive_growing_of_gans).
- **Data preparation**. Run, e.g.,
  ```
  cd classifier/
  python3 data_preparation.py \
  --in_dir ../GAN_classifier_datasets/train/ \
  --out_dir datasets/train/
  ```
  where
  - `in_dir`: The input directory containing subdirectories of images. Each subdirectory represents a data source, either from the real dataset or generated by a GAN.
  - `out_dir`: The output directory containing the prepared data format and its source label format that enable efficient streaming.
- **Training**. Run, e.g.,
  ```
  cd classifier/
  python3 run.py \
  --app train \
  --training_data_dir datasets/training/ \
  --validation_data_dir datasets/validation/ \
  --out_model_dir models/GAN_classifier/ \
  --training_seed 0
  ```
  where
  - `training_data_dir`: The prepared training dataset directory that can be efficiently called by the code.
  - `validation_data_dir`: The prepared validation dataset directory that can be efficiently called by the code.
  - `out_model_dir`: The output directory containing trained models, training configureation, training log, and training snapshots.
  - `training_seed`: The random seed that differentiates training instances.
- **Pre-trained models**. Download our pre-trained models [here](https://drive.google.com/drive/folders/1gbfUjHsjs8929cas-8TRsBtQaJKLcXXC?usp=sharing) and put them at `classifier/models/`. They separately apply to the two datasets and differentiate either from the 5 sources (1 real + 4 GAN architectures) or from the 11 sources (1 real + 10 ProGANs pre-trained with different random seeds).
- **Testing**. Run, e.g.,
  ```
  cd classifier/
  python3 run.py \
  --app test \
  --model_path models/CelebA_ProGAN_SNGAN_CramerGAN_MMDGAN_128.pkl \
  --testing_data_path ../GAN_classifier_datasets/test/
  ```
  where
  - `model_path`: The pre-trained GAN model.
  - `testing_data_path`: The path of testing image file or the directory containing a collection of testing images. There is no need to execute data preparation for testing image(s).


## Citation
```
@inproceedings{yu2019attributing,
    author = {Yu, Ning and Davis, Larry and Fritz, Mario},
    title = {Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints},
    booktitle = {IEEE International Conference on Computer Vision (ICCV)},
    year = {2019}
}
```

## Acknowledgement
- This repository is heavily borrowed from [ProGAN](https://github.com/tkarras/progressive_growing_of_gans) repository.
