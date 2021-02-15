# GAN Attribution via Latent Recovery

This is the tensorflow implementation for GAN attribution via Latent Recovery

<!-- 
### [paper](https://arxiv.org/pdf/1811.08180.pdf) 
### [Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints](https://arxiv.org/pdf/1811.08180.pdf)


## GAN fingerprints demo
<img src='classifier_visNet/demo/demo.gif' width=800>
-->

## Prerequisites
- Linux
- NVIDIA GPU + CUDA 11.0
- Python 3.6
- tensorflow-gpu 1.14
- Other Python dependencies: numpy, scipy, Pillow, lmdb, opencv-python

## For Raiden User
If you can access to Raiden, you can use following command to reproduct the exact enviroment in paper.
1. Login in to Raiden:
  ```
  ssh username@raiden.riken.jp
  ```
2. Login to the container :
  ```
  qrsh -jc gpu-container_g1_dev -ac d=nvcr-tensorflow-1909-py3,ep1=8888
  ```
3. Execute the following shell scripts to set environmental value.
  ```
  . /fefs/opt/dgx/env_set/nvcr-tensorflow-1909-py3.sh 
  /usr/local/bin/nvidia_entrypoint.sh
  ```
4. Set the following environmental values to connect internet
  ```
  export MY_PROXY_URL="http://10.1.10.1:8080/" 
  export HTTP_PROXY=$MY_PROXY_URL 
  export HTTPS_PROXY=$MY_PROXY_URL
  export FTP_PROXY=$MY_PROXY_URL
  export http_proxy=$MY_PROXY_URL
  export https_proxy=$MY_PROXY_URL
  export ftp_proxy=$MY_PROXY_URL 
  ```
5. Set installation directories and install nessesary packages
  ```
  mkdir -p ~/.raiden/nvcr-tensorflow-1909-py3
  export PATH="${HOME}/.raiden/nvcr-tensorflow-1909-py3/bin:$PATH"
  export LD_LIBRARY_PATH="${HOME}/.raiden//nvcr-tensorflow-1909-py3/lib:$LD_LIBRARY_PATH"  
  export LDFLAGS=-L/usr/local/nvidia/lib64
  export PYTHONPATH="${HOME}/.raiden/nvcr-tensorflow-1909-py3/lib/python3.5/site-packages" 
  export PYTHONUSERBASE="${HOME}/.raiden/nvcr-tensorflow-1909-py3" 
  export PREFIX="${HOME}/.raiden/nvcr-tensorflow-1909-py3" 
  
  pip install pillow --user
  ...
  ```

## Datasets
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) aligned and cropped face dataset. We crop each image centered at (x,y) = (89,121) with size 128x128 before training.
  
## GAN architecture
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

## Latent recovery
Given target model and random input latents(stored in target.npz), we conduct latent recovery using source model. 
 - **Conduct Recovery**. Run, e.g.,
  ```
  python3 recovery.py \
  --name='celeba_01_4init' \
  --pkl_path1='models/celeba_align_png_cropped_seed_v0.pkl' \
  --pkl_path2='models/celeba_align_png_cropped_seed_v1.pkl' \
  --out_dir='recover_result' \
  --num_init 4 --num_total_sample 100\
  --loss 'lpips'
  ```
  where
  - `name`: experiment name
  - `pkl_path1`: The pkl file path of source GAN model.
  - `pkl_path2`: The pkl file path of target GAN model.  
  - `out_dir`: The output directory for recovered latents.
  - `num_init`: Rcovery iteration for each target image
  - `num_total_sample`: Number of target images
  - `loss`: loss function('lpips' or 'l2')
  
## Accuracy Tests
Given recovered 

## Acknowledgement
- The code in this repository is heavily borrowed from [ProGAN](https://github.com/tkarras/progressive_growing_of_gans).
