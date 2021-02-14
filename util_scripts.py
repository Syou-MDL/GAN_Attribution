# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np

import config
import misc

# ----------------------------------------------------------------------------
# Generate random images or image grids using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_fake_images(pkl_path, out_dir, num_pngs, image_shrink=1, random_seed=1000, minibatch_size=1):
    random_state = np.random.RandomState(random_seed)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print('Loading network...')
    G, D, Gs = misc.load_network_pkl(pkl_path)

    latents = misc.random_latents(num_pngs, Gs, random_state=random_state)
    labels = np.zeros([latents.shape[0], 0], np.float32)
    images = Gs.run(latents, labels, minibatch_size=config.num_gpus*256, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
    for png_idx in range(num_pngs):
        print('Generating png to %s: %d / %d...' % (out_dir, png_idx, num_pngs), end='\r')
        if not os.path.exists(os.path.join(out_dir, 'ProGAN_%08d.png' % png_idx)):
            misc.save_image_grid(images[png_idx:png_idx+1], os.path.join(out_dir, 'ProGAN_%08d.png' % png_idx), [0,255], [1,1])
    print()


def generate_fake_images_all(run_id, out_dir, num_pngs, image_shrink=1, random_seed=1000, minibatch_size=1,num_pkls=50):
    random_state = np.random.RandomState(random_seed)
    out_dir = os.path.join(out_dir,str(run_id))
    
    result_subdir = misc.locate_result_subdir(run_id)
    snapshot_pkls = misc.list_network_pkls(result_subdir, include_final=False)
    assert len(snapshot_pkls) >= 1
 
    
    for snapshot_idx, snapshot_pkl in enumerate(snapshot_pkls[:num_pkls]):
        prefix = 'network-snapshot-'; postfix = '.pkl'
        snapshot_name = os.path.basename(snapshot_pkl)
        tmp_dir = os.path.join(out_dir, snapshot_name.split('.')[0])
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        assert snapshot_name.startswith(prefix) and snapshot_name.endswith(postfix)
        snapshot_kimg = int(snapshot_name[len(prefix) : -len(postfix)])

        print('Loading network...')
        G, D, Gs = misc.load_network_pkl(snapshot_pkl)

        latents = misc.random_latents(num_pngs, Gs, random_state=random_state)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=config.num_gpus*32, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        for png_idx in range(num_pngs):
            print('Generating png to %s: %d / %d...' % (tmp_dir, png_idx, num_pngs), end='\r')
            if not os.path.exists(os.path.join(out_dir, 'ProGAN_%08d.png' % png_idx)):
                misc.save_image_grid(images[png_idx:png_idx+1], os.path.join(tmp_dir, 'ProGAN_%08d.png' % png_idx), [0,255], [1,1])
    print()

import tensorflow as tf
from PIL import Image
def recover_latent(pkl_path, out_dir, num_pngs, image_shrink=1, random_seed=1000, minibatch_size=1):
    random_state = np.random.RandomState(random_seed)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Choose a random starting point
    zp = tf.Variable(np.random.normal(size=(1, 512)), dtype=tf.float32)

    # Load the image for which we want to recover the latent vector
    # and crete an appropriate tensor for it
    start_img = Image.open("gen/celeba_v2/ProGAN_00000001.png")
    start_img.resize((128, 128), Image.ANTIALIAS)
    start_img_np = np.array(start_img) / 255
    fz = tf.Variable(start_img_np, tf.float32)
    fz = tf.expand_dims(fz, 0)
    fz = tf.cast(fz, tf.float32)

    print('Loading network...')
    G, D, Gs = misc.load_network_pkl(pkl_path)
    print('loaded!')
    labels = np.zeros([fz.shape[0], 0], np.float32)

    # Define the optimization problem
    fzp = Gs.run(fz, labels, minibatch_size=config.num_gpus * 256, num_gpus=config.num_gpus,
                 out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
    loss = tf.losses.mean_squared_error(labels=fz, predictions=fzp)

    # Decayed gradient descent
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.99
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               10000, 0.005)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    # Optimize on the variable zp
    train = opt.minimize(loss, var_list=zp, global_step=global_step)


