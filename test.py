import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import misc
import config
import tfutil
import argparse

pkl_path = 'models/celeba_align_png_cropped_seed_v0.pkl'
pkl_path2 = 'models/celeba_align_png_cropped_seed_v1.pkl'
num_init = 20
image_shrink = 1
random_seed = 2020
minibatch_size = 1


def addGaussianNoise(src, sigma=5):
    gauss = np.random.normal(0, sigma, (3, 128, 128))
    noisy = src + gauss
    return noisy


def rescale_output(output):
    scalor = tf.constant([127.5], dtype='float32')
    output = tf.multiply(output, scalor)
    output = tf.add(output, scalor)
    return output


def recovery(name,pkl_path1,pkl_path2, out_dir, target_latents_dir,num_init=20, num_total_sample=100,image_shrink = 1,random_seed = 2020,minibatch_size = 1,noise_sigma= 0):
#     misc.init_output_logging()
#     np.random.seed(random_seed)
#     print('Initializing TensorFlow...')
#     os.environ.update(config.env)
#     tfutil.init_tf(config.tf_config)

    print('num_init:'+str(num_init))
    
    # load sorce model
    print('Loading network1...'+pkl_path1)
    _, _, G_sorce = misc.load_network_pkl(pkl_path1)
    
    # load target model
    print('Loading  network2...'+pkl_path2)
    _, _, G_target = misc.load_network_pkl(pkl_path2)
    
    # load Gt
    Gt = tfutil.Network('Gt', num_samples=num_init, num_channels=3,resolution=128, func='networks.G_recovery')
    latents = misc.random_latents(num_init, Gt, random_state=None)
    labels = np.zeros([latents.shape[0], 0], np.float32)
    Gt.copy_vars_from_with_input(G_target, latents)
    
    # load Gs
    Gs = tfutil.Network('Gs', num_samples=num_init, num_channels=3,resolution=128, func='networks.G_recovery')
    Gs.copy_vars_from_with_input(G_sorce, latents)
    
    out_dir = os.path.join(out_dir,name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    def G_loss(G, target_images):
        tmp_latents = tfutil.run(G.trainables['Input/weight'])
        G_out = G.get_output_for(tmp_latents, labels, is_training=True)
        G_out = rescale_output(G_out)
        return tf.losses.mean_squared_error(target_images, G_out)
    
    z_init = []
    z_recovered=[]
    
    #load target z
    if target_latents_dir is not None:
        print('using latents:'+target_latents_dir)
        pre_latents = np.load(target_latents_dir)
    

    for k in range(num_total_sample):
        result_dir =os.path.join(out_dir,str(k)+'.png') 
            
        #============sample target image
        if target_latents_dir is not None:
            latent = pre_latents[k]
        else:
            latents = misc.random_latents(1, Gs, random_state=None)
            latent = latents[0]
        z_init.append(latent)
        
        latents = np.zeros((num_init,512))
        for i in range(num_init):
            latents[i] = latent
        Gt.change_input(inputs=latents)

        #================add_noise
        target_images = Gt.get_output_for(latents, labels, is_training=False)
        target_images_tf = rescale_output(target_images)
        target_images = tfutil.run(target_images_tf)

        target_images_noise = addGaussianNoise(target_images, sigma=noise_sigma)
        target_images_noise = tf.cast(target_images_noise, dtype='float32')
        target_images = target_images_noise

        #=============select random start point
        latents_2 = misc.random_latents(num_init, Gs, random_state=None)
        Gs.change_input(inputs=latents_2)

        #==============define loss&optimizer
        regularizer = tf.abs(tf.norm(latents_2) - np.sqrt(512))
        loss = G_loss(G=Gs, target_images=target_images)  # + regularizer
        # init_var = OrderedDict([('Input/weight',Gs.trainables['Input/weight'])])
        # decayed_lr = tf.train.exponential_decay(0.1,500, 50, 0.5, staircase=True)
        G_opt = tfutil.Optimizer(name='latent_recovery', learning_rate=0.01)
        G_opt.register_gradients(loss, Gs.trainables)
        G_train_op = G_opt.apply_updates()

        #===========recovery==========
        EPOCH = 500
        losses = []
        losses.append(tfutil.run(loss))
        for i in range(EPOCH):
            G_opt.reset_optimizer_state()
            tfutil.run([G_train_op])

        ########
        learned_latent = tfutil.run(Gs.trainables['Input/weight'])
        result_images = Gs.run(learned_latent, labels, minibatch_size=config.num_gpus * 256,
                               num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.float32)

        sample_losses = []
        tmp_latents = tfutil.run(Gs.trainables['Input/weight'])
        G_out = Gs.get_output_for(tmp_latents, labels, is_training=True)
        G_out = rescale_output(G_out)
        for i in range(num_init):
            loss = tf.losses.mean_squared_error(target_images[i], G_out[i])
            sample_losses.append(tfutil.run(loss))

        #========save best optimized image
        plt.subplot(1, 2, 1)
        plt.imshow(tfutil.run(target_images)[0].transpose(1, 2, 0) / 255.0)
        plt.subplot(1, 2, 2)
        plt.imshow(result_images[np.argmin(sample_losses)].transpose(1, 2, 0) / 255.0)
        plt.savefig(result_dir)
        
        #========store optimized z
        z_recovered.append(tmp_latents)

        #=========save losses
#         loss=min(sample_losses)

        with open(out_dir+"/losses.txt","a") as f:
            for loss in sample_losses:
                f.write(str(loss)+' ')
            f.write('\n')
        np.save(out_dir+'/z_init',np.array(z_init))
        np.save(out_dir+'/z_re',np.array(z_recovered))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #------------------- training arguments -------------------
    parser.add_argument('--name', type=str, default='original')
    parser.add_argument('--pkl_path1', type=str, default='models/celeba_align_png_cropped_seed_v0.pkl')
    parser.add_argument('--pkl_path2', type=str, default='models/celeba_align_png_cropped_seed_v1.pkl') 
    parser.add_argument('--target_latents_dir', type=str, default='targets.npy') 
    parser.add_argument('--out_dir', type=str, default='recover_result3') 
    parser.add_argument('--num_init', type=int, default=16) 
    parser.add_argument('--num_total_sample', type=int, default=50) 
    parser.add_argument('--random_seed', type=int, default=5) 
    parser.add_argument('--noise_sigma', type=float, default=0)
    
    args = parser.parse_args()
    assert args.pkl_path1 != '' and args.pkl_path2 != ''
    misc.init_output_logging()
    np.random.seed(args.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)

    recovery(name=args.name, pkl_path1=args.pkl_path1,pkl_path2=args.pkl_path2,target_latents_dir=args.target_latents_dir, out_dir=args.out_dir,num_init=args.num_init, num_total_sample=args.num_total_sample,random_seed = args.random_seed,noise_sigma= args.noise_sigma)
