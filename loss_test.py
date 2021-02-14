# +
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import projector
import misc as _misc
from training import dataset
# from training import misc 

import tfutil
import config

import matplotlib.pyplot as plt
import argparse
import os


# +
def noise(src, sigma=0.1):
    gauss = np.random.normal(0, sigma*2, src[0].shape)
    noisy = src + gauss
    return noisy

def blur(src, k=5):#(N,C,H,W)
    import cv2
    k = int(k)
    kernel = np.ones(shape=(k,k),dtype=np.float32)/(k*k)
    result = []
    for img in src:
        img=(img.transpose(1,2,0)+1)/2
        dst = cv2.filter2D((img),-1,kernel)
        img = dst.transpose(2,0,1)*2-1
        result.append(img)
    return np.array(result)

def center_crop(src,size=64):
    size = int(size)
    assert size % 2 == 0 and size<src.shape[-1]
    l = src.shape[-1]//2 - size//2
    b = src.shape[-1]//2 + size//2
    croped = np.ones(src.shape)*-1
    croped[:,:,l:b,l:b] = src[:,:,l:b,l:b]
    return croped

def jpeg(src,Quality=100):
    import cv2
    Quality = int(Quality)
    imgs = []
    for img in src:
        img=(img.transpose(1,2,0)+1)*127.5
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        img =cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
        img = img.transpose(2,0,1)/127.5-1
        imgs.append(img)
    return np.array(imgs)



# -

def recovery(name,pkl_path1,pkl_path2, out_dir, target_latents_dir, \
             num_init=4, num_total_sample=50, minibatch_size = 1, attack=None, denoiseing=False, param=None, loss_func='lpips'):

    print(name)
    print('num_init:'+str(num_init))
    print(f'num_sample:{num_total_sample}')
    print(f'loss_func:{loss_func}')
    
    # load sorce model
    print('Loading network1...'+pkl_path1)
    _, _, Gs = _misc.load_network_pkl(pkl_path1)
    
    # load target model
    print('Loading  network2...'+pkl_path2)
    _, _, Gt = _misc.load_network_pkl(pkl_path2)
    
    proj = projector.Projector( loss_func = loss_func, crop_size=param)
    proj.set_network(Gs, minibatch_size=num_init)
    
    out_dir = os.path.join(out_dir,name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        z_init = []
        l2_log = []
        lpips_log = []
        latents_log = []
    else:     
        z_init = np.load(os.path.join(out_dir,'z_init.npy'))
        z_init = [z_init[i] for i in range(len(z_init))]
        l2_log = [np.load(os.path.join(out_dir,'l2.npy'))[i] for i in range (len(z_init))]
        lpips_log = [np.load(os.path.join(out_dir,'lpips.npy'))[i] for i in range (len(z_init)) ]
        latents_log = [np.load(os.path.join(out_dir,'z_re.npy'))[i] for i in range (len(z_init)) ]
        
        
    #load target z
    assert os.path.exists(target_latents_dir) , 'latent_dir not exisit'
    print('using latents:'+target_latents_dir)
    pre_latents = np.load(target_latents_dir)
    
    start_time = time.time()
    for k in range(len(z_init),num_total_sample):
        #=======sample target image=====
        latent = pre_latents[k]
        z_init.append(latent)
        
        latents = np.zeros((num_init, len(latent)), np.float32)
        latents[:] = latent
        labels = np.zeros([latents.shape[0], 0], np.float32)
        
        target_images = Gt.get_output_for(latents, labels, is_training=False)
        target_images = tfutil.run(target_images)

        #================attack
        if attack is not None:
            target_images = attack(target_images, param)
        if denoiseing:
            target_images = blur(target_images, 3)


        #===========recovery==========
        l2_dists = []
        lpips_dists = []
        learned_latents=[]
        proj.start(target_images)
        while proj.get_cur_step() < proj.num_steps:
            l2_dists.append(proj.get_l2())
            lpips_dists.append(proj.get_lpips())
            learned_latents.append(proj.get_dlatents())
            proj.step()
        print('epoch:\r%d / %d ... %12f  %12f ' % (k, num_total_sample, np.min(proj.get_l2()), np.min(proj.get_lpips()), time.time()))
        
        l2_log.append(l2_dists)
        lpips_log.append(lpips_dists)
        latents_log.append(learned_latents)

        np.save(out_dir+'/l2', np.array(l2_log))
        np.save(out_dir+'/lpips', np.array(lpips_log))
        np.save(out_dir+'/z_init',np.array(z_init))
        np.save(out_dir+'/z_re',np.array(latents_log))

# +
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #------------------- training arguments -------------------
    parser.add_argument('--name', type=str, default='original')
    parser.add_argument('--pkl_path1', type=str, default='models/celeba_align_png_cropped_seed_v0.pkl')
    parser.add_argument('--pkl_path2', type=str, default='models/celeba_align_png_cropped_seed_v1.pkl') 
    parser.add_argument('--target_latents_dir', type=str, default='targets.npy') 
    parser.add_argument('--out_dir', type=str, default='recover_result_lpips') 
    parser.add_argument('--num_init', type=int, default=4) 
    parser.add_argument('--num_total_sample', type=int, default=50) 
    parser.add_argument('--random_seed', type=int, default=5) 
    parser.add_argument('--attack', type=str, default=None) 
    parser.add_argument('--param', type=float, default=None)
    parser.add_argument('--loss', type=str, default='lpips')
    
    args = parser.parse_args()
    assert args.pkl_path1 != '' and args.pkl_path2 != ''
    _misc.init_output_logging()
    np.random.seed(args.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    attack_type = args.attack
    param = args.param
    
    if attack_type == 'noise':
        assert param<1,'sigma should <1'
        print(f'attack:{attack_type},param{param}')
        attack = noise
    elif attack_type == 'blur':
        assert param>0,'kernel should >0'
        print(f'attack:{attack_type},param{param}')
        attack = blur
    elif attack_type == 'centor_crop':
        assert param<128
        print(f'attack:{attack_type},param{param}')
        attack = center_crop
    elif attack_type == 'jpeg':
        param
        assert param<100
        print(f'attack:{attack_type},param{param}')
        attack = jpeg
    elif attack_type == None:
        print('no attack')
        attack = None
    else:
        raise Exception("attack type error")

    recovery(name=args.name, pkl_path1=args.pkl_path1,pkl_path2=args.pkl_path2,\
             target_latents_dir=args.target_latents_dir, out_dir=args.out_dir,num_init=args.num_init, \
             num_total_sample=args.num_total_sample, attack=attack, param = args.param, loss_func=args.loss)
    
    
