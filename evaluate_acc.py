import numpy as np
import argparse

def get_min_losses(losses):
    min_idx = np.argmin(losses[:,-1,:] ,axis=1)
    min_losses = []
    for i in range(len(losses)):
        losses_min = losses[i,-1,min_idx[i]]
        min_losses.append(losses_min)
    return np.array(min_losses)


def cal_accuracy(latents_dir, data_name, num_sample,Gs_set,Gt_set, epsilon):
    result = []
    truth=[]
    print(f'Gs:{Gs_set}')
    print(f'Gt:{Gt_set}')
    for i in Gt_set:
        temp = []
        for j in Gs_set:
            losses = np.load(f'{latents_dir}/{data_name}_{i}{j}_4init_0/l2.npy')[:num_sample]
            min_losses=get_min_losses(losses)
            temp.append(min_losses.tolist())
        temp = np.array(temp)
        min_idx = np.argmin(temp ,axis=0)
        for k in range(num_sample):
            if temp[min_idx[k],k]>epsilon:
                min_idx[k]=-1
        result.append(min_idx)
        if i in Gs_set:
            truth.append([i]*len(min_idx))
        else:
            truth.append([-1]*len(min_idx))
    result = np.array(result)
    truth = np.array(truth)
    return np.count_nonzero( result == truth)/(num_sample*len(Gt_set))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--latents_dir', type=str, default='recover_result')
    parser.add_argument('--num_sample', type=int, default=25)
    parser.add_argument('--epsilon', type=int, default=400)
    parser.add_argument('--data_name', type=str, default='celeba')
    parser.add_argument('--Gs_set', nargs='*', type=int, default = [0,1,2,3])
    parser.add_argument('--Gt_set', nargs='*', type=int, default = [0,1,2,3])

    args = parser.parse_args()

    accuracy = cal_accuracy(args.latents_dir, args.data_name, args.num_sample,args.Gs_set,args.Gt_set, args.epsilon)

    print(accuracy)
