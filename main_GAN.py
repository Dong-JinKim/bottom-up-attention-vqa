import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, train_GAN
import utils

import random #------!!!!!
from torch.utils.data.sampler import SubsetRandomSampler#----!!!!
import pdb
import os
from torch.autograd import Variable#----!!!!

class SubsetSequentialSampler():
    r"""Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs_first', type=int, default=20)
    parser.add_argument('--epochs_active', type=int, default=10)#-----!!!
    parser.add_argument('--method', type=str, default='gan',help='random / gan')#-----!!!
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


class Discriminator(nn.Module):
    def __init__(self, num_hid):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_hid, num_hid/4),#1024->256
            nn.ReLU(),
            nn.Linear(num_hid/4, num_hid/16),#256->64
            nn.ReLU(),
            nn.Linear(num_hid/16, 1),
            #nn.Sigmoid()
        )
    def forward(self, features):
        return self.net(features)
        
        
def get_uncertainty(model, unlabeled_loader,SUBSET,args):#-------------!!!!!!!    
    print("Find samples to label....")
    ##----random baseline
    if args.method=='random':
      uncertainty = torch.Tensor(SUBSET).cuda()
    else:
      model.eval()
      uncertainty = torch.Tensor([]).cuda()
      #with torch.no_grad():
      for _, (v, b, q, a) in enumerate(unlabeled_loader):
          v = Variable(v,volatile=True).cuda()
          b = Variable(b,volatile=True).cuda()
          q = Variable(q,volatile=True).cuda()
          a = Variable(a,volatile=True).cuda()
    
          _,feat = model(v, b, q, a)
          prob = DD(feat)*(-1)
          
          uncertainty = torch.cat((uncertainty, prob[:,0].data),0)
          #else:
          #  pdb.set_trace()
    
    return uncertainty.cpu()


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size    
    
    ############################################
    # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
    ADDENDUM=40000#3000
    CYCLE=10#10
    SUBSET=443757
    
    indices = list(range(443757))#list(range(NUM_TRAIN))
    random.shuffle(indices)
    labeled_set = indices[:ADDENDUM]
    unlabeled_set = indices[ADDENDUM:]
    #############################################
    
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    DD = Discriminator(args.num_hid).cuda()#-----!!!!!
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()
    DD = nn.DataParallel(DD).cuda()#-----!!!!
      
    train_loader = DataLoader(train_dset, batch_size, num_workers=2,sampler=SubsetRandomSampler(labeled_set))
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)
    
    
    ######################################### -------- Modified for active learning setup ------- ########################
    logger = utils.Logger(os.path.join(args.output, 'scores.txt'))
        
    for cycle in range(CYCLE): 
      print("##########CYCLE (%d/%d)###########"%(cycle,CYCLE))
      
      if cycle == 0:
        epochs = args.epochs_first
      else:
        epochs = args.epochs_active

      # Randomly sample 10000 unlabeled data points
      random.shuffle(unlabeled_set)#-----!!!!!
      subset = unlabeled_set[:SUBSET]#-----!!!!!

      # Create unlabeled dataloader for the unlabeled subset
      #pdb.set_trace()
      unlabeled_loader =  DataLoader(train_dset, batch_size, num_workers=1,
                            sampler=SubsetRandomSampler(random.sample(unlabeled_set,  min(len(labeled_set),len(unlabeled_set))   )))     
        
        
      model.train()
      score = train_GAN(model,DD, train_loader, unlabeled_loader, eval_loader, epochs, args.output,cycle)#----!!!!
      logger.write('eval score [CYCLE %d] : %.2f' % (cycle, 100 * score))
 
      ##
      #  Update the labeled dataset via loss prediction-based uncertainty measurement
      if cycle<CYCLE-1:
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader =  DataLoader(train_dset, batch_size*4, num_workers=1,sampler=SubsetSequentialSampler(subset))# non shuffle version
        
        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader,len(subset),args)

        # Index in ascending order
        arg = np.argsort(uncertainty)
              
        # Update the labeled dataset and the unlabeled dataset, respectivelypy()] #list(torch.Tensor(subset)[arg][-ADDENDUM:].numpy())
        labeled_set += [dd for dd in torch.LongTensor(subset)[arg][-ADDENDUM:].numpy()] #list(torch.Tensor(subset)[arg][-ADDENDUM:].numpy())
        unlabeled_set = [dd for dd in list(torch.LongTensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]]

        # Create a new dataloader for the updated labeled dataset
        train_loader = DataLoader(train_dset, batch_size, num_workers=2,sampler=SubsetRandomSampler(labeled_set))
    
