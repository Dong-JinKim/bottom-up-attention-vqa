import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train_multimodal
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
    parser.add_argument('--epochs_active', type=int, default=20)#-----!!!
    parser.add_argument('--method', type=str, default='random',help='random / entropy / multimodal')#-----!!!
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='multimodal_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


def KL_div(P,Q):
    return (P*(P/Q).log()).sum(1)

def get_uncertainty(model, unlabeled_loader,SUBSET,cycle,args):#-------------!!!!!!!    
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
    
          pred_all,feat_v,feat_q = model(v, b, q, a)
          prob_all = torch.nn.functional.softmax(pred_all,dim=1)
          
          ##entropy baseline
          if args.method=='entropy':
            
            logprob = torch.nn.functional.log_softmax(pred_all,dim=1)
            entropy = ((-1)*prob_all*logprob).sum(1)
            uncertainty = torch.cat((uncertainty, entropy.data),0)
    
          ## multimodal baseline
          elif args.method=='multimodal':
            feat_v = Variable(feat_v.data,volatile=True).cuda()
            feat_q = Variable(feat_q.data,volatile=True).cuda()
          
            pred_v = model.classifier_V(feat_v)
            pred_q = model.classifier_Q(feat_q)
            
            prob_v = torch.nn.functional.softmax(pred_v,dim=1)
            prob_q = torch.nn.functional.softmax(pred_q,dim=1)
            
            #distance1  = KL_div(prob_all,prob_v)#----(1)
            #distance1  = KL_div(prob_all,prob_q)#----(2)
            #distance1  = (KL_div(prob_v,prob_q) + KL_div(prob_q,prob_v))/2#----(3_RE)
            distance1  = (KL_div(prob_v,(prob_q+prob_v)/2) + KL_div(prob_q,(prob_q+prob_v)/2))/2#----(3_REAL)
            #distance  = KL_div(prob_all,prob_q)-KL_div(prob_all,prob_v)   #----(4)
            #distance  = KL_div(prob_all,prob_v)-KL_div(prob_all,prob_q)  #----(5)
            #distance1  = (-1)*torch.abs(KL_div(prob_all,prob_q)-KL_div(prob_all,prob_v))   #----(6_RE)
            #distance  = KL_div(prob_all,prob_v)*KL_div(prob_all,prob_q)   #----(1*2)
            #distance  = KL_div(prob_all,prob_v)+KL_div(prob_all,prob_q)   #----(1+2)
            #distance  = torch.max(KL_div(prob_all,prob_v),KL_div(prob_all,prob_q))   #----(max12)
            
            
            entropy_all = ((-1)*prob_all*torch.nn.functional.log_softmax(pred_all,dim=1)).sum(1)
            
            entropy_v = ((-1)*prob_v*torch.nn.functional.log_softmax(pred_v,dim=1)).sum(1)
            #distance1 = (entropy_all - entropy_v) #----------------------------------------------------(7)
            
            entropy_q = ((-1)*prob_q*torch.nn.functional.log_softmax(pred_q,dim=1)).sum(1)
            #distance1 = (entropy_all - entropy_q)*(-1) #----------------------------------------------------(8_RE)
            
            #distance = torch.abs(entropy_v - entropy_q) #---------(9)
            
            
            
            distance2 = entropy_all * 10 #----------------------want to add entropy as well?           
            
            
            
            #distance1 = entropy_v#------- entropy of V
            #distance1 = entropy_q#------- entropy of Q
            
            
            
            
            
            #distance = entropy_all
            distance = distance1+distance2
            #distance = distance1+distance2+entropy_all * 10
            #distance = distance1*distance2
            #distance = distance1*distance2*entropy_all
            #distance = torch.max(distance1,distance2)
            
            
            
            distance = (10-cycle) * entropy_v + cycle * distance  # Linear scheduling from multimodalV to other distance.
            
            
            
            uncertainty = torch.cat((uncertainty, distance.data),0)
            #pdb.set_trace()
          else:
            pdb.set_trace()
    
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
    
    constructor = 'build_%s' % args.model
    
    
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
  
    train_loader = DataLoader(train_dset, batch_size, num_workers=2,sampler=SubsetRandomSampler(labeled_set))
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)
    
    
    ######################################### -------- Modified for active learning setup ------- ########################
    logger = utils.Logger(os.path.join(args.output, 'scores.txt'))
        
    for cycle in range(CYCLE): 
      print("##########CYCLE (%d/%d)###########"%(cycle,CYCLE))
      
      model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
      model.w_emb.init_embedding('data/glove6b_init_300d.npy')
      model = model.cuda()
      #model = nn.DataParallel(model).cuda()
    
      if cycle == 0:
        epochs = args.epochs_first
      else:
        epochs = args.epochs_active
      
      model.train()
      score = train_multimodal(model, train_loader, eval_loader, epochs, args.output,cycle)
      logger.write('eval score [CYCLE %d] : %.2f' % (cycle, 100 * score))
 
      ##
      #  Update the labeled dataset via loss prediction-based uncertainty measurement
      if cycle<CYCLE-1:
        # Randomly sample 10000 unlabeled data points
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:SUBSET]

        # Create unlabeled dataloader for the unlabeled subset
        #train_dset1 = torch.utils.data.Subset(train_dset,subset)#----------------!!!
        #unlabeled_loader =  DataLoader(train_dset1, batch_size*4, num_workers=1)#------!!!!
        unlabeled_loader =  DataLoader(train_dset, batch_size*4, num_workers=1,sampler=SubsetSequentialSampler(subset))
        
        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader,len(subset),cycle,args)
        
        # Index in ascending order
        arg = np.argsort(uncertainty)
              
        # Update the labeled dataset and the unlabeled dataset, respectivelypy()] 
        labeled_set += [dd for dd in torch.LongTensor(subset)[arg][-ADDENDUM:].numpy()] #list(torch.Tensor(subset)[arg][-ADDENDUM:].numpy())
        unlabeled_set = [dd for dd in list(torch.LongTensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]]

        # Create a new dataloader for the updated labeled dataset
        train_loader = DataLoader(train_dset, batch_size, num_workers=2,sampler=SubsetRandomSampler(labeled_set))
    
