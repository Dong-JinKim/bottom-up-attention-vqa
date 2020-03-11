import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train
import utils

import random #------!!!!!
from torch.utils.data.sampler import SubsetRandomSampler#----!!!!
import pdb
import os

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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

def get_uncertainty(model, unlabeled_loader,SUBSET):#-------------!!!!!!!
    print("Find samples to label....")
    #uncertainty = torch.Tensor(range(SUBSET)).cuda()
    uncertainty = torch.Tensor(SUBSET).cuda()
    '''
    models.eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    '''
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
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model = nn.DataParallel(model).cuda()
    
    
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
    
    train_loader = DataLoader(train_dset, batch_size, num_workers=4,sampler=SubsetRandomSampler(labeled_set))
    #train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4)
    
    #pdb.set_trace()
    
    ######################################### -------- Modified for active learning setup ------- ########################
    logger = utils.Logger(os.path.join(args.output, 'scores.txt'))
        
    for cycle in range(CYCLE): 
      print("##########CYCLE (%d/%d)###########"%(cycle,CYCLE))
      score = train(model, train_loader, eval_loader, args.epochs, args.output,cycle)
      logger.write('\teval score [CYCLE %d] : %.2f' % (cycle, 100 * score))
 
      ##
      #  Update the labeled dataset via loss prediction-based uncertainty measurement

      # Randomly sample 10000 unlabeled data points
      random.shuffle(unlabeled_set)
      subset = unlabeled_set[:SUBSET]

      # Create unlabeled dataloader for the unlabeled subset
      unlabeled_loader =  DataLoader(train_dset, batch_size, num_workers=4,sampler=SubsetSequentialSampler(subset))
      

      # Measure uncertainty of each data points in the subset
      #uncertainty = get_uncertainty(model, unlabeled_loader,SUBSET)
      uncertainty = get_uncertainty(model, unlabeled_loader,len(subset))

      # Index in ascending order
      arg = np.argsort(uncertainty)
            
      # Update the labeled dataset and the unlabeled dataset, respectively
      #pdb.set_trace()
      labeled_set += [dd for dd in torch.LongTensor(subset)[arg][-ADDENDUM:].numpy()] #list(torch.Tensor(subset)[arg][-ADDENDUM:].numpy())
      unlabeled_set = [dd for dd in list(torch.LongTensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]]

      # Create a new dataloader for the updated labeled dataset
      train_loader = DataLoader(train_dset, batch_size, num_workers=4,sampler=SubsetRandomSampler(labeled_set))
    
