import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import cPickle
from dataset import Dictionary, VQAFeatureDataset
import base_model
import utils

from matplotlib import pyplot as plt

import random #------!!!!!
import pdb
import os
from torch.autograd import Variable#----!!!!

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='multimodal_newatt')
    parser.add_argument('--input', type=str, default='saved_models/multimodalV_to_3_REAL+entropy+distillation_try2/model-9.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size    
    
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = model.cuda()
    #model = nn.DataParallel(model).cuda()
    
    
    
    question = json.load(open('data/v2_OpenEnded_mscoco_val2014_questions.json'))                                                                      
    id2question = {qq['question_id']:qq['question']  for qq in question['questions']}
    
    
    label2ans = cPickle.load(open('data/cache/trainval_label2ans.pkl', 'rb'))
    
    model_path = args.input#os.path.join(args.input, 'model.pth')
    model.load_state_dict(torch.load(model_path))
  
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)


    model.train(False)
    score = 0
    upper_bound = 0
    num_data = 0
    for ii, (v, b, q, a,iid,qid) in enumerate(iter(eval_loader)):
        if ii<0 or not (iid[0]  in [328, 397,520,536,544  ,632,711]):#,]):
          continue
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred,_,_ = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        num_data += pred.size(0)
        
        
        
        
        
        image_path = 'data/val2014/COCO_val2014_%012d.jpg'%iid
        im = plt.imread(image_path)
        
        
        
        #qqq = AA[qid]
        _,idx=pred[0].sort(descending=True)
        _,idx_GT=a.max(1)
        #dictionary.idx2word[idx.data[0]]
        
        #pdb.set_trace()
        print('image name: val2014/COCO_val2014_%012d.jpg'%iid)
        #print('Question: %s\n\nGT:%s\t\t \nTOP1:%s//\nTOP2:%s//\nTOP3:%s\n'%(id2question[qid[0]],label2ans[idx_GT[0]], label2ans[idx.data[1]], dictionary.idx2word[idx.data[0]], label2ans[idx.data[2]]))
        print('Question: %s\n\nGT:%s\t\t \nTOP1:%s//\n'%(id2question[qid[0]],label2ans[idx_GT[0]], label2ans[idx.data[1]]))
        
        plt.imshow(im)
        plt.show()
        
        #pdb.set_trace()
        print('')

    score = score / len(eval_loader.dataset)  
    logger.write('\teval score: %.2f (%.2f)' % (100 * score))
      
