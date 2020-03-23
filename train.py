import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import pdb

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader, num_epochs, output,cycle):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log_%d.txt'%cycle))#----!!!!
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred,_ = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        #logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            best_params = model.state_dict()#------------!!!!!
    
    model.load_state_dict(best_params)
    return best_eval_score#-----!!!!!

def train_multimodal(model, train_loader, eval_loader, num_epochs, output,cycle):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log_%d.txt'%cycle))#----!!!!
    best_eval_score = 0
    
    USE_self_distillation = True#----------!!!!!!

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            
            pred_all,feat_v,feat_q = model(v, b, q, a)
            
            feat_v = Variable(feat_v.data).cuda()
            feat_q = Variable(feat_q.data).cuda()
            
            pred_v = model.classifier_V(feat_v)
            pred_q = model.classifier_Q(feat_q)
            
            loss = instance_bce_with_logits(pred_all, a) + instance_bce_with_logits(pred_v, a) + instance_bce_with_logits(pred_q, a)#----!!!!!
            
            if USE_self_distillation:#----------!!!!!!
              temperature = 1
              teacher = Variable(torch.nn.functional.sigmoid(pred_all.data/temperature)).cuda()
              
              loss_distillation = ( instance_bce_with_logits(pred_v/teacher, teacher) + instance_bce_with_logits(pred_q/teacher,teacher) )/2   
              loss += 0.1* loss_distillation
            
            
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred_all, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        #logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            #best_params = model.state_dict()#------------!!!!!
    
    #model.load_state_dict(best_params)
    return best_eval_score#-----!!!!!

def train_GAN(model, DD,  train_loader, unlabeled_loader, eval_loader, num_epochs, output,cycle):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    optim_D = torch.optim.Adamax(DD.parameters())#------!!!!
    logger = utils.Logger(os.path.join(output, 'log_%d.txt'%cycle))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        G_loss = 0#------!!!!
        D_loss = 0#------!!!!
        train_score = 0
        t = time.time()
        
        
        
        
        
        batches = zip(train_loader,unlabeled_loader)#------!!!
        for i, ((v, b, q, a),(v_u, b_u, q_u, a_u)) in enumerate(batches):            
            v_input = Variable(torch.cat((v,v_u),0)).cuda()
            b_input = Variable(torch.cat((b,b_u),0)).cuda()
            q_input = Variable(torch.cat((q,q_u),0)).cuda()
            a_input = Variable(torch.cat((a,a_u),0)).cuda()
            a = Variable(a).cuda()
            
            pred_total,feat_total = model(v_input, b_input, q_input, a_input)#----!!!!
            pred = pred_total[:v.size(0)]
            feat = feat_total[:v.size(0)]
            feat_u = feat_total[v.size(0):]
            
            '''    
            for i, (v, b, q, a) in enumerate(train_loader):            
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred,feat = model(v, b, q, a)#----!!!!
            
            v_u, b_u, q_u, a_u = next(iter(unlabeled_loader))#-----!!!!!!
            v_u = Variable(v_u).cuda()
            b_u = Variable(b_u).cuda()
            q_u = Variable(q_u).cuda()
            a_u = Variable(a_u).cuda()
            _,feat_u = model(v_u, b_u, q_u, a_u)#----!!!!
            '''
            ############################------------- train D-------------#############################
            D_input = Variable(torch.cat((feat.data,feat_u.data),0))#Variable(feat_total.data)#
            D_target = Variable(torch.cat( (torch.ones(feat.size(0),1) , torch.zeros(feat_u.size(0),1) ),0)).cuda()
            GAN_loss_D = instance_bce_with_logits(DD(D_input), D_target)/2#---!!!
            loss_D =  GAN_loss_D * 0.1
            
            loss_D.backward()
            optim_D.step()
            optim_D.zero_grad()
            ############################################################################################
            
            
            loss = instance_bce_with_logits(pred, a)
            
            ############################------------- train G-------------#############################
            train_G=False
            if train_G:
              G_input = torch.cat((
                                feat,
                                feat_u,
                                ),0)
              G_target = Variable(
                                torch.cat( (
                                torch.zeros(feat.size(0),1) , 
                                torch.ones(feat_u.size(0),1) 
                                ),0)
                                ).cuda()
              GAN_loss_G  = instance_bce_with_logits(DD(G_input), G_target)/2
              loss += GAN_loss_G * 0.1
            else:
              GAN_loss_G = Variable(torch.Tensor([-1]))
            ############################################################################################
            
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            G_loss += GAN_loss_G.data[0] * v.size(0)#------!!!!!
            D_loss += GAN_loss_D.data[0] * v.size(0)#------!!!!!
            
            train_score += batch_score
            

        total_loss /= len(train_loader.dataset)
        G_loss /= len(train_loader.dataset)#------!!!!
        D_loss /= len(train_loader.dataset)#------!!!!
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\tD_loss: %.2f, G_loss: %.2f' % (D_loss, G_loss))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            #best_params = model.state_dict()
    
    #model.load_state_dict(best_params)
    return best_eval_score
    
    

def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred,_,_ = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
