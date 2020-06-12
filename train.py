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


def triplet_loss(teacher,student,target,margin=0):
    
    
    TRIPLET = True
    CONTRASTIVE = False
    
    
    if TRIPLET:##for triplet loss_RE
    
      max_val_t = (-teacher).clamp(min=0)
      loss_t = teacher - teacher * target + max_val_t + ((-max_val_t).exp() + (-teacher - max_val_t).exp()).log()
      max_val_s = (-student).clamp(min=0)
      loss_s = student - student * target + max_val_s + ((-max_val_s).exp() + (-student - max_val_s).exp()).log()
      
      
      GT = Variable(torch.ones(teacher.size(0),1).cuda())
      
      
      ## for triplet_! -------------------------------(1)
      loss = nn.functional.margin_ranking_loss(loss_t,loss_s,GT,margin=margin)
      
      ## for triplet_RE ------------------------------(2)
      #loss_s2 = Variable(loss_s.data)
      #loss = nn.functional.margin_ranking_loss(loss_t,loss_s2,GT,margin=margin)
    
    
    if CONTRASTIVE:    ## for contrasive loss
      student2 = Variable(student.data)
      GG = torch.nn.functional.sigmoid(teacher)-torch.nn.functional.sigmoid(student2) # sigmoid(.) somhow works better than logsoftmax(.)
      
      loss = instance_bce_with_logits(GG, target)
      
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


'''
def train_multimodal_A(model,classifier_V,classifier_Q, train_loader, eval_loader, num_epochs, output,cycle):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    optim_D = torch.optim.Adamax(list(classifier_V.parameters())+list(classifier_Q.parameters()))#-------!!!!!!
    logger = utils.Logger(os.path.join(output, 'log_%d.txt'%cycle))
    best_eval_score = 0
    
    USE_semi_supervised = False#-------!!!!!
    USE_triplet = True#-----!!!!

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        ''
        batches = zip(train_loader,unlabeled_loader)#------!!! # batch loader for semi-supervised
        for i, ((v, b, q, a),(v_u, b_u, q_u, a_u)) in enumerate(batches):            
        v_input = Variable(torch.cat((v,v_u),0)).cuda()
        b_input = Variable(torch.cat((b,b_u),0)).cuda()
        q_input = Variable(torch.cat((q,q_u),0)).cuda()
        a_input = Variable(torch.cat((a,a_u),0)).cuda()
        a = Variable(a).cuda()
        
        pred_total,feat_v_total, feat_q_total = model(v_input, b_input, q_input, a_input)#----!!!!
        pred_all = pred_total[:v.size(0)]
        pred_all_u = pred_total[v.size(0):]
        feat_v = feat_v_total[:v.size(0)]
        feat_v_u = feat_v_total[v.size(0):]
        feat_q = feat_q_total[:v.size(0)]
        feat_q_u = feat_q_total[v.size(0):]
        ''
            
        #-----not semi-supervised
        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            
            pred_all,feat_v,feat_q = model(v, b, q, a)
            
            
            feat_v_D = Variable(feat_v.data).cuda()
            feat_q_D = Variable(feat_q.data).cuda()
            pred_v = classifier_V(feat_v_D)
            pred_q = classifier_Q(feat_q_D)
            
            
            if USE_semi_supervised:
              feat_v_uD = Variable(feat_v_u.data).cuda()
              feat_q_uD = Variable(feat_q_u.data).cuda()
              
              pred_v_u = classifier_V(feat_v_uD)
              pred_q_u = classifier_Q(feat_q_uD)
            
            ###########################---------------------------DD--------------------------------------#####################################
            loss_D = instance_bce_with_logits(pred_v, a) + instance_bce_with_logits(pred_q, a)#----!!!!!
            
            #loss = instance_bce_with_logits(pred_all, a) + instance_bce_with_logits(pred_v, a) #----- V only
            #loss = instance_bce_with_logits(pred_all, a) + instance_bce_with_logits(pred_q, a)#------- Q only
            
            
            
            temperature_t = 1
            temperature_s = 1
            teacher = Variable(torch.nn.functional.sigmoid(pred_all.data/temperature_t)).cuda()
            if USE_semi_supervised:
              teacher_u = Variable(torch.nn.functional.sigmoid(pred_all_u.data/temperature_t)).cuda()
              loss_distillation = ( instance_bce_with_logits(pred_v/temperature_s, teacher) + instance_bce_with_logits(pred_q/temperature_s,teacher) + instance_bce_with_logits(pred_v_u/temperature_s, teacher_u) + instance_bce_with_logits(pred_q_u/temperature_s,teacher_u))/4 
            else:
              #loss_distillation = ( instance_bce_with_logits(pred_v/teacher, teacher) + instance_bce_with_logits(pred_q/teacher,teacher) )/2   
              loss_distillation = (instance_bce_with_logits(pred_v/temperature_s, teacher) + instance_bce_with_logits(pred_q/temperature_s,teacher))/2   
              #loss_distillation = instance_bce_with_logits(pred_v/temperature_s, teacher) #-----V only
              #loss_distillation = instance_bce_with_logits(pred_q/temperature_s,teacher) #-----Q only
            #pdb.set_trace()
              
            loss_D += 0.1* loss_distillation
              
            loss_D.backward()
            ##nn.utils.clip_grad_norm(DD.parameters(), 0.25)
            optim_D.step()
            optim_D.zero_grad()
            ###################################################################################################################################
            
            
            
            
            
            
            
            
            ###########################---------------------------GG--------------------------------------#####################################
            
            
            
            
            loss = instance_bce_with_logits(pred_all, a)#----!!!!!
            
            
            
              
              
            if USE_triplet:
              loss_triplet = (triplet_loss(pred_all,pred_v,a,margin=0) + triplet_loss(pred_all,pred_q,a,margin=0))/2
              loss += 0.1* loss_triplet
                
                
                
              
            
            #pdb.set_trace()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()
            ###################################################################################################################################
            
            
            batch_score = compute_score_with_logits(pred_all, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, loss_D: %.2f, distillation loss: %.2f' % (total_loss,loss_D,loss_distillation))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            #best_params = model.state_dict()#------------!!!!!
    
    #model.load_state_dict(best_params)
    return best_eval_score#-----!!!!!
'''    
    
    


def train_multimodal(model, train_loader, eval_loader, num_epochs, output,cycle):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log_%d.txt'%cycle))#----!!!!
    best_eval_score = 0
    
    USE_self_distillation = True#----------!!!!!! fz
    USE_semi_supervised = False#-------!!!!!
    USE_triplet = False#-----!!!!

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        '''
        batches = zip(train_loader,unlabeled_loader)#------!!! # batch loader for semi-supervised
        for i, ((v, b, q, a),(v_u, b_u, q_u, a_u)) in enumerate(batches):            
        v_input = Variable(torch.cat((v,v_u),0)).cuda()
        b_input = Variable(torch.cat((b,b_u),0)).cuda()
        q_input = Variable(torch.cat((q,q_u),0)).cuda()
        a_input = Variable(torch.cat((a,a_u),0)).cuda()
        a = Variable(a).cuda()
        
        pred_total,feat_v_total, feat_q_total = model(v_input, b_input, q_input, a_input)#----!!!!
        pred_all = pred_total[:v.size(0)]
        pred_all_u = pred_total[v.size(0):]
        feat_v = feat_v_total[:v.size(0)]
        feat_v_u = feat_v_total[v.size(0):]
        feat_q = feat_q_total[:v.size(0)]
        feat_q_u = feat_q_total[v.size(0):]
        '''
            
        #-----not semi-supervised
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
            
            if USE_semi_supervised:
              feat_v_u = Variable(feat_v_u.data).cuda()
              feat_q_u = Variable(feat_q_u.data).cuda()
              
              pred_v_u = model.classifier_V(feat_v_u)
              pred_q_u = model.classifier_Q(feat_q_u)
            
            loss = instance_bce_with_logits(pred_all, a) + instance_bce_with_logits(pred_v, a) + instance_bce_with_logits(pred_q, a)#----!!!!!
            #loss = instance_bce_with_logits(pred_all, a) + instance_bce_with_logits(pred_v, a) #----- V only
            #loss = instance_bce_with_logits(pred_all, a) + instance_bce_with_logits(pred_q, a)#------- Q only
            
            if USE_self_distillation:#----------!!!!!!
              temperature_t = 1
              temperature_s = 1
              teacher = Variable(torch.nn.functional.sigmoid(pred_all.data/temperature_t)).cuda()
              if USE_semi_supervised:
                teacher_u = Variable(torch.nn.functional.sigmoid(pred_all_u.data/temperature_t)).cuda()
                loss_distillation = ( instance_bce_with_logits(pred_v/temperature_s, teacher) + instance_bce_with_logits(pred_q/temperature_s,teacher) + instance_bce_with_logits(pred_v_u/temperature_s, teacher_u) + instance_bce_with_logits(pred_q_u/temperature_s,teacher_u))/4 
              else:
                #loss_distillation = ( instance_bce_with_logits(pred_v/teacher, teacher) + instance_bce_with_logits(pred_q/teacher,teacher) )/2   
                loss_distillation = (instance_bce_with_logits(pred_v/temperature_s, teacher) + instance_bce_with_logits(pred_q/temperature_s,teacher))/2   
                #loss_distillation = instance_bce_with_logits(pred_v/temperature_s, teacher) #-----V only
                #loss_distillation = instance_bce_with_logits(pred_q/temperature_s,teacher) #-----Q only
              
              
              if USE_triplet:
                loss_triplet = (triplet_loss(pred_all,pred_v,a,margin=0) + triplet_loss(pred_all,pred_q,a,margin=0))/2
                loss += 0.1* loss_triplet
                
                
                
              #pdb.set_trace()
              loss += 0.1* loss_distillation
            
            #pdb.set_trace()
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
        if USE_triplet:
          logger.write('\ttrain_loss: %.2f, triplet loss: %.2f' % (total_loss, loss_triplet))
        elif USE_self_distillation:
          logger.write('\ttrain_loss: %.2f, distillation loss: %.2f' % (total_loss, loss_distillation))
        #else:
        #  logger.write('\ttrain_loss: %.2f' % (total_loss))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model-%d.pth'%cycle)
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
