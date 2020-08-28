import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet

class LL4ALModel(nn.Module):#------!!!!!
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier_LL,classifier_All):
        super(LL4ALModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier_LL = classifier_LL#-----!!!!
        self.classifier_All = classifier_All#----!!!!

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits_all = self.classifier_All(joint_repr)#----!!!!
        #logits_v = self.classifier_V(v_repr)#----!!!!
        #logits_q = self.classifier_Q(q_repr)#----!!!!
        
        
        return logits_all,v.sum(1),q_repr, joint_repr
        
         
class MultiModalModel(nn.Module):#------!!!!!
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier_V,classifier_Q,classifier_All):
        super(MultiModalModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier_V = classifier_V#-----!!!!
        self.classifier_Q = classifier_Q#-----!!!!
        self.classifier_All = classifier_All#----!!!!

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits_all = self.classifier_All(joint_repr)#----!!!!
        #logits_v = self.classifier_V(v_repr)#----!!!!
        #logits_q = self.classifier_Q(q_repr)#----!!!!
        
        
        #return logits_all,v_repr,q_repr
        return logits_all,v.sum(1),q_repr 
        
        
        
class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits,joint_repr


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)
    
def build_multimodal_newatt(dataset, num_hid):#---------------------------------!!!!!!
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier_V = SimpleClassifier(
        2048, num_hid * 2, dataset.num_ans_candidates, 0.5)#-------!!!!!
    classifier_Q = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    classifier_All = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return MultiModalModel(w_emb, q_emb, v_att, q_net, v_net, classifier_V,classifier_Q,classifier_All)
    
def build_LL_newatt(dataset, num_hid):#---------------------------------!!!!!!
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier_LL = SimpleClassifier(#--------!!!!!
        num_hid*4, num_hid /8, 1, 0.5)#-----!!!!!
    classifier_All = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return LL4ALModel(w_emb, q_emb, v_att, q_net, v_net, classifier_LL,classifier_All)
    
def build_DD(dataset,num_hid):
    classifier_V = SimpleClassifier(
        2048, num_hid * 2, dataset.num_ans_candidates, 0.5)#-------!!!!!
    classifier_Q = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
        
    return classifier_V,classifier_Q
