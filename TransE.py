import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_prepare import TrainSet, ValidSet, TestSet
from torch.utils.data import DataLoader


class TransE(nn.Module):
    def __init__(self, entity_num, relation_num, device, dim, d_norm, margin):
        super(TransE, self).__init__()
        self.dim = dim
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.d_norm = d_norm
        self.margin = torch.FloatTensor([margin]).to(device)
        self.device = device

        self.entity_embedding = nn.Embedding.from_pretrained(
            torch.empty(entity_num, self.dim).uniform_(-6/math.sqrt(self.dim), 6/math.sqrt(self.dim)), freeze=False)
        self.relation_embedding = nn.Embedding.from_pretrained(
            torch.empty(relation_num, self.dim).uniform_(-6/math.sqrt(self.dim), 6/math.sqrt(self.dim)), freeze=False)
        # l <= l / ||l||
        relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data/relation_norm

    def forward(self, pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail):
        """
        :param pos_head: [batch size]
        :param pos_relation: [batch size]
        :param pos_tail: [batch size]
        :param neg_head: [batch size]
        :param neg_relation: [batch size]
        :param neg_tail: [batch size]
        :return: triple loss
        """
        pos_distance = self.entity_embedding(pos_head) + \
                       self.relation_embedding(pos_relation) - \
                       self.entity_embedding(pos_tail)
        neg_distance = self.entity_embedding(neg_head) + \
                       self.relation_embedding(neg_relation) - \
                       self.entity_embedding(neg_tail)
        loss = torch.sum(F.relu(self.margin +
                                torch.norm(pos_distance, p=self.d_norm, dim=1) -
                                torch.norm(neg_distance, p=self.d_norm, dim=1)))
        return loss


    def link_predict(self, head, relation, tail, k=10):
        """
        link predict, Mean Rank, Hits@10
        :param head: [batch size]
        :param relation: [batch size]
        :param tail: [batch size]
        :param k: hits@k
        :return: rank number, hits number
        """
        # h_add_r: [batch size, embed size] -> [batch size, 1, embed size] -> [batch size, entity num, embed size]
        h_add_r = self.entity_embedding(head) + self.relation_embedding(relation)
        h_add_r = torch.unsqueeze(h_add_r, dim=1)
        h_add_r = h_add_r.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # embed_tail: [batch size, embed size] -> [batch size, entity num, embed size]
        embed_tail = self.entity_embedding.weight.data.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # tail
        tail = tail.view(-1, 1)
        # values: [batch size, k] scores, the smaller, the better
        # indices: [batch size, k] indices of entities ranked by scores
        values, indices = torch.topk(torch.norm(h_add_r - embed_tail, dim=2), k=self.entity_num, dim=1, largest=False)
        rank_num = torch.sum(torch.eq(indices, tail).nonzero(), dim=0)[1].item()
        hits_num = torch.sum(torch.eq(indices[:, :k], tail)).item()
        return rank_num, hits_num


if __name__ == '__main__':
    train_set = TrainSet()
    valid_set = ValidSet()
    valid_set.convert_to_index(train_set.entity_dic, train_set.relation_dic)
    test_set = TestSet()
    test_set.convert_to_index(train_set.entity_dic, train_set.relation_dic)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
    for batch_idx, (pos, neg) in enumerate(train_loader):
        print(pos.shape)
        break
