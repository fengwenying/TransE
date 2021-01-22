import torch
from torch import optim
from torch.utils.data import DataLoader
from TransE import TransE
from data_prepare import TrainSet, ValidSet, TestSet


class Config:
    def __init__(self):
        self.device = torch.device('cuda: 2')
        self.embed_dim = 50
        self.epoch_num = 50
        self.batch_size = 1240
        self.lr = 0.01
        self.margin = 1.0
        self.d_norm = 2
        self.top_k = 10
        self.valid_epoch = 5


class Runner:
    def __init__(self, config):
        self.config = config

        # prepare dataset
        self.train_set = TrainSet()
        self.valid_set = ValidSet()
        self.test_set = TestSet()
        self.valid_set.convert_to_index(self.train_set.entity_dic, self.train_set.relation_dic)
        self.test_set.convert_to_index(self.train_set.entity_dic, self.train_set.relation_dic)
        self.train_loader = DataLoader(self.train_set, batch_size=self.config.batch_size, num_workers=4, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.config.batch_size, num_workers=4, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.config.batch_size, num_workers=4, shuffle=True)

        # prepare model
        self.model = TransE(entity_num=self.train_set.entity_num,
                            relation_num=self.train_set.relation_num,
                            device=self.config.device,
                            dim=self.config.embed_dim,
                            d_norm=self.config.d_norm,
                            margin=self.config.margin).to(self.config.device)

        self.best_mean_rank = float("inf")

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        for epoch in range(self.config.epoch_num):
            # e <= e / ||e||
            entity_norm = torch.norm(self.model.entity_embedding.weight.data, dim=1, keepdim=True)
            self.model.entity_embedding.weight.data = self.model.entity_embedding.weight.data / entity_norm
            total_loss = 0

            for (pos, neg) in self.train_loader:
                pos, neg = pos.long().to(self.config.device), neg.long().to(self.config.device)
                # pos, neg: [batch size, 3] => [3, batch size]
                pos = torch.transpose(pos, 0, 1)
                neg = torch.transpose(neg, 0, 1)
                pos_head, pos_relation, pos_tail = pos[0], pos[1], pos[2]
                neg_head, neg_relation, neg_tail = neg[0], neg[1], neg[2]
                loss = self.model(pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch {}, loss = {:.4}'.format(epoch + 1, total_loss / len(self.train_set)))

            if (epoch + 1) % self.config.valid_epoch == 0:
                self.validate()

    def validate(self):
        rank, hits = 0, 0
        for data in self.valid_loader:
            data = data.long().to(self.config.device)
            data = torch.transpose(data, 0, 1)
            batch_rank, batch_hits = self.model.link_predict(data[0], data[1], data[2], k=self.config.top_k)
            rank += batch_rank
            hits += batch_hits
        print('Validate: mean rank = {}, hits@{} = {:.4f}'.format(int(rank / len(self.valid_set)),
                                                                  self.config.top_k, hits / len(self.valid_set)))
        if rank < self.best_mean_rank:
            self.best_mean_rank = rank
            torch.save(self.model.state_dict(), './model.param')

    def test(self):
        rank, hits = 0, 0
        self.model.load_state_dict(torch.load('./model.param'))
        for data in self.test_loader:
            data = data.long().to(self.config.device)
            data = torch.transpose(data, 0, 1)
            batch_rank, batch_hits = self.model.link_predict(data[0], data[1], data[2], k=self.config.top_k)
            rank += batch_rank
            hits += batch_hits
        print('Test: mean rank = {}, hits@{} = {:.4f}'.format(int(rank / len(self.test_set)),
                                                              self.config.top_k, hits / len(self.test_set)))


if __name__ == '__main__':
    config = Config()
    runner = Runner(config)
    runner.train()
    runner.test()
