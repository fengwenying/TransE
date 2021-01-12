import random
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset


class TrainSet(Dataset):
    def __init__(self):
        super(TrainSet, self).__init__()
        self.raw_data, self.entity_dic, self.relation_dic = self.load_data()
        self.triple_num = self.raw_data.shape[0]
        self.entity_num, self.relation_num = len(self.entity_dic), len(self.relation_dic)
        print('Train set: {} triplets, {} entities, {} relations'.format(self.triple_num, self.entity_num, self.relation_num))

        self.pos_data = self.convert_to_index(self.raw_data)
        self.related_dic = self.get_related_entity()
        self.neg_data = self.generate_neg()

    def __len__(self):
        return self.triple_num

    def __getitem__(self, item):
        return [self.pos_data[item], self.neg_data[item]]

    def load_data(self):
        raw_data = pd.read_csv(filepath_or_buffer='./FB15k/freebase_mtr100_mte100-train.txt',
                               sep='\t',
                               header=None,
                               names=['head', 'relation', 'tail'],
                               keep_default_na=False,
                               encoding='utf-8')
        raw_data = raw_data.applymap(lambda x: str(x).strip())
        head_count = Counter(raw_data['head'])
        tail_count = Counter(raw_data['tail'])
        relation_count = Counter(raw_data['relation'])
        entity_list = list((head_count + tail_count).keys())
        relation_list = list(relation_count.keys())
        entity_dic = dict([(item, idx) for idx, item in enumerate(entity_list)])
        relation_dic = dict([(item, idx) for idx, item in enumerate(relation_list)])
        return raw_data.values, entity_dic, relation_dic

    def convert_to_index(self, data):
        return np.array([[self.entity_dic[triple[0]],
                          self.relation_dic[triple[1]],
                          self.entity_dic[triple[2]]] for triple in data])

    def get_related_entity(self):
        related_dic = {}
        for triple in self.pos_data:
            if related_dic.get(triple[0]) is None:
                related_dic[triple[0]] = {triple[2]}
            else:
                related_dic[triple[0]].add(triple[2])
            if related_dic.get(triple[2]) is None:
                related_dic[triple[2]] = {triple[0]}
            else:
                related_dic[triple[2]].add(triple[0])
        return related_dic

    def generate_neg(self):
        i = 0
        entity_candidates = []
        neg_data = []
        for triple in self.pos_data:
            while True:
                if i == len(entity_candidates):
                    i = 0
                    entity_candidates = random.choices(population=list(range(self.entity_num)), k=self.entity_num)
                neg_candidate, i = entity_candidates[i], i+1
                if random.randint(0, 1) == 0:
                    # replace head
                    if neg_candidate not in self.related_dic[triple[2]]:
                        neg_data.append([neg_candidate, triple[1], triple[2]])
                        break
                else:
                    # replace tail
                    if neg_candidate not in self.related_dic[triple[0]]:
                        neg_data.append([triple[0], triple[1], neg_candidate])
                        break
        return np.array(neg_data)


class ValidSet(Dataset):
    def __init__(self):
        super(ValidSet, self).__init__()
        self.raw_data = self.load_data()
        self.triple_num = self.raw_data.shape[0]
        print('Valid set: {} triplets'.format(self.triple_num))

    def __len__(self):
        return self.triple_num

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        raw_data = pd.read_csv(filepath_or_buffer='./FB15k/freebase_mtr100_mte100-valid.txt',
                               sep='\t',
                               header=None,
                               names=['head', 'relation', 'tail'],
                               keep_default_na=False,
                               encoding='utf-8')
        raw_data = raw_data.applymap(lambda x: str(x).strip())
        return raw_data.values

    def convert_to_index(self, entity_dic, relation_dic):
        self.data = np.array([[entity_dic[triple[0]],
                               relation_dic[triple[1]],
                               entity_dic[triple[2]]] for triple in self.raw_data])


class TestSet(Dataset):
    def __init__(self):
        super(TestSet, self).__init__()
        self.raw_data = self.load_data()
        self.triple_num = self.raw_data.shape[0]
        print('Test set: {} triplets'.format(self.triple_num))

    def __len__(self):
        return self.triple_num

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        raw_data = pd.read_csv(filepath_or_buffer='./FB15k/freebase_mtr100_mte100-test.txt',
                               sep='\t',
                               header=None,
                               names=['head', 'relation', 'tail'],
                               keep_default_na=False,
                               encoding='utf-8')
        raw_data = raw_data.applymap(lambda x: str(x).strip())
        return raw_data.values

    def convert_to_index(self, entity_dic, relation_dic):
        self.data = np.array([[entity_dic[triple[0]],
                               relation_dic[triple[1]],
                               entity_dic[triple[2]]] for triple in self.raw_data])


if __name__ == '__main__':
    train_set = TrainSet()
    valid_set = ValidSet()
    valid_set.convert_to_index(train_set.entity_dic, train_set.relation_dic)
    print(valid_set.data[5])
    test_set = TestSet()
    test_set.convert_to_index(train_set.entity_dic, train_set.relation_dic)
    print(test_set.data[5])
