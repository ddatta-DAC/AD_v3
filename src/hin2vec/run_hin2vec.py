import os
import numpy as np
import sys
sys.path.append('./..')
import argparse
from torch import LongTensor as LT
try:
    from .data import InputData
    from .hin2vec_model import Hin2Vec_model
except:
    from data import InputData
    from hin2vec_model import Hin2Vec_model

from torch.autograd import Variable
import torch
from tqdm import tqdm

class Hin2Vec():

    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension=32,
                 batch_size=12,
                 iteration=100000,
                 initial_lr=5e-4,
                 neg_sample_size = 1):
        self.output_file_name = output_file_name
        self.neg_sample_size = neg_sample_size
        self.data = InputData(batch_size, input_file_name)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iteration = iteration
        self.initial_lr = initial_lr

        print(emb_dimension)
        self.Hin2Vec_model = Hin2Vec_model(
            self.data.attr_size,
            self.data.rel_size,
            self.emb_dimension
        )
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.Hin2Vec_model.cuda()
        self.optimizer = torch.optim.Adam(self.Hin2Vec_model.parameters(), lr=self.initial_lr)

        # self.optimizer = optim.SGD(self.Hin2Vec_model.parameters(), lr=self.initial_lr)

    def train(self):
        """Multiple training.
        Returns:
            None.
        """
        batch_count = self.iteration
        process_bar = tqdm(range(int(self.iteration)))

        for i in process_bar:
            e1, e2, rel_ids, ground_truth  = self.data.generate_batch_sampling(self.neg_sample_size)

            e1 = Variable(LT(e1))
            e2 = Variable(LT(e2))
            rel_ids = Variable(LT(rel_ids))
            ground_truth = Variable(LT(ground_truth))


            if self.use_cuda:
                e1 = e1.cuda()
                e2 = e2.cuda()
                ground_truth = ground_truth.cuda()
                rel_ids = rel_ids.cuda()
            self.optimizer.zero_grad()
            loss = self.Hin2Vec_model.forward(e1, e2, rel_ids, ground_truth)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description(
                "Loss: %0.8f, lr: %0.6f" % (
                    loss.item() / self.batch_size,
                    self.optimizer.param_groups[0]['lr'])
            )

            if i * self.batch_size % 10000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

        # --- Save embedding ---- #
        self.Hin2Vec_model.save_embedding(
            self.output_file_name,
            use_cuda=self.use_cuda
        )

# ========================================= #

# ========================================= #
# Input format:
# node_id_1, node_id_2, relation_type, 1 [ground truth]
# ========================================= #

def exec(
        _dataset,
        input_file_name =None,
        output_file_name =None,
        model_use_data_DIR = None,
        emb_dimension=128,
        batch_size = 64,
        iteration = 100000
):
    if output_file_name is None:
        output_file_name = os.path.join(model_use_data_DIR, 'output_emb.npy')

    print(' Output file :: ', output_file_name )
    h2v = Hin2Vec(
        input_file_name=input_file_name,
        output_file_name=output_file_name,
        emb_dimension=emb_dimension
    )
    h2v.train()
    return output_file_name

