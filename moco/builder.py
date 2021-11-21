# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            try:
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            except:
                dim_mlp = self.encoder_q.head.weight.shape[1]
                self.encoder_q.head= nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.head)
                self.encoder_k.head= nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.head)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # Creating different queues for the clusters 
        self.register_buffer("queue_1", torch.randn(dim, K))
        self.queue_1 = nn.functional.normalize(self.queue_1, dim=0)
        self.register_buffer("queue_1_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_2", torch.randn(dim, K))
        self.queue_2 = nn.functional.normalize(self.queue_2, dim=0)
        self.register_buffer("queue_2_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_3", torch.randn(dim, K))
        self.queue_3 = nn.functional.normalize(self.queue_3 ,dim=0)
        self.register_buffer("queue_3_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_4", torch.randn(dim, K))
        self.queue_4 = nn.functional.normalize(self.queue_4, dim=0)
        self.register_buffer("queue_4_ptr", torch.zeros(1, dtype=torch.long))

        self.queues = torch.stack([self.queue_1,self.queue_2,self.queue_3,self.queue_4],dim=0)
        self.queues_ptr = torch.stack([self.queue_1_ptr,self.queue_2_ptr,self.queue_3_ptr,self.queue_4_ptr],dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys,label):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        for idx,k in enumerate(keys):
            # print(self.queues[label[idx]][:,int(self.queues_ptr[label[idx]]):int(self.queues_ptr[label[idx]]) +1 ].shape,k.shape)
            self.queues[label[idx]][:,int(self.queues_ptr[label[idx]])] = k.T
            self.queues_ptr[label[idx]] = (self.queues_ptr[label[idx]] + 1) % self.K
            # print(self.queues_ptr)
        # queue = self.queues[label]
        # queue_ptr = self.queues_ptr[label]

        # batch_size = keys.shape[0]

        # ptr = int(queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # # replace the keys at ptr (dequeue and enqueue)
        # queue[:, ptr:ptr + batch_size] = keys.T
        # ptr = (ptr + batch_size) % self.K  # move pointer

        # queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k,label):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.Tensor((q.shape[0]),self.queues[0].shape[1]).cuda()
        for idx,n in enumerate(q):
            te = n.unsqueeze(0)
            idx_other = (label[idx]-1)%4
            # print(label[idx],' to ',idx_other)
            qq = self.queues[idx_other].clone().detach()
            l_p= torch.einsum('nc,ck->nk',[te.cuda(),qq.cuda()]).cuda()
            # print(l_p.shape)
            l_neg[idx]=l_p
            # print(l_neg.shape)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        try:
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        except:
            labels = torch.zeros(logits.shape[0], dtype=torch.long)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k,label)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
