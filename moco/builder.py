# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from main_moco import accuracy
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class MoCo(pl.LightningModule):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder,criterion, dim=128, K=65536, m=0.999, T=0.07, mlp=False,steps_per_epoch=1):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.save_hyperparameters()
        self.criterion=criterion
        self.K = K
        self.m = m
        self.T = T
        self.temperature = 0.0722
        self.base_temperature = 0.0722
        self.contrast_mode = 'all'
        self.steps_per_epoch= steps_per_epoch 
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
        self.register_buffer("queue_1", torch.randn(K, dim))
        self.queue_1 = nn.functional.normalize(self.queue_1, dim=1)
        self.register_buffer("queue_1_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_1_labels", torch.tensor([0,1,2,3],dtype=torch.long).repeat(K))
        self.queue_1_labels = self.queue_1_labels[:K]

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
        queue = self.queue_1
        queue_ptr = self.queue_1_ptr

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[ ptr:ptr + batch_size,:] = keys
        self.queue_1_labels[ptr:ptr + batch_size] =label[:keys.shape[0]]
        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr
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
        features = torch.cat([q,k,self.queue_1],dim=0)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        batch_size = q.shape[0]
        label = torch.cat([label,label,self.queue_1_labels],dim=0)
        labels = label.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[0]
        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count).mean()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, label)

        return loss ,logits,mask

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        images,label = batch  #
        loss,o,t= self.forward(images[0], images[1],label)
        # loss = self.criterion(o,t)

        acc1, acc5 = accuracy(o, t, topk=(1, 5))
        return loss,(acc1,acc5)

    def training_step(self, batch, batch_idx):
        loss,acc = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        self.log('acc1', acc[0])
        self.log('acc5', acc[1])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.05,steps_per_epoch=self.steps_per_epoch,epochs=self.trainer.max_epochs,div_factor=100)
        scheduler = {'scheduler':scheduler,'interval':'step','monitor':'val_loss'}
        return [optimizer] , [scheduler]

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
