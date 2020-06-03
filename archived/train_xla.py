import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn import model_selection, metrics
from transformers import AdamW, get_linear_schedule_with_warmup

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as parallel_loader
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from model import BERTBaseUncased
import config, engine_xla, dataset


import warnings
warnings.filterwarnings("ignore")

def get_data_loader(train_set1, train_set2, df_valid, epoch):

    df_train = sample_train_set2(train_set1, train_set2)  # sample subset of toxic==0

    # Train with { train + valid } datasets
    if config.TRAIN_VAL_COMBINE:

        df_valid_en1 = pd.read_csv(config.VALID_CAMARON, usecols=["comment_text_en", "toxic"]) # Camaron Eng [comment_text_en, toxic]
        df_valid_en1.rename(columns = {'comment_text_en':'comment_text'}, inplace = True)
        df_valid_en2 = pd.read_csv(config.VALID_YURY, usecols=["translated", "toxic"]) # Yury Eng [translated, toxic]
        df_valid_en2.rename(columns = {'translated':'comment_text'}, inplace = True)
        df_valid_en3 = pd.read_csv(config.VALID_SHIROK, usecols=["comment_text", "toxic"]) # ShiroK Eng [comment_text, toxic]

        df_train = pd.concat([
                df_train, 
                df_valid,
                df_valid_en1,
                df_valid_en2,
                df_valid_en3
            ], axis=0).reset_index(drop=True)  # train + valid
        df_train = df_train.sample(frac=1).reset_index(drop=True)  # shuffle

    if config.TEST_MODE:
        df_train = df_train.sample(frac=1).reset_index(drop=True).head(2000)  

    if epoch == 1: 
        num0 = df_train.toxic[df_train.toxic == 0].count()
        num1 = df_train.toxic[df_train.toxic == 1].count()
        xm.master_print(f">>> (train set) toxic label%: {num1 / (num0 + num1) :.2f}%")
        num0 = df_valid.toxic[df_valid.toxic == 0].count()
        num1 = df_valid.toxic[df_valid.toxic == 1].count()
        xm.master_print(f">>> (valid set) toxic label%: {num1 / (num0 + num1) :.2f}%")
        xm.master_print(f">>> Total training examples: {len(df_train)} - Toxic Threshold: {config.TOXIC_THRESHOLD}")

    # train set for XLA (dataset + data_sampler)
    train_dataset = dataset.BERTDatasetTrain(
        comment_text=df_train.comment_text.values,
        target=df_train.toxic.values
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        num_workers=1
    )
    
    # validation set for XLA (dataset + data_sampler)
    valid_dataset = dataset.BERTDatasetTrain(
        comment_text=df_valid.comment_text.values,
        target=df_valid.toxic.values
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=1
    )

    return train_data_loader, valid_data_loader


def sample_train_set2(train_set1, train_set2):
    df_train   = pd.concat([
            train_set1, 
            train_set2.query('toxic==1'),
            train_set2.query('toxic==0').sample(n=config.NON_TOXIX_NUM)  # no random state: sample new subset for each epoch
        ], axis=0).reset_index(drop=True)

    return df_train


def run():

    train_set1 = pd.read_csv(config.TRAIN_DATA1, usecols=["comment_text", "toxic"]).fillna("none")

    if config.TRAIN_WITH_2018:
        train_es_pavel = pd.read_csv(config.TRAIN_ES_PAVEL, usecols=["comment_text", "toxic"]).fillna("none")
        train_zafar = pd.read_csv(config.TRAIN_ZAFAR, usecols=["comment_text", "toxic"]).fillna("none")
        train_set1 = pd.concat([
            train_set1, train_es_pavel, train_zafar
        ], axis=0).reset_index(drop=True)

    train_set2 = pd.read_csv(config.TRAIN_DATA2, usecols=["comment_text", "toxic"]).fillna("none")
    train_set2["toxic"] = (train_set2.toxic >= config.TOXIC_THRESHOLD).astype(int)  # toxic threshold 
    df_valid = pd.read_csv(config.VALID_DATA, usecols=["comment_text", "toxic"])

    device = xm.xla_device()
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    best_roc_auc = 0
    best_epoch   = 1
    for epoch in range(1, config.EPOCHS+1):


        # resample toxic==0 for each epoch
        train_data_loader, valid_data_loader = get_data_loader(train_set1, train_set2, df_valid, epoch)

        # set up schedule 
        if epoch == 1:

            lr = 0.4 * 1e-5 * xm.xrt_world_size()
            num_train_steps = int(len(train_data_loader) / xm.xrt_world_size() * config.EPOCHS)
            # print on master node
            xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')
            optimizer = AdamW(optimizer_parameters, lr=lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_train_steps * 0.15), # WARMUP_PROPORTION = 0.1 as default
                num_training_steps=num_train_steps
            )


        para_loader = parallel_loader.ParallelLoader(train_data_loader, [device])
        para_loader = parallel_loader.ParallelLoader(valid_data_loader, [device])

        engine_xla.train_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)
        outputs, targets = engine_xla.eval_fn(para_loader.per_device_loader(device), model, device)

        # threshold the Traget value (0.3 -> 0 ; 0.9 -> 1)
        targets = np.array(targets) >= config.TOXIC_THRESHOLD
        # roc_auc evaluation metric
        roc_auc = metrics.roc_auc_score(targets, outputs)

        xm.master_print(f"Epoch: {epoch} ROC_AUC Score = {roc_auc}")
        xm.save(model.state_dict(), f"xla_{config.SAVE_NAME}_{epoch}.bin")
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_epoch = epoch
        xm.master_print(f"Best ROC_AUC Score = {best_roc_auc} in Epoch {best_epoch}")

if __name__ == "__main__":
    # Start training on XLA device
    def _mp_fn(rank, flags):
        torch.set_default_tensor_type('torch.FloatTensor')
        a = run()

    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=config.XLA_CORES, start_method='fork')

