import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn import model_selection, metrics
from transformers import AdamW, get_linear_schedule_with_warmup

from model import XLMRobertaLarge
import config, engine_roberta_base, dataset

import warnings
warnings.filterwarnings("ignore")



def get_data_loader(train_set1, train_set2, df_valid, epoch, train_with_alex):

    if train_with_alex:
        df_train = train_set1
    else:
        df_train = sample_train_set2(train_set1, train_set2, config.TOXIC_THRESHOLD)  # sample subset of toxic==0

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
        df_train = df_train.sample(frac=1).reset_index(drop=True).head(1000)  # test: 1000 examples


    if epoch == 1: 
        num0 = df_train.toxic[df_train.toxic == 0].count()
        num1 = df_train.toxic[df_train.toxic == 1].count()
        print(f">>> (train set) toxic label%: {100*(num1 / (num0 + num1)) :.2f}%")
        num0 = df_valid.toxic[df_valid.toxic == 0].count()
        num1 = df_valid.toxic[df_valid.toxic == 1].count()
        print(f">>> (valid set) toxic label%: {100*(num1 / (num0 + num1)) :.2f}%")
        # loss weight
        if config.LOSS_WEIGHT:
            loss_w = train_toxic_ratio / (num1 / (num0 + num1))
            df_train.loc[:, "weight"] = df_train.loc[:, "toxic"]
            df_train.loc[:, "weight"].replace(0, loss_w, inplace=True)  # replace non-toxic label
            print(f">>>  Loss Weight for non-toxic lable: {loss_w :.2f}.")
        else: 
            df_train.loc[:, "weight"] = 1
        print(f">>> Total training examples: {len(df_train)} - Toxic Threshold: {config.TOXIC_THRESHOLD}")

    # default training on GPU
    train_dataset = dataset.BERTDatasetTrain(
        comment_text=df_train.comment_text.values,
        target=np.concatenate(
            (df_train.toxic.values[:, None], 
            df_train.weight.values[:, None]), axis=1)
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=config.TRAIN_WORKERS
    )

    valid_dataset = dataset.XLMRobertaDatasetTrain(
        comment_text=df_valid.comment_text.values,
        target=df_valid.toxic.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=config.VALID_WORKERS
    )

    return train_data_loader, valid_data_loader


def sample_train_set2(train_set1, train_set2, threshold):
    df_train   = pd.concat([
            train_set1, 
            train_set2.query(f'toxic>={threshold}'),
            train_set2.query('toxic==0').sample(n=config.NON_TOXIX_NUM)  # no random state: sample new subset for each epoch
        ], axis=0).reset_index(drop=True)

    return df_train


def run():

    # alex set: train1+2 + 6_lang_trans_micheal
    if config.TRAIN_WITH_ALEX:
        train_set_alex = pd.read_csv(config.TRAIN_ALEX, usecols=["comment_text", "toxic"]).fillna("none")

    else:
        train_set1 = pd.read_csv(config.TRAIN_DATA1, usecols=["comment_text", "toxic"]).fillna("none")

        if config.TRAIN_WITH_2018:
            # train_de_pavel = pd.read_csv(config.TRAIN_DE_PAVEL, usecols=["comment_text", "toxic"]).fillna("none")
            # train_fr_pavel = pd.read_csv(config.TRAIN_FR_PAVEL, usecols=["comment_text", "toxic"]).fillna("none")
            train_es_pavel = pd.read_csv(config.TRAIN_ES_PAVEL, usecols=["comment_text", "toxic"]).fillna("none")
            train_zafar = pd.read_csv(config.TRAIN_ZAFAR, usecols=["comment_text", "toxic"]).fillna("none")
            train_set1 = pd.concat([
                train_set1, train_es_pavel, train_zafar
            ], axis=0).reset_index(drop=True)

        train_set2 = pd.read_csv(config.TRAIN_DATA2, usecols=["comment_text", "toxic"]).fillna("none")

        # don't round up -> if train2 as float
        if config.TRAIN_FLOAT_SET2:
            train_set2["toxic"] = train_set2["toxic"][train_set2.toxic >= config.TOXIC_THRESHOLD] 
            train_set2 = train_set2.fillna(0)

        else:
            train_set2["toxic"] = (train_set2.toxic >= config.TOXIC_THRESHOLD).astype(int)  # toxic threshold 

    df_valid = pd.read_csv(config.VALID_DATA, usecols=["comment_text", "toxic"])

    gpu_cnt = torch.cuda.device_count()
    print(f">>> Available GPUs: {gpu_cnt}")
    for i in range(gpu_cnt):
        print(torch.cuda.get_device_name(i))


    device = torch.device("cuda")
    model  = XLMRobertaLarge()
    model.to(device)

    # For multiple GPUs
    if config.PARALLEL:
        model = nn.DataParallel(model)
        


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
    if config.TRAIN_WITH_ALEX:
        train_data_loader, valid_data_loader = get_data_loader(train_set_alex, None, df_valid, 1, config.TRAIN_WITH_ALEX)

    for epoch in range(1, config.EPOCHS+1):

        if not config.TRAIN_WITH_ALEX:
            # resample toxic==0 for each epoch
            train_data_loader, valid_data_loader = get_data_loader(train_set1, train_set2, df_valid, epoch, config.TRAIN_WITH_ALEX)

        # set up schedule 
        if epoch == 1:
            # num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
            num_train_steps = int(len(train_data_loader) * config.EPOCHS) # len(train_data_loader)==len(train_dataset)/batchsize
            optimizer = AdamW(optimizer_parameters, lr=config.LR)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_train_steps * config.WARM_UP), # WARMUP_PROPORTION = 0.1 as default
                num_training_steps=num_train_steps
            )

        # lr = { 2e-5, 1e-5 } for epoch 1, 2
        if epoch == 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
                param_group['warmup'] = 0

        # clear gradients (accumulated backprop in each batch -> train_fn)
        optimizer.zero_grad()

        # train + save
        engine_roberta_base.train_fn(train_data_loader, model, optimizer, device, scheduler)
        torch.save(model.state_dict(), f"{config.SAVE_NAME}_{epoch}.bin")

        # # eval
        # outputs, targets = engine_roberta_base.eval_fn(valid_data_loader, model, device)
        # # threshold the Traget value (0.3 -> 0 ; 0.9 -> 1)
        # targets = np.array(targets) >= config.TOXIC_THRESHOLD
        # # roc_auc evaluation metric
        # roc_auc = metrics.roc_auc_score(targets, outputs)
        
        # print(f"Epoch: {epoch} ROC_AUC Score = {roc_auc}")
        # if roc_auc > best_roc_auc:
        #     best_roc_auc = roc_auc
        #     best_epoch = epoch
        # print(f"Best ROC_AUC Score = {best_roc_auc} in Epoch {best_epoch}")

if __name__ == "__main__":
    run()
