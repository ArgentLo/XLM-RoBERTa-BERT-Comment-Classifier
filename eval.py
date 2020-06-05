import os
import torch
import pandas as pd
from scipy import stats
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
import sys

from model import BERTBaseUncased
import config, engine, dataset


def scale_min_max_submission(submission):
    min_, max_ = submission['toxic'].min(), submission['toxic'].max()
    submission['toxic'] = (submission['toxic'] - min_) / (max_ - min_)
    return submission

def run(model_path):

    device = "cuda"
    model = BERTBaseUncased().to(device)
    # For multiple GPUs
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(f"{model_path}"))
    model.eval()
    print(">>> model loaded.")

    print(">>> Start evaluation on EN 1 ....")
    df_en = pd.read_csv(config.TEST_CAMARON)
    test_dataset = dataset.BERTDatasetTest(comment_text=df_en.content_en.values)
    print(f"test size: {len(test_dataset)}")
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=config.TEST_WORKERS
    )

    with torch.no_grad():
        fin_outputs_en = []
        for bi, d in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            outputs_np = outputs.cpu().detach().numpy().tolist()
            fin_outputs_en.extend(outputs_np)


    print(">>> Start evaluation on EN 2 ....")
    df_en = pd.read_csv(config.TEST_YURY)
    test_dataset = dataset.BERTDatasetTest(comment_text=df_en.translated.values)
    print(f"test size: {len(test_dataset)}")
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=config.TEST_WORKERS
    )

    with torch.no_grad():
        fin_outputs_en2 = []
        for bi, d in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            outputs_np = outputs.cpu().detach().numpy().tolist()
            fin_outputs_en2.extend(outputs_np)

    return fin_outputs_en, fin_outputs_en2

if __name__ == "__main__":

    for i in range(2, 6):
        trained_model = f"LossWeight_600k_{i}.bin"
        print(">>> evaluating Model: ", trained_model)
        model_path = "./checkpoints/" + trained_model

        fin_outputs_en, fin_outputs_en2 = run(model_path)
        fin_outputs_en = [item for sublist in fin_outputs_en for item in sublist]
        fin_outputs_en2 = [item for sublist in fin_outputs_en2 for item in sublist]

        sample = pd.read_csv("../data/sample_submission.csv")
        sample.loc[:, "toxic"] = (np.array(fin_outputs_en) + np.array(fin_outputs_en2)) / 2.0

        # # sacle the predictions
        # sample = scale_min_max_submission(sample)

        sample.to_csv(f"./checkpoints/{trained_model[:-4]}.csv", index=False)
