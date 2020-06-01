import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn import model_selection, metrics
from transformers import AdamW, get_linear_schedule_with_warmup
import tqdm

from model import BERTBaseUncased
import config, engine_xla, dataset

def run():

    df_test = pd.read_csv(config.TEST_DATA, usecols=["content"])  # raw multi-languages
    # df_test_en1 = pd.read_csv(config.TEST_CAMARON, usecols=["content_en"])  # Camaron translation
    # df_test_en2 = pd.read_csv(config.TEST_YURY, usecols=["translated"])  # Yury translation


    test_dataset = dataset.BERTDatasetTest(comment_text=df_test.content.values)
    # test_dataset = dataset.BERTDatasetTest(comment_text=df_test_en1.content_en.values)
    # test_dataset = dataset.BERTDatasetTest(comment_text=df_test_en2.translated.values)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=8
    )

    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)


    with torch.no_grad():
        predictions = []
        for bi, d in tqdm(enumerate(test_data_loader)):
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
            predictions.extend(outputs_np)
    return predictions

if __name__ == "__main__":
    predictions = run()
    predictions = [item for sublist in predictions for item in sublist]

    sample = pd.read_csv(config.SAMPLE_SUB)
    sample.loc[:, "toxic"] = np.array(predictions)
    sample.to_csv("submission.csv", index=False)