import torch
import torch.nn as nn
from tqdm import tqdm

# Only run in XLA device environment
import torch_xla
import torch_xla.core.xla_model as xm

from focal_loss import FocalLoss
import config

def loss_fn(outputs, targets, focal_loss=False):
    if focal_loss:
        return FocalLoss()(outputs, targets.view(-1, 1))
    else:
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
    # for bi, d in enumerate(data_loader):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)
        # if bi % 200 == 0:
        #     xm.master_print(f'Batch Index={bi}, Loss={loss}')
        loss.backward()
        xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():

        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        # for bi, d in enumerate(data_loader):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            targets_np = targets.cpu().detach().numpy().tolist()
            outputs_np = outputs.cpu().detach().numpy().tolist()
            fin_targets.extend(targets_np)
            fin_outputs.extend(outputs_np)

    return fin_outputs, fin_targets
