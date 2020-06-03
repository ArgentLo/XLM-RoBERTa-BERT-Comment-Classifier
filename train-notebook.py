import numpy as np
import pandas as pd

import os
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn

import time
import random
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()

from transformers import BertModel, BertTokenizer
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

import gc
import re
import nltk
# python -m nltk.downloader 'punkt'
nltk.download('punkt')
from nltk import sent_tokenize

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=4, progress_bar=True)


from nltk import sent_tokenize
from random import shuffle
import random
import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform


MAX_LENGTH = 196
BACKBONE_PATH = '../pretrained_models/xlm-roberta-large'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = 42
seed_everything(SEED)


class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1)
        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = sklearn.metrics.roc_auc_score(self.y_true, self.y_pred, labels=np.array([0, 1]))
    
    @property
    def avg(self):
        return self.score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



LANGS = {
    'en': 'english',
    'it': 'italian', 
    'fr': 'french', 
    'es': 'spanish',
    'tr': 'turkish', 
    'ru': 'russian',
    'pt': 'portuguese'
}

def get_sentences(text, lang='en'):
    return sent_tokenize(text, LANGS.get(lang, 'english'))

def exclude_duplicate_sentences(text, lang='en'):
    sentences = []
    for sentence in get_sentences(text, lang):
        sentence = sentence.strip()
        if sentence not in sentences:
            sentences.append(sentence)
    return ' '.join(sentences)

def clean_text(text, lang='en'):
    text = str(text)
    text = re.sub(r'[0-9"]', '', text)
    text = re.sub(r'#[\S]+\b', '', text)
    text = re.sub(r'@[\S]+\b', '', text)
    text = re.sub(r'https?\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = exclude_duplicate_sentences(text, lang)
    return text.strip()


class NLPTransform(BasicTransform):
    """ Transform for nlp task."""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(text, LANGS.get(lang, 'english'))

class ShuffleSentencesTransform(NLPTransform):
    """ Do shuffle by sentence """
    def __init__(self, always_apply=False, p=0.5):
        super(ShuffleSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = self.get_sentences(text, lang)
        random.shuffle(sentences)
        return ' '.join(sentences), lang

class ExcludeDuplicateSentencesTransform(NLPTransform):
    """ Exclude equal sentences """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeDuplicateSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = []
        for sentence in self.get_sentences(text, lang):
            sentence = sentence.strip()
            if sentence not in sentences:
                sentences.append(sentence)
        return ' '.join(sentences), lang

class ExcludeNumbersTransform(NLPTransform):
    """ exclude any numbers """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeNumbersTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang

class ExcludeHashtagsTransform(NLPTransform):
    """ Exclude any hashtags with # """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeHashtagsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'#[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang

class ExcludeUsersMentionedTransform(NLPTransform):
    """ Exclude @users """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUsersMentionedTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'@[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang

class ExcludeUrlsTransform(NLPTransform):
    """ Exclude urls """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUrlsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'https?\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang


class SynthesicOpenSubtitlesTransform(NLPTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(SynthesicOpenSubtitlesTransform, self).__init__(always_apply, p)
        df = pd.read_csv('../data/external_data/open-subtitles-synthesic_alex.csv', index_col='id')[['comment_text', 'toxic', 'lang']]
        df = df[~df['comment_text'].isna()]
        df['comment_text'] = df.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
        df = df.drop_duplicates(subset='comment_text')
        df['toxic'] = df['toxic'].round().astype(np.int)

        self.synthesic_toxic = df[df['toxic'] == 1].comment_text.values
        self.synthesic_non_toxic = df[df['toxic'] == 0].comment_text.values

        del df
        gc.collect()

    def generate_synthesic_sample(self, text, toxic):
        texts = [text]
        if toxic == 0:
            for i in range(random.randint(1,5)):
                texts.append(random.choice(self.synthesic_non_toxic))
        else:
            for i in range(random.randint(0,2)):
                texts.append(random.choice(self.synthesic_non_toxic))
            
            for i in range(random.randint(1,3)):
                texts.append(random.choice(self.synthesic_toxic))
        random.shuffle(texts)
        return ' '.join(texts)

    def apply(self, data, **params):
        text, toxic = data
        text = self.generate_synthesic_sample(text, toxic)
        return text, toxic





def get_train_transforms():
    return albumentations.Compose([
        ExcludeUsersMentionedTransform(p=0.95),
        ExcludeUrlsTransform(p=0.95),
        ExcludeNumbersTransform(p=0.95),
        ExcludeHashtagsTransform(p=0.95),
        ExcludeDuplicateSentencesTransform(p=0.95),
    ], p=1.0)

def get_synthesic_transforms():
    return SynthesicOpenSubtitlesTransform(p=0.5)


train_transforms = get_train_transforms()
synthesic_transforms = get_synthesic_transforms()
tokenizer = XLMRobertaTokenizer.from_pretrained(BACKBONE_PATH)
shuffle_transforms = ShuffleSentencesTransform(always_apply=True)




def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):

    def __init__(self, labels_or_ids, comment_texts, langs, use_train_transforms=False, test=False):
        self.test = test
        self.labels_or_ids = labels_or_ids
        self.comment_texts = comment_texts
        self.langs = langs
        self.use_train_transforms = use_train_transforms
        
    def get_tokens(self, text):
        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=MAX_LENGTH, 
            pad_to_max_length=True
        )
        return encoded['input_ids'], encoded['attention_mask']

    def __len__(self):
        return self.comment_texts.shape[0]

    def __getitem__(self, idx):
        text = self.comment_texts[idx]
        lang = self.langs[idx]
        if self.test is False:
            label = self.labels_or_ids[idx]
            target = onehot(2, label)

        if self.use_train_transforms:
            text, _ = train_transforms(data=(text, lang))['data']
            tokens, attention_mask = self.get_tokens(str(text))
            token_length = sum(attention_mask)
            if token_length > 0.8*MAX_LENGTH:
                text, _ = shuffle_transforms(data=(text, lang))['data']
            elif token_length < 60:
                text, _ = synthesic_transforms(data=(text, label))['data']
            else:
                tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)
                return target, tokens, attention_mask

        tokens, attention_mask = self.get_tokens(str(text))
        tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)

        if self.test is False:
            return target, tokens, attention_mask
        return self.labels_or_ids[idx], tokens, attention_mask

    def get_labels(self):
        return list(np.char.add(self.labels_or_ids.astype(str), self.langs))



df_train = pd.read_csv('../data/external_data/train12_trans_alex.csv')

df_train = pd.concat([
    df_train.query("toxic==1"),
    df_train.query("toxic==0").sample(n=500),
], axis=0).reset_index(drop=True)


train_dataset = DatasetRetriever(
    labels_or_ids=df_train['toxic'].values, 
    comment_texts=df_train['comment_text'].values, 
    langs=df_train['lang'].values,
    use_train_transforms=True,
)

del df_train
gc.collect()

for targets, tokens, attention_masks in train_dataset:
    break
    
print(targets)
print(tokens.shape)
print(attention_masks.shape)



df_val = pd.read_csv('../data/validation.csv', index_col='id')

validation_tune_dataset = DatasetRetriever(
    labels_or_ids=df_val['toxic'].values, 
    comment_texts=df_val['comment_text'].values, 
    langs=df_val['lang'].values,
    use_train_transforms=True,
)

df_val['comment_text'] = df_val.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)

validation_dataset = DatasetRetriever(
    labels_or_ids=df_val['toxic'].values, 
    comment_texts=df_val['comment_text'].values, 
    langs=df_val['lang'].values,
    use_train_transforms=False,
)

del df_val
gc.collect()

for targets, tokens, attention_masks in validation_dataset:
    break

print(targets)
print(tokens.shape)
print(attention_masks.shape)


df_test = pd.read_csv('../data/test.csv', index_col='id')
df_test['comment_text'] = df_test.parallel_apply(lambda x: clean_text(x['content'], x['lang']), axis=1)

test_dataset = DatasetRetriever(
    labels_or_ids=df_test.index.values, 
    comment_texts=df_test['comment_text'].values, 
    langs=df_test['lang'].values,
    use_train_transforms=False,
    test=True
)

del df_test
gc.collect()

for ids, tokens, attention_masks in test_dataset:
    break

print(ids)
print(tokens.shape)
print(attention_masks.shape)



class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)




from tqdm import tqdm


class NLP_Fitter:
    
    def __init__(self, model, device, config):
        if not os.path.exists('node_submissions'):
            os.makedirs('node_submissions')

        self.config = config
        self.epoch = 0
        self.log_path = 'log.txt'

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.criterion = config.criterion
        print(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
#             para_loader = pl.ParallelLoader(train_loader, [self.device])
            losses, final_scores = self.train_one_epoch(train_loader)
            
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
#             para_loader = pl.ParallelLoader(validation_loader, [self.device])
            losses, final_scores = self.validation(validation_loader)

            self.log(f'[RESULT]: Validation. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=final_scores.avg)

            self.epoch += 1
    
    def run_tuning_and_inference(self, test_loader, validation_tune_loader):
        for e in range(2):
            self.optimizer.param_groups[0]['lr'] =  (2e-5) / (e + 1)
#             para_loader = pl.ParallelLoader(validation_tune_loader, [self.device])
            losses, final_scores = self.train_one_epoch(validation_tune_loader)
#             para_loader = pl.ParallelLoader(test_loader, [self.device])
            self.run_inference(validation_tune_loader)

    def validation(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        final_scores = RocAucMeter()

        t = time.time()
        for step, (targets, inputs, attention_masks) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Valid Step {step}, loss: ' + \
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )
            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long) 
                attention_masks = attention_masks.to(self.device, dtype=torch.long) 
                targets = targets.to(self.device, dtype=torch.float) 

                outputs = self.model(inputs, attention_masks)
                loss = self.criterion(outputs, targets)
                
                batch_size = inputs.size(0)

                final_scores.update(targets, outputs)
                losses.update(loss.detach().item(), batch_size)
                
        return losses, final_scores
         
    def train_one_epoch(self, train_loader):
        self.model.train()

        losses = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (targets, inputs, attention_masks) in tqdm(enumerate(train_loader), total=len(train_loader)):   
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    self.log(
                        f'Train Step {step}, loss: ' + \
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )

            inputs = inputs.to(self.device, dtype=torch.long)
            attention_masks = attention_masks.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, attention_masks)
            loss = self.criterion(outputs, targets)

            batch_size = inputs.size(0)
            
            final_scores.update(targets, outputs)
            
            losses.update(loss.detach().item(), batch_size)

            loss.backward()
            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()
        
        self.model.eval()
        self.save('xlm-roberta-checkpoint.bin')
        return losses, final_scores

    def run_inference(self, test_loader):
        self.model.eval()
        result = {'id': [], 'toxic': []}
        t = time.time()
        for step, (ids, inputs, attention_masks) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(f'Prediction Step {step}, time: {(time.time() - t):.5f}')

            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long) 
                attention_masks = attention_masks.to(self.device, dtype=torch.long)
                outputs = self.model(inputs, attention_masks)
                toxics = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]

            result['id'].extend(ids.cpu().numpy())
            result['toxic'].extend(toxics)

        result = pd.DataFrame(result)
        node_count = len(glob('node_submissions/*.csv'))
        result.to_csv(f'node_submissions/submission_{node_count}_{datetime.utcnow().microsecond}_{random.random()}.csv', index=False)

    def save(self, path):        
        torch.save(self.model.state_dict(), path)

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            print(f'{message}', logger)





class ToxicSimpleNNModel(nn.Module):

    def __init__(self):
        super(ToxicSimpleNNModel, self).__init__()
        self.backbone = XLMRobertaModel.from_pretrained(BACKBONE_PATH)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(
            in_features=self.backbone.pooler.dense.out_features*2,
            out_features=2,
        )

    def forward(self, input_ids, attention_masks):
        bs, seq_length = input_ids.shape
        seq_x, _ = self.backbone(input_ids=input_ids, attention_mask=attention_masks)
        apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)
        return self.linear(x)




class TrainGlobalConfig:
    num_workers = 8
    batch_size = 1
    n_epochs = 3
    lr = 2e-5

    # -------------------
    verbose = False
    verbose_step = 50
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='max',
        factor=0.7,
        patience=0,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------

    # -------------------
    criterion = LabelSmoothing()
    # -------------------



net = ToxicSimpleNNModel()

device = torch.device("cuda")
net.to(device)
print("Loaded to CUDA GPU.")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=TrainGlobalConfig.batch_size,
    num_workers=TrainGlobalConfig.num_workers
)


validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=TrainGlobalConfig.batch_size,
    num_workers=TrainGlobalConfig.num_workers
)



validation_tune_loader = torch.utils.data.DataLoader(
    validation_tune_dataset,
    batch_size=TrainGlobalConfig.batch_size,
    num_workers=TrainGlobalConfig.num_workers
)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=TrainGlobalConfig.batch_size,
    num_workers=TrainGlobalConfig.num_workers
)

fitter = NLP_Fitter(model=net, device=device, config=TrainGlobalConfig)


# fitter.fit(train_loader, validation_loader)
fitter.train_one_epoch(train_loader)

