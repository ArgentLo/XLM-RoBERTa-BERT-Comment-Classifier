import torch 
from tqdm import tqdm 
import numpy as np
import config

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64) 
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -= 2
    all_tokens = []

    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        cur_len = len(tokens_a)
        if cur_len > max_seq_length:
            half_seq_length = (int)(max_seq_length/2)
            if config.MAX_LEN % 2 == 0: 
                tokens_a = tokens_a[:half_seq_length] + tokens_a[cur_len-half_seq_length:]
            else:
                tokens_a = tokens_a[:half_seq_length] + tokens_a[cur_len-half_seq_length-1:]


        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - cur_len)
        all_tokens.append(one_token)

    return np.array(all_tokens)


def trim_tensors(tsrs):
    max_len = torch.max(torch.sum( (tsrs[0] != 0  ), 1))
    if max_len > 2: 
        tsrs = [tsr[:, :max_len] for tsr in tsrs]
    return tsrs 