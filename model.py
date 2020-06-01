import transformers
import torch.nn as nn
import torch

import config


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        # extra layers
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768 * 2, 1)

    def forward(self, ids, mask, token_type_ids):

       # https://huggingface.co/transformers/model_doc/bert.html?highlight=bert#transformers.BertModel.forward
       # BERT return: <last_hidden_state>, <pooler_output> [opt. hidden_states, attentions]
       # Note that: in doc, it says <last_hidden_state> is better semantic summery than <pooler_output>.
       # thus, we use out1 <last_hidden_state>.
        out1, _ = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        mean_pooling = torch.mean(out1, dim=1)
        max_pooling, _ = torch.max(out1, dim=1)
        cat = torch.cat((mean_pooling, max_pooling), dim=1)

        bo = self.bert_drop(cat)
        output = self.out(bo)
        return output
