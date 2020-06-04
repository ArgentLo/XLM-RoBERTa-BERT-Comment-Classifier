import torch

import config


class XLMRobertaDatasetTrain:
    def __init__(self, comment_text, target):
        self.comment_text = comment_text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, idx):
        comment_text = str(self.comment_text[idx])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,  # 1st text to tokenize
            None,    # no 2nd text in this task
            add_special_tokens=True,
            # max_length=self.max_len,
            pad_to_max_length=False
        )

        cur_len = len(inputs["input_ids"])
        if cur_len > self.max_len:
            half_len = (int)(self.max_len / 2)
            inputs["input_ids"]      = inputs["input_ids"][:half_len] + inputs["input_ids"][cur_len - half_len:]
            inputs["attention_mask"] = inputs["attention_mask"][:half_len] + inputs["attention_mask"][cur_len - half_len:]

        # add paddings
        padding_length = self.max_len - cur_len
        ids            = inputs["input_ids"] + ([0] * padding_length)
        mask           = inputs["attention_mask"] + ([0] * padding_length)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.target[idx], dtype=torch.float)
        }


class XLMRobertaDatasetTest:
    def __init__(self, comment_text):
        self.comment_text = comment_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, idx):
        comment_text = str(self.comment_text[idx])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            # max_length=self.max_len,
            pad_to_max_length=False
        )

        cur_len = len(inputs["input_ids"])
        if cur_len > self.max_len:
            half_len = (int)(self.max_len / 2)
            inputs["input_ids"]      = inputs["input_ids"][:half_len] + inputs["input_ids"][cur_len - half_len:]
            inputs["attention_mask"] = inputs["attention_mask"][:half_len] + inputs["attention_mask"][cur_len - half_len:]

        # add paddings
        padding_length = self.max_len - cur_len
        ids            = inputs["input_ids"] + ([0] * padding_length)
        mask           = inputs["attention_mask"] + ([0] * padding_length)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }




class BERTDatasetTrain:
    def __init__(self, comment_text, target):
        self.comment_text = comment_text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, idx):
        comment_text = str(self.comment_text[idx])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,  # 1st text to tokenize
            None,    # no 2nd text in this task
            add_special_tokens=True,
            # max_length=self.max_len,
            pad_to_max_length=False
        )

        cur_len = len(inputs["input_ids"])
        if cur_len > self.max_len:
            half_len = (int)(self.max_len / 2)
            inputs["input_ids"]      = inputs["input_ids"][:half_len] + inputs["input_ids"][cur_len - half_len:]
            inputs["attention_mask"] = inputs["attention_mask"][:half_len] + inputs["attention_mask"][cur_len - half_len:]

        # add paddings
        padding_length = self.max_len - cur_len
        ids            = inputs["input_ids"] + ([0] * padding_length)
        mask           = inputs["attention_mask"] + ([0] * padding_length)
        token_type_ids = [0] * self.max_len  # single text_seq, token_type_ids==0

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[idx], dtype=torch.float)
        }


class BERTDatasetTest:
    def __init__(self, comment_text):
        self.comment_text = comment_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, idx):
        comment_text = str(self.comment_text[idx])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            # max_length=self.max_len,
            pad_to_max_length=False
        )

        cur_len = len(inputs["input_ids"])
        if cur_len > self.max_len:
            half_len = (int)(self.max_len / 2)
            inputs["input_ids"]      = inputs["input_ids"][:half_len] + inputs["input_ids"][cur_len - half_len:]
            inputs["attention_mask"] = inputs["attention_mask"][:half_len] + inputs["attention_mask"][cur_len - half_len:]

        # add paddings
        padding_length = self.max_len - cur_len
        ids            = inputs["input_ids"] + ([0] * padding_length)
        mask           = inputs["attention_mask"] + ([0] * padding_length)
        token_type_ids = [0] * self.max_len  # single text_seq, token_type_ids==0

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
        }

