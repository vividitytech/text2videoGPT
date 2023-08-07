import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class TextVideoDataset(Dataset):

    def __init__(self, tokenizer, examples, name2label, transform, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.examples[i]['answer'])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")
        self.name2label = name2label
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.examples[idx]['answer']
        if self.transform is not None:
            ans_tokens = self.transform(ans_tokens)

        pad_tokens = [50256] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        # tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        # tokens = th.tensor(tokens)
        mask = torch.tensor(mask)
        label = self.name2label[self.examples[idx]['name']]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
    
        query_pad_tokens = [0] * (self.max_len - len(qn_tokens) )
        return dict(answer=torch.tensor(ans_tokens), attention_mask=mask, query=torch.tensor(qn_tokens), label=torch.tensor(label), data=self.examples[idx])
