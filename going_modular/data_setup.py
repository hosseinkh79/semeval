from torch.utils.data import Dataset
from torch import nn
import torch

class SemEvalDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer    
        self.max_length = 25

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']

        encoding = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
            )
        
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        # resize the tensors to the same size
        input_ids = nn.functional.pad(input_ids, (0, self.max_length - input_ids.shape[0]), value=0)
        attention_mask = nn.functional.pad(attention_mask, (0, self.max_length - attention_mask.shape[0]), value=0)
        
        return input_ids, attention_mask, torch.tensor(label)