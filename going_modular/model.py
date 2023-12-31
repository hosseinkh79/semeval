from torch import nn

from transformers import  BertModel

#our Model
class Bert(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        #freeze the parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
            )

    def forward(self, input, attention_mask=None):

        outputs = self.bert(input, attention_mask=attention_mask)
        # output = outputs['pooler_output']
        output = outputs['last_hidden_state'][:, 0, :]
        out = self.classifier(output)

        return out
