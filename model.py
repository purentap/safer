import torch
import torch.nn as nn

class MLPModel(nn.Module):

    def __init__(self, input_dim=2176, hidden_dim=64, output_dim=1):
        super(MLPModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            #nn.Linear(hidden_dim, 64, bias=True),
            #nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.layers(x)
        # Accumulate the scores over the time dimension
        
        x = torch.cumsum(x, dim=-2) # (batch_size, seq_len, 1)

        return x
        
class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(cfg.input_dim, cfg.hidden_dim, cfg.num_layers,  batch_first=True)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(self, img_embeddings, action_embeddings):
        out, _ = self.lstm(img_embeddings)
        out = torch.sigmoid(self.fc(out))
        return out

class FusionLSTMModel(nn.Module):
    def __init__(self, img_dim, action_dim, lstm_hidden_dim=256, gate_hidden_dim=512, output_dim=1, use_gate=True):
        super(FusionLSTMModel, self).__init__()
        self.use_gate = use_gate
        concat_dim = img_dim + action_dim
        #self.img_weight = nn.Parameter(torch.tensor(1.0))
        #self.action_weight = nn.Parameter(torch.tensor(1.0))

        '''
        self.img_gate = nn.Sequential(
            nn.Linear(concat_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid()
        )
        self.action_gate = nn.Sequential(
            nn.Linear(concat_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid()
        )
        '''
        if use_gate:
            self.img_weight = nn.Sequential(
                nn.Linear(concat_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid()
            )
        self.lstm = nn.LSTM(concat_dim, lstm_hidden_dim, num_layers=1,  batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
    def forward(self, img_embeddings, action_embeddings):
        
        if self.use_gate:
            x = torch.cat([img_embeddings, action_embeddings], dim=-1)
            learned_weight = self.img_weight(x)
            img_embeddings = img_embeddings * learned_weight
            action_embeddings = action_embeddings * (1 - learned_weight)
        concat_input = torch.cat([img_embeddings, action_embeddings], dim=-1)
        out, _ = self.lstm(concat_input)
        out = torch.sigmoid(self.fc(out))
        #out = torch.sigmoid(self.fc(out))
        return out