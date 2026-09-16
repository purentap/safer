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
        if cfg.mode == "img_action": 
            self.mode = "img_action"
            input_dim = cfg.img_dim + cfg.action_dim
        elif cfg.mode == "action":
            self.mode = "action"
            input_dim = cfg.action_dim
        else:
            self.mode = "img"
            input_dim = cfg.img_dim
        self.lstm = nn.LSTM(input_dim, cfg.hidden_dim, cfg.num_layers,  batch_first=True, )
        self.fc = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.dropout = nn.Dropout(p=0.0, inplace=False)


    def forward(self, img_embeddings, action_embeddings):
        if self.mode == "img_action":
            x = torch.cat([img_embeddings, action_embeddings], dim=-1)
        elif self.mode == "action":
            x = action_embeddings
        else:
            x = img_embeddings
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))
        return out

class FusionLSTMModel_v2(nn.Module):
    def __init__(self, cfg):
        super(FusionLSTMModel_v2, self).__init__()
        #self.use_gate = use_gate
        concat_dim = cfg.img_dim + cfg.action_dim

        self.img_weight = nn.Sequential(
            #nn.Dropout(p=0.3),
            nn.Linear(cfg.img_dim, 128),
            nn.Tanh(),
        )
        self.action_weight = nn.Sequential(
            #nn.Dropout(p=0.3),
            nn.Linear(cfg.action_dim, 128),
            nn.Tanh(),
        )
        self.fusion_weight = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.Sigmoid(),
        )

        self.lstm = nn.LSTM(128, cfg.lstm_hidden_dim, num_layers=1,  batch_first=True)
        self.fc = nn.Linear(cfg.lstm_hidden_dim, cfg.output_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()
    def forward(self, img_embeddings, action_embeddings):

        img_hidden_state= self.img_weight(img_embeddings)
        action_hidden_state = self.action_weight(action_embeddings)
        concat_input = torch.cat([img_embeddings, action_embeddings], dim=-1)
        z = self.fusion_weight(concat_input)        
        hidden_state = img_hidden_state * z + action_hidden_state * (1 - z)

        out, _ = self.lstm(hidden_state)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))
        return out
