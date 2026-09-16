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

class FusionLSTMModel(nn.Module):
    def __init__(self, cfg):
        super(FusionLSTMModel, self).__init__()
        #self.use_gate = use_gate
        concat_dim = cfg.img_dim + cfg.action_dim
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
        #if use_gate:
        self.img_weight = nn.Sequential(
            nn.Linear(concat_dim, cfg.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(cfg.gate_hidden_dim, 1),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(concat_dim, cfg.lstm_hidden_dim, num_layers=1,  batch_first=True )
        self.fc = nn.Linear(cfg.lstm_hidden_dim, cfg.output_dim)
        self.dropout = nn.Dropout(p=0.3)
    def forward(self, img_embeddings, action_embeddings):
        
        x = torch.cat([img_embeddings, action_embeddings], dim=-1)
        learned_weight = self.img_weight(x)
        img_embeddings = img_embeddings * learned_weight
        action_embeddings = action_embeddings * (1 - learned_weight)
        concat_input = torch.cat([img_embeddings, action_embeddings], dim=-1)
        #concat_input = self.dropout(concat_input)
        out, _ = self.lstm(concat_input)
        out=self.dropout(out)
        out = torch.sigmoid(self.fc(out))
        #out = torch.sigmoid(self.fc(out))
        return out

class FusionLSTMModel_v2(nn.Module):
    def __init__(self, cfg):
        super(FusionLSTMModel_v2, self).__init__()
        #self.use_gate = use_gate
        concat_dim = cfg.img_dim + cfg.action_dim

        self.img_weight = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(cfg.img_dim, 128),
            #nn.LayerNorm(128, elementwise_affine=False),
            #nn.ReLU(),
            #nn.Dropout(p=0.3),
            #nn.Linear(cfg.gate_hidden_dim, 1),
            #nn.Dropout(p=0.3),
            nn.Tanh(),
        )
        self.action_weight = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(cfg.action_dim, 128),
            #nn.LayerNorm(128, elementwise_affine=False),
            #nn.ReLU(),
            #nn.Dropout(p=0.3),
            #nn.Linear(cfg.gate_hidden_dim, 1),
            #nn.Dropout(p=0.3),
            nn.Tanh(),
        )
        self.fusion_weight = nn.Sequential(
            #nn.LayerNorm(concat_dim,elementwise_affine=True),
            nn.Linear(concat_dim, 128),
            #nn.LayerNorm(128,elementwise_affine=False),
            nn.Sigmoid(),
        )
        # self.img_layernorm = nn.LayerNorm(cfg.img_dim, elementwise_affine=True)
        # self.action_layernorm = nn.LayerNorm(cfg.action_dim, elementwise_affine=True)
        # self.fusion_layernorm = nn.LayerNorm(concat_dim, elementwise_affine=True)

        self.lstm = nn.LSTM(128, cfg.lstm_hidden_dim, num_layers=1,  batch_first=True)
        self.fc = nn.Linear(cfg.lstm_hidden_dim, cfg.output_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()
    def forward(self, img_embeddings, action_embeddings):
        #img_embeddings = self.img_layernorm(img_embeddings)
        #action_embeddings = self.action_layernorm(action_embeddings)

        img_hidden_state= self.img_weight(img_embeddings)
        #img_hidden_state = self.img_layernorm(img_hidden_state)
        action_hidden_state = self.action_weight(action_embeddings)
        # action_hidden_state = self.action_layernorm(action_embeddings)
        #img_hidden_state = self.tanh(img_hidden_state)
        #action_hidden_state = self.tanh(action_hidden_state)
        concat_input = torch.cat([img_embeddings, action_embeddings], dim=-1)
        z = self.fusion_weight(concat_input)

        
        hidden_state = img_hidden_state * z + action_hidden_state * (1 - z)
        out, _ = self.lstm(hidden_state)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))
        return out#, z, img_hidden_state, action_hidden_state
    
class DoubleLSTMModel(nn.Module):
    def __init__(self,cfg):
        super(DoubleLSTMModel, self).__init__()
        concat_dim = cfg.img_dim + cfg.action_dim
        self.img_lstm = nn.LSTM(cfg.img_dim, cfg.lstm_hidden_dim, num_layers=1,  batch_first=True)
        self.action_lstm = nn.LSTM(cfg.action_dim, cfg.lstm_hidden_dim, num_layers=1,  batch_first=True)
        self.fc = nn.Linear(cfg.lstm_hidden_dim * 2, cfg.output_dim)
        self.img_weight = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim*2,cfg.gate_hidden_dim),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(cfg.gate_hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, img_embeddings, action_embeddings):
        img_out, _ = self.img_lstm(img_embeddings)
        action_out, _ = self.action_lstm(action_embeddings)
        #print(img_out.shape, action_out.shape)
        concat_input = torch.cat([img_out, action_out], dim=-1)
        #print(concat_input.shape)
        learned_weight = self.img_weight(concat_input)
        #print(learned_weight.shape)
        img_out = img_out * learned_weight
        action_out = action_out * (1 - learned_weight)
        concat_input = torch.cat([img_out, action_out], dim=-1)
        out = torch.sigmoid(self.fc(concat_input))
        return out