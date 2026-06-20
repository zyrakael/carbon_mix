import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeAwareVarMixer(nn.Module):
    def __init__(self, d_vars, rank=4, ctx_len=16, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_vars, rank, bias=False)
        self.fc2 = nn.Linear(rank, d_vars, bias=False)
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_vars, d_vars),
            nn.Sigmoid()
        )
        self.ctx_len = ctx_len
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        y = self.fc2(F.gelu(self.fc1(x)))
        ctx = x[:, -self.ctx_len:, :].mean(dim=1)
        g = self.ctx_proj(ctx).unsqueeze(1)
        y = self.dropout(y * g)
        return x + self.gate * y

class CarbonMixBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        d = configs.enc_in
        # self.pre_norm = nn.LayerNorm(d)
        self.alpha_temp = nn.Parameter(torch.tensor(0.0))
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )
        self.var_mixer = TimeAwareVarMixer(
            d_vars=d,
            rank=getattr(configs, "var_rank", 4),
            ctx_len=getattr(configs, "ctx_len", 16),
            dropout=configs.dropout
        )
    def forward(self, x):
        x_t = x.transpose(1, 2)  # [B, D, L]
        # 1) temporal mixing
        temp = self.temporal(x_t)
        x = x + self.alpha_temp * temp.transpose(1, 2)
        # 2) time-aware variable interaction
        x = self.var_mixer(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len

        self.model = nn.ModuleList([
            CarbonMixBlock(configs) for _ in range(configs.e_layers)
        ])
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return out[:, -self.pred_len:, :]
        else:
            raise ValueError('Only forecast tasks implemented yet')
