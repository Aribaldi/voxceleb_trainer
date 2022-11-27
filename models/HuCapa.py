from transformers import HubertModel, HubertConfig
import torch
from speechbrain.lobes.models import ECAPA_TDNN
from torch import nn


class HuCapa(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert.config.output_hidden_states = True
        self.hubert.to(device)
        self.hubert.eval()
        self.hubert.requires_grad_(False)
        
        self.hs_weights = nn.Linear(13, 1, bias=False, device=device)
        nn.init.ones_(self.hs_weights.weight)


        self.ecapa = ECAPA_TDNN.ECAPA_TDNN(768)
        self.ecapa.to(device)

    def forward(self, x):
        backbone_embs = self.hubert(x).hidden_states
        backbone_embs = torch.stack(backbone_embs).squeeze(1)
        backbone_embs = torch.transpose(backbone_embs, 0, 1)
        backbone_embs = torch.transpose(backbone_embs, 3, 1)
        linear_comb_hs = self.hs_weights(backbone_embs)
        linear_comb_hs = linear_comb_hs.squeeze(3)
        linear_comb_hs = linear_comb_hs.transpose(2, 1)
        ecapa_embs = self.ecapa(linear_comb_hs)
        return ecapa_embs