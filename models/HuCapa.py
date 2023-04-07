from transformers import HubertModel, HubertConfig
import torch
from speechbrain.lobes.models import ECAPA_TDNN
from torch import nn
from loss.aamsoftmax import LossFunction

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


        self.ecapa = ECAPA_TDNN.ECAPA_TDNN(768, channels=[1024, 1024, 1024, 1024, 3072])
        self.ecapa.to(device)

        self.loss = LossFunction(nOut=192, nClasses=5994, margin=0.2, scale=30)
        self.loss.to(device)


    def forward(self, x, label, mode):
        backbone_embs = self.hubert(x).hidden_states
        backbone_embs = torch.stack(backbone_embs).squeeze(1)
        backbone_embs = torch.transpose(backbone_embs, 0, 1)
        backbone_embs = torch.transpose(backbone_embs, 3, 1)
        linear_comb_hs = self.hs_weights(backbone_embs)
        linear_comb_hs = linear_comb_hs.squeeze(3)
        linear_comb_hs = linear_comb_hs.transpose(2, 1)
        ecapa_embs = self.ecapa(linear_comb_hs)
        if mode=="train":
            return self.loss(ecapa_embs.squeeze(1), label)
        else:
            return ecapa_embs.squeeze(1)


if __name__ == "__main__":
    import glob
    import numpy as np
    from DatasetLoader import loadWAV
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavs_list = glob.glob("./data/voxceleb2/*/*/*.wav")
    test = [np.random.choice(wavs_list) for t in range(4)]
    waveforms = [loadWAV(t, 200, False) for t in test]
    print(waveforms[0].shape)
    tensor = torch.tensor(waveforms, dtype=torch.float, device=device).squeeze(1)
    print(tensor.shape)
    model = HuCapa(device)
    model.to(device)
    out = model(tensor, torch.ones(4, device=device, dtype=torch.long), "train")
    print(out.shape)