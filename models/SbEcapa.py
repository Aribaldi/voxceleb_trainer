from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
import torchaudio
from torch import nn
import torch
import glob
import numpy as np
from utils import PreEmphasis
from DatasetLoader import loadWAV



class CustomEcapa(ECAPA_TDNN):
    def __init__(self, input_size) -> None:
        super().__init__(input_size=input_size)
        self.fbank = nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MFCC(
                sample_rate=16000, n_mfcc=80, 
                log_mels=True, 
                dct_type=2,
                melkwargs={'n_mels': 80, 'n_fft':512, 'win_length':400, 'hop_length':160, 'f_min':20, 'f_max':7600, 'window_fn':torch.hamming_window}), 
            )
    
    def forward(self, x, lengths=None):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.fbank(x) + 1e-6
                x = x - torch.mean(x, dim=-1, keepdim=True)
    
        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        x = x.transpose(1, 2)
        return x.squeeze(1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavs_list = glob.glob("./data/voxceleb2/*/*/*.wav")
    test = [np.random.choice(wavs_list) for t in range(4)]
    waveforms = [loadWAV(t, 200, False) for t in test]
    print(waveforms[0].shape)
    tensor = torch.tensor(waveforms, dtype=torch.float, device=device).squeeze(1)
    print(tensor.shape)
    model = CustomEcapa(input_size=80)
    model.to(device)
    out = model(tensor)
    print(out.shape)