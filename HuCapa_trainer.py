import torch
from torch import nn
from models.HuCapa import HuCapa
from DatasetLoader import train_dataset_loader, test_dataset_loader, train_dataset_sampler, worker_init_fn
from torch.utils.data import DataLoader
from loss.aamsoftmax import LossFunction as AAM
import time
import sys
from tuneThreshold import *
import itertools
import random
import torch.nn.functional as F
import numpy as np

def evaluateFromList(test_list, test_path, model, nDataLoaderThread, print_interval=100, num_eval=10, eval_frames=300):

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, eval_frames=eval_frames)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=None)

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = model(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        tstart = time.time()
        print("")

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

   
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

            score = -1 * np.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                sys.stdout.flush()

        return (all_scores, all_labels, all_trials)


class HuCapaTrainer(nn.Module):
    def __init__(
        self, 
        max_frames: int=500, 
        device:torch.device=torch.device("cuda"), 
        batch_size:int = 32, 
        num_wokers:int=16, 
        seed:int=10,
        lr:float=1e-3,
        test_step:int=1,
        lr_decay:float=0.97
    ) -> None:
        super().__init__()
        self.device = device
        self.model = HuCapa(self.device)
        train_dataset = train_dataset_loader(
            train_list="data/train_list.txt",
            augment=False,
            musan_path="",
            rir_path="",
            max_frames=max_frames,
            train_path="data/voxceleb2"
            )
        train_sampler = train_dataset_sampler(
            train_dataset,
            nPerSpeaker=1, 
            max_seg_per_spk=500, 
            batch_size=batch_size,
            distributed=False,
            seed=seed
            )
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_wokers,
            sampler=train_sampler,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            drop_last=True
            )
        self.test_file = "./data/test_list.txt"
        self.test_path = "./data/voxceleb1/"
        self.optim = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        self.loss = AAM(nOut=192, nClasses=5994, margin=0.2, scale=30)
        print(time.strftime("%m-%d %H:%M:%S") + " Overall parameters: = %.2f"%(sum(param.numel() for param in self.model.parameters())))
        print(f"Learnable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def load_params(self, path):
        checkpoint = torch.load(path)
        self.model.ecapa.load_state_dict(checkpoint["ecapa"])
        self.model.hs_weights.load_state_dict(checkpoint["hs_weights"])
        self.optim.load_state_dict(checkpoint["opt"])
        return checkpoint["epoch"]

    def train(self, epoch):
        self.model.ecapa.train()
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]["lr"]
        for num, (data, labels) in enumerate(self.dataloader, start=1):
            self.zero_grad()
            data = data.to(self.device)
            data = data.squeeze(1)
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            out = self.model(data)
            out = out.squeeze(1)
            nloss, prec = self.loss(out, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / self.dataloader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
            sys.stderr.flush()
        self.scheduler.step()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index*len(labels)


    def eval(self, epoch):
        self.model.ecapa.eval()
        sc, lab, _ = evaluateFromList(self.test_file, self.test_path, self.model, self.dataloader.num_workers)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1])
        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)
        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(epoch, result[1], mindcf))
        return result[1], mindcf