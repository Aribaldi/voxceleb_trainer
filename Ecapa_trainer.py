import torch
from torch import nn
from models.SbEcapa import CustomEcapa
from models.CustomEcapa import ECAPA_TDNN
from DatasetLoader import train_dataset_loader, train_dataset_sampler, worker_init_fn
from torch.utils.data import DataLoader
from loss.aamsoftmax import LossFunction as AAM
import time
import sys
from tuneThreshold import *
from torch.cuda.amp import autocast, GradScaler
from HuCapa_trainer import evaluateFromList


class EcapaTrainer(nn.Module):
    def __init__(
        self, 
        max_frames: int=200, 
        device:torch.device=torch.device("cuda"), 
        batch_size:int = 32, 
        num_wokers:int=16, 
        seed:int=10,
        lr:float=1e-3,
        test_step:int=1,
        scheduler_type = "step",
        lr_decay:float = 0.97,
        loss_classes:int = 5994,
        train_list:str = "data/train_list_debug.txt",
        train_path:str = "data/voxceleb2" ,
        test_path:str = "./data/voxceleb1/",
        test_file:str = "./data/test_list.txt" 
    ) -> None:
        super().__init__()
        self.device = device
        self.model = CustomEcapa(80)
        self.model.to(device)
        train_dataset = train_dataset_loader(
            train_list=train_list,
            augment=True,
            musan_path="./data/musan_split",
            rir_path="./data/RIRS_NOISES/simulated_rirs",
            max_frames=max_frames,
            train_path=train_path
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
        self.test_file = test_file
        self.test_path = test_path
        self.loss_classes = loss_classes
        self.loss = AAM(nOut=192, nClasses=loss_classes, margin=0.2, scale=30)
        self.loss.to(device)
        self.optim = torch.optim.Adam([{"params": self.model.parameters()}, {"params": self.loss.parameters(), "weight_decay": 2e-4}], lr=lr, weight_decay=2e-5)
        if scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        elif scheduler_type == "cycle":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=1e-8, max_lr=1e-6, step_size_up=2729, mode="triangular2", cycle_momentum=False)
        self.scaler = GradScaler()
        

        print(time.strftime("%m-%d %H:%M:%S") + " Overall parameters: = %.2f"%(sum(param.numel() for param in self.model.parameters())))
        print(f"Learnable model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(f"Learnable loss parameters: {sum(p.numel() for p in self.loss.parameters() if p.requires_grad)}")
        print(f"Sched type: {self.scheduler.__class__.__name__}")

    def load_params(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["ecapa"])
        self.optim.load_state_dict(checkpoint["opt"])
        if self.loss_classes == 5994:
            self.loss.load_state_dict(checkpoint["loss"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        return checkpoint["epoch"]

    def train(self, epoch):
        self.model.train()
        self.loss.train()
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]["lr"]
        for num, (data, labels) in enumerate(self.dataloader, start=1):
            self.zero_grad()
            data = data.to(self.device)
            data = data.squeeze(1)
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            with autocast():
                out = self.model(data)
                out = out.squeeze(1)
                nloss, prec = self.loss(out, labels)
            self.scaler.scale(nloss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / self.dataloader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
            sys.stderr.flush()
            if self.scheduler.__class__.__name__ != "StepLR":
                self.scheduler.step()
        if self.scheduler.__class__.__name__ == "StepLR":
            self.scheduler.step()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index*len(labels)


    def eval(self, epoch):
        self.model.eval()
        sc, lab, _ = evaluateFromList(self.test_file, self.test_path, self.model, self.dataloader.num_workers)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1])
        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)
        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(epoch, result[1], mindcf))
        return result[1], mindcf