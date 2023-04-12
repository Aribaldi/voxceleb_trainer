from models.HuCapa import HuCapa
from Ecapa_trainer import EcapaTrainer
from HuCapa_trainer import HuCapaTrainer
from models.CustomEcapa import ECAPA_TDNN
from neural_compressor.training import prepare_compression
from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig, IntermediateLayersKnowledgeDistillationLossConfig
from DatasetLoader import train_dataset_loader, train_dataset_sampler, worker_init_fn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from loss.aamsoftmax import LossFunction as AAM
import sys
from HuCapa_trainer import evaluateFromList
from tuneThreshold import *
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path


save_path = Path("./exps/distill_reduced_ecapa_128/")

def train(num_epochs = 20, batch_size = 364, max_frames = 200):
    device = torch.device("cuda")
    teacher = HuCapa(device=device)
    student = ECAPA_TDNN(128)
    student.to(device)

    checkpoint = torch.load("./exps/HuCapa/cyclic_sched_2/16-24/16-24_cp.tar")
    teacher.ecapa.load_state_dict(checkpoint["ecapa"])
    teacher.hs_weights.load_state_dict(checkpoint["hs_weights"])

    train_dataset = train_dataset_loader(
    train_list="data/train_list.txt",
    augment=True,
    musan_path="./data/musan_split",
    rir_path="./data/RIRS_NOISES/simulated_rirs",
    max_frames=max_frames,
    train_path="data/voxceleb2",
    )
    train_sampler = train_dataset_sampler(
        train_dataset,
        nPerSpeaker=1, 
        max_seg_per_spk=500, 
        batch_size=batch_size,
        distributed=False,
        seed=10
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    distillation_criterion = KnowledgeDistillationLossConfig(loss_types=["CE", "MSE"], loss_weights=[0, 1])
    conf = DistillationConfig(teacher_model=teacher, criterion=distillation_criterion, optimizer={"SGD": {"learning_rate": 0.0001}})
    compression_manager = prepare_compression(student, conf)
    compression_manager.callbacks.on_train_begin()
    model = compression_manager.model


    optim = torch.optim.Adam(model.model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.97)
    loss = AAM(nOut=192, nClasses=5994, margin=0.2, scale=30)
    loss.to(device)
    scaler = GradScaler()

    scorefile = open(save_path/ "scores.txt", "a+")

    def eval(model):
        model.eval()
        sc, lab, _ = evaluateFromList("./data/test_list.txt", "./data/voxceleb1/", model, 16)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1])
        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)
        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(epoch, result[1], mindcf))
        return result[1], mindcf

    for epoch in range(1, num_epochs + 1):
        model.train()
        loss.train()
        g_loss = 0
        for num, (data, labels) in enumerate(dataloader, start=1):
            model.zero_grad()
            data = data.to(device)
            data = data.squeeze(1)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            with autocast():  
                out = model(data)
                print(out.shape)
                out = out.squeeze(1)
                print(out.shape)
                print("#"*128)
                nloss, prec = loss(out, labels)
                nloss = compression_manager.callbacks.on_after_compute_loss(data, out, nloss)
            scaler.scale(nloss).backward()
            scaler.step(optim)
            scaler.update()
            g_loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + "[%2d] Training: %.2f%%, " %(epoch, 100 * (num / dataloader.__len__())) + \
            " Loss: %.5f \r"        %(g_loss/num))
            sys.stderr.flush()
        if epoch % 2 == 0:
            val_eer, dcf = eval(model)
            torch.save({
                "ecapa": model.state_dict(),
                "opt": optim.state_dict(),
                "epoch": epoch,
                "loss": loss.state_dict(),
                "scaler": scaler.state_dict()
            }, save_path / f"0-20_distill_ecapa_cp.tar")
            scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(epoch, val_eer, dcf))
            scorefile.flush()

        sys.stdout.write("\n")
        scheduler.step()

    scorefile.close()


    


if __name__ == "__main__":
    train()


