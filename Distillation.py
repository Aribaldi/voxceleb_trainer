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
from models.RawNet3 import RawNet3
from models.ResNetSE34L import ResNetSE
import argparse
from models.RawNetBasicBlock import Bottle2neck
from models.ResNetBlocks import SEBasicBlock

parser = argparse.ArgumentParser(description="Distillation pipeline wrapper")

EXPS_PATH = Path("./exps/")
all_exps = EXPS_PATH.iterdir()
latest_exp = max(all_exps, key=os.path.getctime)


parser.add_argument("epochs_num", type=int)
parser.add_argument("--test_every", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--exp_name", nargs="?", type=str, default="ignore")
parser.add_argument("--train_list", nargs="?", type=str, default="data/train_list.txt")
parser.add_argument("--train_path", nargs="?", type=str, default="data/voxceleb2")
parser.add_argument("--student_model", nargs="?", type=str, default="ecapa")
parser.add_argument("--max_frames", type=int, default=200)
parser.add_argument("--cpt_path", nargs="?", type=str, default="", const=f"{latest_exp}/{latest_exp.stem}_cp.tar")


def main(args):
    start_epoch = 0
    exp_name = args.exp_name
    device = torch.device("cuda")
    teacher = HuCapa(device=device)

    if args.student_model == "ecapa":
        teacher = ECAPA_TDNN(512)
    elif args.student_model == "rawnet":
        student = RawNet3(
            Bottle2neck, 
            model_scale=8, 
            context=True, 
            summed=True, 
            out_bn=False, 
            log_sinc=True, 
            norm_sinc="mean", 
            grad_mult=1, 
            nOut=192,
            encoder_type="ECA",
            sinc_stride=10
        )
    else:
        num_filters = [16, 32, 64, 128]
        student =  ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, 192)

    student.to(device)

    checkpoint = torch.load("./exps/HuCapa/cyclic_sched_2/16-24/16-24_cp.tar")
    teacher.ecapa.load_state_dict(checkpoint["ecapa"])
    teacher.hs_weights.load_state_dict(checkpoint["hs_weights"])

    train_dataset = train_dataset_loader(
    train_list=args.train_list,
    augment=True,
    musan_path="./data/musan_split",
    rir_path="./data/RIRS_NOISES/simulated_rirs",
    max_frames=args.max_frames,
    train_path=args.train_path,
    )
    train_sampler = train_dataset_sampler(
        train_dataset,
        nPerSpeaker=1, 
        max_seg_per_spk=500, 
        batch_size=args.batch_size,
        distributed=False,
        seed=10
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
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

    if args.cpt_path:
        exp_name = Path(args.cpt_path).parent.parent.stem
        checkpoint = torch.load(args.cpt_path)
        model.load_state_dict(checkpoint["ecapa"])
        start_epoch = checkpoint["epoch"]
    
    save_path = Path(f"exps/distill/{exp_name}/{start_epoch}-{start_epoch + args.epochs_num - 1}")
    os.makedirs(save_path, exist_ok=True)
    scorefile = open(save_path/ "scores.txt", "a+")


    optim = torch.optim.Adam(model.model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.97)
    loss = AAM(nOut=192, nClasses=5994, margin=0.2, scale=30)
    loss.to(device)
    scaler = GradScaler()


    def eval(model):
        model.eval()
        sc, lab, _ = evaluateFromList("./data/test_list.txt", "./data/voxceleb1/", model, 16)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1])
        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)
        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(epoch, result[1], mindcf))
        return result[1], mindcf
    

    start_epoch +=1
    for epoch in range(start_epoch, start_epoch + args.epochs_num):
        dataloader.sampler.set_epoch(epoch)
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
                out = out.squeeze(1)
                nloss, prec = loss(out, labels)
                nloss = compression_manager.callbacks.on_after_compute_loss(data, out, nloss)
            scaler.scale(nloss).backward()
            scaler.step(optim)
            scaler.update()
            g_loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + "[%2d] Training: %.2f%%, " %(epoch, 100 * (num / dataloader.__len__())) + \
            " Loss: %.5f \r"        %(g_loss/num))
            sys.stderr.flush()
        if epoch % args.test_every == 0:
            val_eer, dcf = eval(model)
            torch.save({
                "ecapa": model.state_dict(),
                "opt": optim.state_dict(),
                "epoch": epoch,
                "loss": loss.state_dict(),
                "scaler": scaler.state_dict()
            }, save_path / f"{start_epoch}-{start_epoch + args.epochs_num - 1}_{args.student_model}_cp.tar")
            scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(epoch, val_eer, dcf))
            scorefile.flush()

        sys.stdout.write("\n")
        scheduler.step()

    scorefile.close()


    


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parser.parse_args()
    main(args)


