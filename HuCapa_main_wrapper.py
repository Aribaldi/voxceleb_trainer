from HuCapa_trainer import HuCapaTrainer
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import os
import torch


EXPS_PATH = Path("./exps/HuCapa")
all_exps = EXPS_PATH.iterdir()
latest_exp = max(all_exps, key=os.path.getctime)

parser = argparse.ArgumentParser(description="Speaker ID model with HuBERT as a backbone and ECAPA-TDNN as a prediction head")
parser.add_argument("epochs_num", type=int)
parser.add_argument("--test_every", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cpt_path", nargs="?", type=str, default="", const=f"{latest_exp}/{latest_exp.stem}_cp.tar")
parser.add_argument("--sched_type", nargs="?", type=str, default="step")


def main(args):
    start_epoch = 0
    trainer = HuCapaTrainer(batch_size=args.batch_size, test_step=args.test_every, scheduler_type=args.sched_type)
    if args.cpt_path:
        start_epoch = trainer.load_params(args.cpt_path)
    start_epoch +=1
    save_path = Path(f"exps/HuCapa/cyclic_sched_2/{start_epoch}-{start_epoch + args.epochs_num - 1}")
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=save_path.parent / "tb_logs")
    #data, _ = next(iter(trainer.dataloader))
    #data = data.squeeze(1)
    #data = data.to(trainer.device)
    #writer.add_graph(trainer.model, data)
    #writer.close()
    scorefile = open(save_path.parent / "scores.txt", "a+")
    for epoch in range(start_epoch, start_epoch + args.epochs_num):
        trainer.dataloader.sampler.set_epoch(epoch)
        loss, lr, acc = trainer.train(epoch)
        writer.add_scalar("Train/loss", loss, epoch)
        writer.add_scalar("Train/acc", acc, epoch)
        writer.add_scalar("LR", lr, epoch)
        print(f"hidden states weights: {trainer.model.hs_weights.weight}")
        print(f"hidden states weights: {trainer.loss.weight}")
        scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(epoch, acc.item(), loss, lr))

        if epoch % args.test_every == 0:
            val_eer, dcf = trainer.eval(epoch)
            writer.add_scalar("Val/EER", val_eer, epoch)
            writer.add_scalar("Val/minDCF", dcf, epoch)
            scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(epoch, val_eer, dcf))
            torch.save({
                "ecapa": trainer.model.ecapa.state_dict(),
                "hs_weights": trainer.model.hs_weights.state_dict(),
                "opt": trainer.optim.state_dict(),
                "epoch": epoch,
                "loss": trainer.loss.state_dict(),
                "scaler": trainer.scaler.state_dict()
            }, save_path / f"{start_epoch}-{start_epoch + args.epochs_num - 1}_cp.tar")
            scorefile.flush()
    scorefile.close()

    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parser.parse_args()
    main(args)