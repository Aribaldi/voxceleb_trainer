from Ecapa_trainer import EcapaTrainer
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import os
import torch


EXPS_PATH = Path("./exps/Ecapa")
all_exps = EXPS_PATH.iterdir()
latest_exp = max(all_exps, key=os.path.getctime)

parser = argparse.ArgumentParser(description="ECAPA-TDNN trainer")
parser.add_argument("epochs_num", type=int)
parser.add_argument("--test_every", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--cpt_path", nargs="?", type=str, default="", const=f"{latest_exp}/{latest_exp.stem}_cp.tar")
parser.add_argument("--exp_name", nargs="?", type=str, default="ignore")
parser.add_argument("--train_list", nargs="?", type=str, default="data/train_list.txt")
parser.add_argument("--test_list", nargs="?", type=str, default="data/test_list.txt")
parser.add_argument("--train_path", nargs="?", type=str, default="data/voxceleb2")
parser.add_argument("--test_path", nargs="?", type=str, default="./data/voxceleb1/")
parser.add_argument("--sched_type", nargs="?", type=str, default="step")


def main(args):
    start_epoch = 0
    exp_name = args.exp_name
    trainer = EcapaTrainer(
        batch_size=args.batch_size, 
        test_step=args.test_every, 
        scheduler_type=args.sched_type,
        train_list=args.train_list,
        train_path=args.train_path,
        test_file=args.test_list,
        test_path=args.test_path
    )
    if args.cpt_path:
        start_epoch = trainer.load_params(args.cpt_path)
        exp_name = args.cpt_path.parent.stem
    start_epoch +=1
    save_path = Path(f"exps/Ecapa/{exp_name}/{start_epoch}-{start_epoch + args.epochs_num - 1}")
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=save_path.parent / "tb_logs")
    scorefile = open(save_path.parent / "scores.txt", "a+")
    for epoch in range(start_epoch, start_epoch + args.epochs_num):
        trainer.dataloader.sampler.set_epoch(epoch)
        loss, lr, acc = trainer.train(epoch)
        writer.add_scalar("Train/loss", loss, epoch)
        writer.add_scalar("Train/acc", acc, epoch)
        writer.add_scalar("LR", lr, epoch)
        scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(epoch, acc.item(), loss, lr))

        if epoch % args.test_every == 0:
            val_eer, dcf = trainer.eval(epoch)
            writer.add_scalar("Val/EER", val_eer, epoch)
            writer.add_scalar("Val/minDCF", dcf, epoch)
            scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(epoch, val_eer, dcf))
            torch.save({
                "ecapa": trainer.model.state_dict(),
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