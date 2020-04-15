from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
import torch
import pytorch_lightning as pl
from config import preEncDec
from eng_utils import gen_model_loaders
import model as M

def preproc():
  from data import split_data
  split_data('../data/hin-eng/hin.txt', '../data/hin-eng')

if __name__=='__main__':
    pre_proc()
    train_loader, eval_loader = gen_model_loaders(preEncDec)
    hparams = preEncDec.lr
    writer = SummaryWriter()
    net = M.itr_Net(hparams)
    logger = TensorBoardLogger("tb_logs", name="translation_model")
    trainer = pl.Trainer(gpus=1, max_epochs = preEncDec.epochs, logger = logger)
    trainer.fit(net)
    writer.close()