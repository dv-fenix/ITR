from config import preEncDec
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import pytorch_lightning as pl

from eng_utils import gen_model_loaders, flat_accuracy, save_model, build_model


class itr_Net(pl.LightningModule):
    def __init__(self, hparams):
        super(itr_Net, self).__init__()
        self.config = preEncDec
        self.encoder, self.decoder, _ = build_model(self.config)
        self.train_loss = 0
        self.eval_loss = 0
        self.eval_acc = 0
        self.avg_train_loss = 0
        self.avg_val_loss = 0
        self.avg_val_acc = 0
        self.epoch = 0
        self.eval_steps = 0

    def forward(self, encoder_input_ids, decoder_input_ids):
        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states,
                                    masked_lm_labels=decoder_input_ids)
        self.loss = loss
        self.logits = logits
        return loss, logits

    def save(self, output):
        save_model(self.encoder, output.encoder)
        save_model(self.decoder, output.decoder)

    def pad_seq(self):
        pad_sequence = PadSequence(self.tokenizers.src.pad_token_id, self.tokenizers.tgt.pad_token_id)

        return pad_sequence

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return eval_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def at_epoch_end(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=self.config.lr)
        lr_scheduler.step()

    def training_step(self, batch, batch_nb):
        source, target = batch
        loss, _ = self.forward(source, target)
        self.train_loss += loss.item()
        tensorboard_logs = {'loss': loss}

        return {'loss': loss, 'logs': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        source, target = batch
        with torch.no_grad():
            loss, logits = self.forward(source, target)
        self.eval_loss += loss
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        self.eval_acc += tmp_eval_accuracy
        tensorboard_logs = {'eval_loss': self.eval_loss, 'eval_acc': self.eval_acc}

        return {'eval_loss': self.eval_loss, 'eval_acc': self.eval_acc, 'logs': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        self.epoch += 1
        current_epoch = self.epoch - 1
        self.avg_val_acc += self.eval_acc / len(eval_loader)
        self.avg_val_loss = self.eval_loss / len(eval_loader)

        self.avg_train_loss = self.train_loss / len(train_loader)
        writer.add_scalar('Avg_Train_loss', self.avg_train_loss, current_epoch)
        writer.add_scalar('Avg_Validation_Acc', self.avg_val_acc, current_epoch)
        writer.add_scalar('Avg_Validation_Loss', self.avg_val_loss, current_epoch)
        for weights in self.parameters():
            writer.add_histogram('Weights', weights, current_epoch)

        tensorboard_logs = {'avg_valid_loss': self.avg_val_loss, 'avg_valid_acc': self.avg_val_acc,
                            'avg_train_loss': self.avg_train_loss}
        return {'avg_valid_loss': self.avg_val_loss, 'avg_valid_acc': self.avg_val_acc,
                'avg_train_loss': self.avg_train_loss, 'logs': tensorboard_logs}



