from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, Accuracy
# from .backbones.encoder import ImageEncoder
from methods.backbones.encoder import ImageEncoder


class FinetuneClassifier(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Extract values from the config
        self.num_classes = config["cls"]["num_classes"]
        self.in_features = config["cls"]["in_features"]
        self.hidden_dim = config["cls"].get("hidden_dim", None)
        self.dropout = config["cls"].get("dropout", 0.0)
        self.learning_rate = config["cls"].get("learning_rate", 5e-4)
        self.weight_decay = config["cls"].get("weight_decay", 1e-6)
        self.multilabel = config["cls"].get("multilabel", False)
        self.model_name = config["cls"].get("model_name", "resnet_50")

        # Initialize metrics based on task type
        if self.multilabel and self.num_classes > 1:
            self.train_auc = AUROC(task='multilabel', num_labels=self.num_classes)
            self.val_auc = AUROC(task='multilabel', num_labels=self.num_classes, compute_on_step=False)
            self.test_auc = AUROC(task='multilabel', num_labels=self.num_classes, compute_on_step=False)
        else:
            task_type = 'binary' if self.num_classes == 2 else 'multiclass'
            # self.train_auc = AUROC(task=task_type, num_classes=self.num_classes)
            # self.val_auc = AUROC(task=task_type, num_classes=self.num_classes, compute_on_step=False)
            # self.test_auc = AUROC(task=task_type, num_classes=self.num_classes, compute_on_step=False)
            self.train_acc = Accuracy(task=task_type,num_classes=self.num_classes)
            self.val_acc = Accuracy(task=task_type,num_classes=self.num_classes, compute_on_step=False)
            self.test_acc = Accuracy(task=task_type,num_classes=self.num_classes, compute_on_step=False)

        # Initialize the backbone and classification head
        self.img_encoder_q = ImageEncoder(model_name=config["cls"]["backbone"], output_dim=config["cls"]['embed_dim'])
        
        for param in self.img_encoder_q.parameters():
            param.requires_grad = False

        self.classification_head = ClassificationHead(
            n_input=self.in_features, n_classes=self.num_classes, p=self.dropout, n_hidden=self.hidden_dim
        )

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.img_encoder_q.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, preds, y = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            auc = self.train_auc(torch.sigmoid(logits).float(), y.long())
            self.log("train_auc_step", auc, prog_bar=True, sync_dist=True)
            self.log("train_auc_epoch", self.train_auc, prog_bar=True, sync_dist=True)
        else:
            acc = self.train_acc(preds, y.long())
            self.log("train_acc_step", acc, prog_bar=True, sync_dist=True)
            self.log("train_acc_epoch", self.train_acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        loss, logits,preds, y = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            self.val_auc(torch.sigmoid(logits).float(), y.long())
            self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            # self.val_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.val_acc(preds, y.long())
            self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, preds, y = self.shared_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        if self.multilabel:
            self.test_auc(torch.sigmoid(logits).float(), y.long())
            self.log("test_auc", self.test_auc, on_epoch=True)
        else:
            self.test_acc(preds, y.long())
            self.log("test_acc", self.test_acc, on_epoch=True)
        return loss

    def shared_step(self, batch):
        x, y = batch #x:(b,3,224,224), y:(b,1)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            feats, _ = self.img_encoder_q(x) #resnet50:(48,2048)
        feats = feats.view(feats.size(0), -1)  #resnet50:(48,2048)
        logits = self.classification_head(feats) #resnet50:(48,2048)->(48,1)
        
        # import pdb; pdb.set_trace()
        # Loss calculation
        if self.multilabel:
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())
            preds = (torch.sigmoid(logits) > 0.5).float()
        else:
            y = y.squeeze()
            loss = F.cross_entropy(logits.float(), y.long())
            preds = torch.argmax(logits, dim=1)
        
        return loss, logits, preds, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classification_head.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        dataset_size = len(dm.train_dataloader())
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * trainer.max_epochs


class ClassificationHead(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
        super().__init__()
        if n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes)
            )
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes)
            )

    def forward(self, x):
        return self.block_forward(x)


class Flatten(nn.Module):
    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
