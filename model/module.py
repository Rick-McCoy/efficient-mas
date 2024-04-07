from lightning import LightningModule
from torch.nn import functional as F
from torch.optim import Adam, AdamW

from model.aligner import Aligner


class AlignerModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = Aligner(cfg)
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        match self.cfg.optim:
            case "Adam":
                return Adam(self.parameters(), lr=self.cfg.lr)
            case "AdamW":
                return AdamW(self.parameters(), lr=self.cfg.lr)
