from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer

from dataset_def import CocoDetection

# Sources:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

# config = DetrConfig(use_pretrained_backbone=True,
#                     backbone = "resnet50")
# model = DetrForObjectDetection(config)

# print(model)


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

train_dataset = CocoDetection(img_folder='datasets/coco/d1/train', processor=processor)
val_dataset = CocoDetection(img_folder='datasets/coco/d1/valid', processor=processor, train=False)


from torch.utils.data import DataLoader

# Create PyTorch dataloaders, which allow us to get batches of data
def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
batch = next(iter(train_dataloader))

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        cats = train_dataset.coco.cats
        id2label = {k: v['name'] for k,v in cats.items()}

        config = DetrConfig(use_pretrained_backbone=True,
                            backbone = "resnet50",
                            num_labels=len(id2label),
                            ignore_mismatched_sizes=True
                            )
        self.model = DetrForObjectDetection(config)
        # 
        # self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
        #                                                    revision="no_timm",
        #                                                    num_labels=len(id2label),
        #                                                    ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

    def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader
     

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
print(outputs.logits.shape)

trainer = Trainer(max_steps=300, 
                gradient_clip_val=0.1,
                max_epochs=1,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu'
                )
trainer.fit(model)

model.model.save_pretrained('DETR/')
