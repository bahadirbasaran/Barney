import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class FrameDataset(Dataset):

  def __init__(self, sequences):
    self.sequences = sequences

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence, label = self.sequences[idx]

    return dict(
        sequence = torch.Tensor(sequence.to_numpy()),
        label    = torch.tensor(label).long()
    )

class DataModule(pl.LightningDataModule):
  
  def __init__(self, train_sequences, val_sequences, test_sequences, batch_size):
    super().__init__()
    self.trainSequences = train_sequences
    self.valSequences   = val_sequences
    self.testSequences  = test_sequences
    self.batchSize      = batch_size

  def setup(self, stage=None):
    self.trainDataset = FrameDataset(self.trainSequences)
    self.valDataset   = FrameDataset(self.valSequences)
    self.testDataset  = FrameDataset(self.testSequences)

  def train_dataloader(self):
    return DataLoader(self.trainDataset, batch_size=self.batchSize, shuffle=False, num_workers=cpu_count())
  
  def val_dataloader(self):
    return DataLoader(self.valDataset, batch_size=self.batchSize, shuffle=False, num_workers=cpu_count())

  def test_dataloader(self):
    return DataLoader(self.testDataset, batch_size=self.batchSize, shuffle=False, num_workers=cpu_count())   

class ModuleLSTM(nn.Module):
  
  def __init__(self, n_features, n_classes, n_hidden=128, n_layers=2):
    super().__init__()

    self.lstm = nn.LSTM(
        input_size  = n_features,
        hidden_size = n_hidden,
        num_layers  = n_layers,
        batch_first = True,
        dropout     = 0.5
    )              
    
    self.classifier = nn.Linear(n_hidden, n_classes)  
  
  def forward(self, x):
    self.lstm.flatten_parameters()
    _, (hidden, _) = self.lstm(x)

    # Since it is a multilayer network, fetch output of the last layer
    out = hidden[-1]
    return self.classifier(out)

class GamestageClassifier(pl.LightningModule):
    
  def __init__(self, n_features, n_classes):
    super().__init__()
    self.model = ModuleLSTM(n_features, n_classes) 
    self.criterion = nn.CrossEntropyLoss()   

  def forward(self, x, labels=None):
    output = self.model(x)
    loss = 0
    if labels is not None:
      loss = self.criterion(output, labels)
    
    return loss, output

  def training_step(self, batch, batch_idx):               
    sequences, labels = batch["sequence"], batch["label"]

    # Pass the data to the forward method
    loss, outputs = self(sequences, labels)

    # Get the maximum values (the biggest class probabilities) from the outputs
    predictions = torch.argmax(outputs, dim=1)

    stepAccuracy = accuracy(predictions, labels)

    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("train_accuracy", stepAccuracy, prog_bar=True, logger=True)

    return {"loss": loss, "accuracy": stepAccuracy}

  def validation_step(self, batch, batch_idx):               
    sequences, labels = batch["sequence"], batch["label"]

    loss, outputs = self(sequences, labels)

    predictions = torch.argmax(outputs, dim=1)

    stepAccuracy = accuracy(predictions, labels)

    self.log("val_loss", loss, prog_bar=True, logger=True)
    self.log("val_accuracy", stepAccuracy, prog_bar=True, logger=True)

    return {"loss": loss, "accuracy": stepAccuracy}

  def test_step(self, batch, batch_idx):               
    sequences, labels = batch["sequence"], batch["label"]

    loss, outputs = self(sequences, labels)

    predictions = torch.argmax(outputs, dim=1)

    stepAccuracy = accuracy(predictions, labels)

    self.log("test_loss", loss, prog_bar=True, logger=True)
    self.log("test_accuracy", stepAccuracy, prog_bar=True, logger=True)

    return {"loss": loss, "accuracy": stepAccuracy}

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=LEARNING_RATE)