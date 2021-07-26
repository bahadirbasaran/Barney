""" Environment Setup """

import os
import sys

import pandas  as pd
import numpy   as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from multiprocessing import cpu_count

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy

import model_subsystem

pl.seed_everything(42)
sns.set_theme()

FPS = 35
NUM_FEATURES  = 2
NUM_CLASSES   = 2
NUM_EPOCHS    = 100
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 5  
EARLY_STOP_AFTER_EPOCHS = 5
SMOOTHING_WINDOW_LENGTH = 18

""" Utilization Functions """

def createSequences(inputData: pd.DataFrame, targetData: pd.DataFrame, targetName: str, sequenceLength: int):
  
  sequences = []

  for idx in tqdm(range(0, len(inputData) - sequenceLength, sequenceLength)):
    sequence = inputData[idx : idx + sequenceLength]
    label    = target_data.iloc[idx + sequenceLength][targetName]
    sequences.append((sequence, int(label)))

  return sequences

def plotConfisuonMatrix(confusionMatrix, xlabel='Predicted Gamestage', ylabel='True Gamestage'):
  heatMap = sns.heatmap(confusionMatrix, annot=True, fmt='f', cmap="Blues")
  heatMap.yaxis.set_ticklabels(heatMap.yaxis.get_ticklabels(), rotation=0, ha='right')
  heatMap.xaxis.set_ticklabels(heatMap.xaxis.get_ticklabels(), rotation=0, ha='right')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def runTest(trainedModel: model_subsystem.GamestageClassifier, testDataset: model_subsystem.FrameDataset):

  predictions, labels = [], []

  for item in tqdm(testDataset):
    sequence, label = item["sequence"], item["label"]
    _, output = trainedModel(sequence.unsqueeze(dim=0))
    prediction = torch.argmax(output, dim=1)
    predictions.append(prediction.item())
    labels.append(label.item())

  confusionMatrix = pd.DataFrame(confusion_matrix(labels, predictions, normalize='true'), index = ["Exploration", "Combat"], columns = ["Exploration", "Combat"])
  plotConfisuonMatrix(confusionMatrix)

if __name__ == '__main__':

  pathDataset = sys.argv[1]  

  # Read dataset from input path
  data = pd.read_csv(pathDataset)

  # Bitrate Calculation in kbps
  data['bitrateKbps']    = data['size'].rolling(window=FPS).sum() * (8 // 1000)

  # Bitrate smoothing
  data['bitrateKbpsEMA'] = data['bitrateKbps'].ewm(span=SMOOTHING_WINDOW_LENGTH, adjust=False).mean()

  # Gamestage smoothing
  data['gamestageEMA']   = data['gamestage'].ewm(span=SMOOTHING_WINDOW_LENGTH, adjust=False).mean()

  # Filtering gamestage after smoothing
  for i in tqdm(range(data.shape[0])):
    data.loc[i, 'gamestageEMA'] = 1 if data.loc[i, 'gamestageEMA'] > 0.5 else 0

  # Ripping first 34 samples that does not consist bitrate value out
  data = data.loc[FPS-1:]
  data.reset_index(drop=True, inplace=True)

  X_train, y_train = data[['bitrateKbpsEMA']], data[['gamestageEMA']]
  X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, shuffle=False)
  X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size = 0.2, shuffle=False)


  """ Insights About Data """

  print("Training Data #Exploration Frames: {}\tTraining Data #Combat Frames: {}"
        .format(y_train[(y_train['gamestageEMA'] == 0)].shape[0], y_train[(y_train['gamestageEMA'] == 1)].shape[0]))

  data['gamestageEMA'].value_counts().plot(kind='bar', rot=0, color=['lightsteelblue', 'maroon'], width=0.6)
  plt.xlabel("Game Stages")
  plt.xticks((0, 1), ("Exploration", "Combat"))
  plt.ylabel("Number of Frames")
  plt.show()

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  sets = ['Training Set', 'Validation Set', 'Test Set']
  counts = [len(X_train), len(X_val), len(X_test)]
  ax.bar(sets,counts, width=0.6, color=['lightsteelblue', 'maroon', 'darkorange'])
  plt.xlabel("Data Splits")
  plt.ylabel("Number of Frames")
  plt.show()


  """ Data Preprocessing """

  scaler = StandardScaler()   
  scaler = scaler.fit(X_train)

  X_train_scaled = pd.DataFrame(data    = scaler.transform(X_train),
                              index   = X_train.index,
                              columns = X_train.columns)

  X_val_scaled   = pd.DataFrame(data    = scaler.transform(X_val),
                              index   = X_val.index,
                              columns = X_val.columns)

  X_test_scaled  = pd.DataFrame(data    = scaler.transform(X_test),
                              index   = X_test.index,
                              columns = X_test.columns)

  """ Data Sequence Creation """

  trainSequences = createSequences(X_train_scaled, y_train, 'gamestageEMA', SEQUENCE_LENGTH)    
  valSequences   = createSequences(X_val_scaled,   y_val,   'gamestageEMA', SEQUENCE_LENGTH)
  testSequences  = createSequences(X_test_scaled,  y_test,  'gamestageEMA', SEQUENCE_LENGTH)

  """ Model Initiation """

  dataModule = DataModule(trainSequences, valSequences, testSequences, batch_size=BATCH_SIZE)

  model = GamestageClassifier(n_features = NUM_FEATURES, n_classes = NUM_CLASSES) 

  callbackCheckpoint = ModelCheckpoint(
    dirpath  = "checkpoints",
    filename = "best_checkpoint",
    save_top_k = 1,
    verbose  = True,
    monitor  = "val_loss",
    mode     = "min"    
  )

  callbackEarlyStopping = EarlyStopping(monitor="val_loss", patience=EARLY_STOP_AFTER_EPOCHS, min_delta=0.001)

  logger = TensorBoardLogger("classification_logs", name="ClassificationLogs")

  trainer = pl.Trainer(
    logger = logger,
    checkpoint_callback = callbackCheckpoint,
    callbacks = [callbackEarlyStopping],
    max_epochs = NUM_EPOCHS,
    gpus = 0,
    progress_bar_refresh_rate = 30
  )

  """ Training Initiation """

  trainer.fit(model, dataModule)

  trainedModel = GamestagePredictor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, n_features = NUM_FEATURES, n_classes = NUM_CLASSES)
  trainedModel.freeze()

  """ Testing the Model on the Test Dataset """

  run_test(trainedModel, dataModule.testDataset)
