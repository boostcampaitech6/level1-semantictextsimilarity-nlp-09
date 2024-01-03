import os
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
import transformers

# Import your model modules
import kyelectra_train as kykim
import MSE_Huber_snelectra_train as snunlp
import MSE_sroberta_train as sroberta

# 체크포인트 경로를 지정합니다.
models_and_checkpoints = [
    (snunlp,'snunlp/KR-ELECTRA-discriminator' ,'/data/ephemeral/home/model/MSE+Huber_snunlp=0-epoch=10-val_pearson=0.93.ckpt'),
    (sroberta, 'jhgan/ko-sroberta-multitask', '../model/MSE_sroberta-epoch=7-val_pearson=0.9111.ckpt'),
    (kykim, 'kykim/electra-kor-base' ,'../model/kykim-epoch=13-val_pearson=0.9076.ckpt')
    ]

# 각 모델의 이름을 리스트로 지정합니다.
model_names = ['snunlp', 'sroberta', 'kykim']

def load_and_predict(model_module, model_name, checkpoints, args):
    dataloader = model_module.Dataloader(
        model_name, 
        args.batch_size, 
        args.shuffle, 
        args.train_path, 
        args.dev_path,
        args.test_path,
        args.predict_path
    )
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)
    model = model_module.Model.load_from_checkpoint(checkpoints)
    
    # # 모델의 상태 딕셔너리를 출력합니다.
    # print(model.state_dict().keys())
    
    predictions = trainer.predict(model=model, datamodule=dataloader)
    return np.array([round(float(i), 1) for i in torch.cat(predictions)])

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_epoch', default=1, type=int)
parser.add_argument('--alpha', default= 0.5, type=float)
parser.add_argument('--shuffle', default=True)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--train_path', default='../data/train.csv')
parser.add_argument('--dev_path', default='../data/dev.csv')
parser.add_argument('--test_path', default='../data/dev.csv')

## valid set에 대해 예측
parser.add_argument('--predict_path', default='../data/dev.csv')
args = parser.parse_args()

predictions = []

for model_module, model_name, checkpoints in models_and_checkpoints:
    prediction = load_and_predict(model_module, model_name, checkpoints, args)
    predictions.append(prediction)
  


# Validation set의 실제 목표값을 로드합니다.
valid_targets = pd.read_csv(args.dev_path)['label']
valid_sen1 = pd.read_csv(args.dev_path)['sentence_1']
valid_sen2 = pd.read_csv(args.dev_path)['sentence_2']


# 각 모델의 예측 점수와 실제 라벨을 DataFrame으로 만듭니다.
df_predictions = pd.DataFrame(dict(zip(model_names, predictions)))
df_predictions['label'] = valid_targets
df_predictions['sentence_1'] = valid_sen1
df_predictions['sentence_2'] = valid_sen2

# DataFrame을 csv 파일로 저장합니다.
df_predictions.to_csv('predictions_and_labels.csv', index=False)