import os
import argparse
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd

# Import your model modules
import kyelectra_train as kykim
import MSE_Huber_snelectra_train as snunlp
import MSE_sroberta_train as sroberta

from scipy.stats import pearsonr

# 체크포인트 경로를 지정합니다.
models_and_checkpoints = [
    (snunlp,'snunlp/KR-ELECTRA-discriminator' ,'/data/ephemeral/home/model/MSE+Huber_snunlp=0-epoch=10-val_pearson=0.93.ckpt'),
    (sroberta, 'jhgan/ko-sroberta-multitask', '../model/MSE_sroberta-epoch=7-val_pearson=0.9111.ckpt'),
    (kykim, 'kykim/electra-kor-base' ,'../model/kykim-epoch=13-val_pearson=0.9076.ckpt')
    ]

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
## devset 검증
parser.add_argument('--predict_path', default='../data/dev.csv')
args = parser.parse_args()

predictions = []

for model_module, model_name, checkpoints in models_and_checkpoints:
    prediction = load_and_predict(model_module, model_name, checkpoints, args)
    predictions.append(prediction)
  
# 모든 모델의 예측을 평균내어 최종 예측치를 생성합니다.
weights = [0.50, 0.20, 0.30]  # 각 모델에 대한 가중치
weighted_predictions = [weight * pred for weight, pred in zip(weights, predictions)]
final_prediction = np.sum(weighted_predictions, axis=0)
# 각 예측값을 소수점 첫 번째 자리까지 반올림합니다.
final_prediction = np.round(final_prediction, 1)

# Validation set의 실제 목표값을 로드합니다.
# 이 부분은 실제 데이터에 맞게 수정해야 합니다.
valid_targets = pd.read_csv(args.dev_path)['label']

# 피어슨 상관 계수를 계산하는 함수를 정의합니다.
def calculate_pearson(actual, predicted):
    return pearsonr(actual, predicted)[0]

# 피어슨 상관 계수를 계산합니다.
pearson_score = calculate_pearson(valid_targets, final_prediction)
print(f'Ensemble Pearson Score: {pearson_score}')