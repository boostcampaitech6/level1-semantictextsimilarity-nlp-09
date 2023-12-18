import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from electra_train import *
from roberta_train import *
import numpy as np

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name1', default='klue/roberta-base', type=str)
    parser.add_argument('--model_name2', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])
    #-------------------------------------------------------------------------------------------
    # roberta
    # dataloader와 model을 생성
    roberta_dataloader = Dataloader(args.model_name1, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    roberta_trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Inference part
    # 저장된 모델1으로 예측을 진행
    roberta_model = torch.load('../model/aug_roberta_model.pt')
    roberta_predictions = roberta_trainer.predict(model=roberta_model, datamodule=roberta_dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    roberta_predictions = np.array(list(round(float(i), 1) for i in torch.cat(roberta_predictions)))
    #-------------------------------------------------------------------------------------------
    # electra
    electra_dataloader = Dataloader(args.model_name2, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    electra_trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Inference part
    # 저장된 모델2로 예측을 진행
    electra_model = torch.load('../model/aug_electra_model.pt')
    electra_predictions = electra_trainer.predict(model=electra_model, datamodule=electra_dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    electra_predictions = np.array(list(round(float(i), 1) for i in torch.cat(electra_predictions)))
    #-------------------------------------------------------------------------------------------
    # Model Weight
    roberta_weight = 0.4
    electra_weight = 0.6
    output = pd.read_csv('../data/sample_submission.csv')
    # ensemble: Weighted sum
    # 다른 방법으로 변경 가능
    output['target'] = np.round((roberta_predictions * roberta_weight + electra_predictions * electra_weight) / 2, 1)
    output.to_csv('../result_data/ansemble_output.csv', index=False)
