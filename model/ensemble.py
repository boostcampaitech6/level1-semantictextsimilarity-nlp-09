import os
import argparse
import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
import re

# Import your model modules
import kyelectra_train as kykim
import snelectra_train as snunlp

# Another model
# import roberta_train as roberta

# Suppress warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

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
    predictions = trainer.predict(model=model, datamodule=dataloader)
    return np.array([round(float(i), 1) for i in torch.cat(predictions)])

def main():
    prj_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(prj_dir)
    model_dir = os.path.join(parent_dir, "model")
    models_checkpoints = [os.path.join(model_dir, i) for i in os.listdir(model_dir) if i[-4:] == 'ckpt']

    pred_list = []
    score_list = []
    for checkpoints in models_checkpoints:
        pattern = r"val_pearson=([0-9]+\.[0-9]+)"
        match = re.search(pattern, checkpoints)
        if match:
            score = float(match.group(1))
            score_list.append(score)

        if 'kykim' in checkpoints:
            predictions = load_and_predict(kykim, 'kykim/electra-kor-base', checkpoints, args)
        elif 'snunlp' in checkpoints:
            predictions = load_and_predict(snunlp, 'snunlp/KR-ELECTRA-discriminator', checkpoints, args)
        # else:
        #     predictions = load_and_predict(roberta, 'klue/roberta-base', checkpoints, args)

        pred_list.append(predictions)

    return pred_list, score_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--alpha', default= 0.5, type=float)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--save_path', default='../result_data/output.csv')
    parser.add_argument('--ensemble_type', default= 'weighted')
    args = parser.parse_args()

    pred_list, score_list = main()
    score_list = torch.Tensor(score_list)
    score = F.softmax(score_list)
    score = score.numpy()
    output = pd.read_csv('../data/sample_submission.csv')

    if args.ensemble_type == 'weighted':
        # 모델의 개수에 따라 변경
        # Weighted sum
        output['target'] = np.round((pred_list[0] * score[0] + pred_list[1] * score[1]), 1)
        # output['target'] = np.round((pred_list[0] * score[0] + pred_list[1] * score[1] + pred_list[2] * score[2]), 1)
    else:
        # Avg output
        output['target'] = np.round((pred_list[0]+ pred_list[1]) / 2, 1)
        # output['target'] = np.round((pred_list[0]+ pred_list[1]+ pred_list[2] ) / 3, 1)

    output.to_csv(args.save_path, index=False)
