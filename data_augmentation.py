import pandas as pd
from K_TACC.BERT_augmentation import BERT_Augmentation
import numpy as np
from tqdm.auto import tqdm
import random
import argparse

def main():
    BERT_aug = BERT_Augmentation()
    # 데이터 증강 방법: 2가지
    # random_masking_replacement: 대체
    # random_masking_insertion: 삽입
    random_masking_insertion = BERT_aug.random_masking_insertion
    # 증강 hyperparameter
    ratio = args.ratio
    
    # csv 파일 불러오기
    train = pd.read_csv('../data/train.csv')
    # label 컬럼을 구간화하여 bind_label이라는 컬럼 생성
    label_binding = pd.cut(train['label'], bins = 5, labels = [i for i in range(5)])
    train['bind_label'] = label_binding

    # bind_label별 분포 파악
    binding_label_counts = label_binding.value_counts()
    print('-' * 100)
    print('Test Dataset Binding Label Distribution')
    print(binding_label_counts.sort_index())

    # 가장 많은 bind_label의 인덱스, 값 찾기
    max_label_type = binding_label_counts.index[0]
    max_label_value = max(binding_label_counts)

    print()
    print(f"Max binding label: {max_label_type} | Max binding value: {max_label_value}")
    print(f"Augmentation for other binding labels up to the maximum binding label value({max_label_value})")
    print('-' * 100)

    # 가장 많은 bind_label을 제외한 나머지 유형의 데이터들 2배씩 증강
    pre_train_len = len(train)
    for label in tqdm(label_binding.unique(), desc = 'Data Augmentation', total = len(label_binding.unique()), leave = True):
        # 가장 많은 bind_label == 0일 때 증강 과정 생략(1,2,3,4,5 증강)
        if label != max_label_type:
            augment_times = max_label_value - binding_label_counts[label]
            if augment_times > 0:
                # 데이터 증강
                data_to_augment = train[train['bind_label'] == label]
                data_to_augment.loc[:,'sentence_1'] = data_to_augment['sentence_1'].map(random_masking_insertion)
                data_to_augment.loc[:,'sentence_2'] = data_to_augment['sentence_2'].map(random_masking_insertion)
                train = pd.concat([train, data_to_augment], ignore_index=True)

    after_train_len = len(train)
    print(f"Train Data Augmentataion Before:{pre_train_len} -> {after_train_len}")
    print(train.bind_label.value_counts())

    # bind_label == 0인 데이터 1/3 drop
    random.seed(42)  # For reproducibility
    max_label_indices = train.index[train['bind_label'] == max_label_type]
    drop_indices = random.sample(list(max_label_indices), max_label_value // 3)
    drop_train = train.drop(train.index[drop_indices])
    print(drop_train.bind_label.value_counts())

    # drop 하지 않을 경우 
    # train = train.drop(labels = 'bind_label', axis = 1)
    # train.to_csv(args.save_path, index = False)
 
    # bind_label drop, csv save
    drop_train = drop_train.drop(labels = 'bind_label', axis = 1)
    drop_train.to_csv(args.save_path, index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', default=0.15, type=float)
    parser.add_argument('--save_path', default='../result_data/aug_max3drop_train.csv', type=str)
    args = parser.parse_args()
    main() 