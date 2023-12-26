### LP_09팀 공용 github
# Semantic Text Similarity (STS)

## NLP-09 (Time Flies)

### 팀원 : 김인수, 오주영, 양서현, 문지원, 손윤환

  

# 1. Intro

## 1.1. 개요

- Semantic Text Similarity: 복수의 문장에 대한 유사도를 선형적 수치로 제시하는 NLP Task.
- 이는 두 문장이 서로 동등하다는 양방향성 가정하고 진행됨.
- 이러한 수치화 가능한 양방향 동등성은 정보 추출, 질문-답변 및 요약과 같은 NLP 작업 전반에 널리 활용 및 응용.
- 대회 목표는 STS 데이터셋을 활용해 두 문장의 유사도를 측정하는 AI 모델의 구축.
- [0, 5] 범위의 유사도 점수를 출력

![](https://lh7-us.googleusercontent.com/_zl7-k8E9oCZ6_saZcDhbm49JXmJTzv3_IAsodh7BwKh04pV336lymw0YC6hnYz4nSTB7dIOyANJg94QtMTtKuNK48N2s9HZ-od6EaZT3fct-9JNvW_KvM5rDL6kz1bosbORGSZMMVxGUUboaLRQ9QQ)
- 학습 데이터셋 9,234개, 검증 데이터셋 550개, 평가 데이터셋 1,100개.

- 평가 데이터의 50%는 Public 점수 계산에 활용, 실시간 리더보드에 반영되며 남은 50%는 Private 결과 계산에 활용되어 최종 평가에 반영.

- 최종 결과물은 .csv 형태로 제출.

- 입력: 두 개의 문장과 ID, 유사도 정보

- 출력: 평가 데이터에 있는 각 문장쌍에 대한 ID와 유사도 점수

![](https://lh7-us.googleusercontent.com/_2tdRz7ayix0SSEnckaAcvJ1HbMjncY4sF2-0vpY4NGboTJAcS-2OAWw6DxG2stkU4Q9KXNCftIuZHj0YGFShIT2HNlNik_8rn_5EvMVEZWiWmvCCgdRIKg7OUtAJStnNPK6_2Z_VQVDL7rXt4bUuHI)

- 평가 기준은 예측과 정답 간의 Pearson Correlation Coefficient으로 삼는다.

- 개별 예측의 일치보다는 전체적인 경향의 유사도가 중요.

![](https://lh7-us.googleusercontent.com/wSimaDFPMiJrUglcBIKY8BYqGmtbpYHk9804MJGBKJcLW6VJhdMZk42LOYr2L_rhC7USiWoGZGWZ4DJiaF8LRO8qb22GlPqF--b0xoDjUm0xBslaxa_khVzfMQApLXn4Yh0LLDYzdgNLT49RVITVX3Y)


## 1.2. 프로젝트 구조

![](https://lh7-us.googleusercontent.com/FyBGLGr7U2cEFkucXuvOyuUgyLZI5t7y8ienwowt_NEPkjtUF2FnUomFsJLLAi0N-l3dRdAvf9lXMHG7tChuLI9pO7xrd5QUnA0ECP_haXSUTr5wsOC1jg7ncTTwxJkY_lI9uR-wnCXocI24T1pM84w)

![](https://lh7-us.googleusercontent.com/yD24sgVgJsvlbOAtpuNqQooqy4CjF07iuHdYj28E7c6yYVeKbhZXy5pF3KsFDKGuw07dPIdA8BB6Ih-df0SbDUeyI8x74Ba_zWkcURbNu-CUsUfkvOBzP6BOFoEmn-aueof_-5Zxp104_a3sqMh1dPk)

![](https://lh7-us.googleusercontent.com/VFIis-zBP-XdHMgHvbk9PbYlI_RyAUXuJMiIoIj_m47WkiDEGxWaJCdrwH7KHRhCrZL9Zx_gBEVOWq1q4NNECzLstlA2rWQ9jGnL39Y1Hu0FpnPxcLSgglwHjyIWylJC2rApG4eFAJ93yNNupOy4hwk)

  

## 1.3. 프로젝트 환경

- GPU: V100 * 5
- 협업 관리: [Notion](https://www.notion.so/Time-Flies-ae9378d5426d4e659ee3b5aacaab0d64), [Github](https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-09)


## 2. 프로젝트 팀 구성 및 역할

|   |   |
|---|---|
|팀원|역할|
|공통|데이터 분석, 실험 수립 및 진행, 하이퍼 파라미터 튜닝, Wrap Up 리포트 작성|
|김인수|모델 학습 및 앙상블 베이스라인 코드 작성, 데이터 증강(bert)|
|오주영|손실함수 탐색, 다양한 손실함수 비교/분석 및 최적화|
|양서현|train-dev 데이터 분석, 데이터 증강(Label Smoothing, Copied sentence, 문장 교정)|
|문지원|한국어 기반 모델 탐색, 데이터 전처리 실험(특수문자 제거, 형태소 분석)|
|손윤환|한국어 기반 모델 탐색, 예측 데이터 이상치 분석, 데이터 증강 실험(역번역)|

  

## 3.  프로젝트 수행 절차 및 방법

### 3.1. 진행 절차

1. 강의 수강 및 사전 학습
    
2. 데이터 EDA, 다양한 접근 방법 조사 및 실험
    
3. 진행한 실험 공유 및 유의미한 접근법 선정
    
4. 선정한 접근법 기반 역할 분배 및 실험 진행
    
5. 모델 튜닝 및 앙상블 통해 최종 결과물 도출
    

  

<img src="https://lh7-us.googleusercontent.com/U9dzQpg-eysN7TXdy06KM-oBC8BapVo0pDSt_2Hk9zKW_rY1tOf-kGCJMz03qOwFznq6YTd2VkTpmMAZS2SW8pYFuJG9k0WyE-44WIPbp7Qx26prxahr6zm3BSW_w9OmoDKarNYXQFXidy6Q2TesYrc" width="50%" height="50%"/>

  
### 3.2. 협업 문화

1. 팀 일일계획표를 작성하며 각자 목표 및 진행 상황 공유함
    
2. 자료 및 정보 공유를 위해 공용 노션 페이지를 활용함
    
3. 질문 또는 어려운 부분이 있을 시 실시간 회의 때 함께 보며 해결함
    

  

## 4. 프로젝트 수행 결과

### 4-1. EDA (Exploratory Data Analysis)
 주어진 데이터의 label 분포, source 분포, 문장 길이, 문장 형태 분석과 관련된 탐색적 데이터    
분석을 진행하였고, 이를 통해 성능 개선 가설, 전략을 수립하였다. 

A. Basic Data Information

 Train data의 개수는 총 9,324개, dev data의 개수는 총 550개 그리고 test data의 개수는 총  
1100개이며, 3개의 dataset 모두 null값은 존재하지 않았다. target column label은 0~5 사이  
float64 type으로 해당 task는 Regression 문제에 해당한다고 보았다. 

B. Label 분포

  train와 dev data의 분포를 barplot 시각화를 통해 확인했다. train data은 0에 편향돼 있고,  
dev data는 골고루 분포되어 있다. raw train data로 학습 시에 label 값 편향이 있어 모델의  
성능이 낮아질 수 있는 가능성을 알 수 있었다.

<img src="https://lh7-us.googleusercontent.com/PZ6qBMVJo49nx9cU9MuQqyd_4ZA_M9PikCsdeEzdx__4mw6Z4M9gneC4maBKO0R9q9rjXuBONw32Xf6U6apK594agXNxx5GYIQSRrG-E9L4FQTHGGe-7WnLAPmX10i1UlmQ3gsLMpkcuFlDS6iZavfE" width="50%" height="50%"/>
<img src="https://lh7-us.googleusercontent.com/hj7hWmr9hMAAqDCaO_HxsF0PPUGMLVg1flSKrsir-LCyuP7n34ZA6PCYf-UExmCb6T_fF2xHaDDCicEoKZv8SFu5BKSNwa9QZzTY3aeuaXpUoXqR6IHCMCGMnoEpHOVWaCHz-7I66SIoHzrYvMOJaCA" width="50%" height="50%"/>

C. Source 분포

  train과 dev data의 분포는 비슷하다. one-hot encoding을 이용해 Object형을 숫자형으로  
변환 이용 가능성을 제시할 수 있었다. 

![](https://lh7-us.googleusercontent.com/U3Cd6NS_c6PGqbeWoxHrcLvwi38TWdLIrAnZZ2MydBIoVigtsazdnq_tpTOOEcCT-uXBS7IsOuG8F8JeqIUUI9yt91R9cpalBp-7n1r22y_uOCss3RiwcGmRi-5GCTVZCJQ-vU4aQzqKw_UbaFMhWDk)

  

D. 문장 길이 분석

 train data의 두 문장 길이는 비슷하나, 이상치의 존재에 대해 결과의 편향이 발생할 수 있음을 인지할 수 있었다. dev data와의 분포도 비슷하였다.
  
<img src="https://lh7-us.googleusercontent.com/nWYJRKrA0XptwX1MnTylBLhlSzwk-LrG9lrtlApJnpfshI6NFRALucz5xOqgtamldaGF7sptOMwoa4W-gJ-L8qnaQdElMgZrxXt6UOpHShm4vzfnq2q7r-HsuwL2rwmhR3DniDGgCJYOc3mA-3nPOAA" width="50%" height="50%"/>
<img src="https://lh7-us.googleusercontent.com/niNHxuQF8WixxFUGmyEikEXsyY-tWtiUgK3U1evSjkjh30hgFxCUG07PJFykwzEjzTnbgTRa6ixTTVwuCMDVrAPa10Vj7QxVzV_p4fu8QGfoLJNKLtczMkFs874jy7lHTIfG6TkaM3-S1NclVm_aNRI" width="70%" height="70%"/>

  

E. 문장 형태 분석

- 특수문자 - <Person>, [UNK], !!!, ^^ 등 마스킹, 특수문자의 중복
- 반복되는 한글 자모음 - ㅠㅠㅠ/ㅎㅎ/ㅋㅋ 등 자음, 모음이 3개 이상 반복되는 경우
- 맞춤법, 띄어쓰기 - 맞춤법과 띄어쓰기가 맞지 않은 문장 다수
    

  
### 4-2. 실험 방법 및 과정

### A. 모델 측면

### 1. kykim/electra-kor-base

![](https://lh7-us.googleusercontent.com/ZEorWhH4VHyrarwk9F7-ztZnGsl7MeeTcf89TANK8cxJIGQaXNIBjfqfYl6ZVv4Z0ImOlbON-TYa4BC34Z41eCConjRHogy5yWWJju__tKcXZ9A5_b3ZsW-4JLPdw-87SMzdFVKQ1k4bHai23kNvhx8)

 기존 klue/roberta-base 모델 학습 당시 valid 예측 결과를 분석한 결과 [UNK] 토큰으로 인해

예측이 제대로 이루어지지 않는 상황 발생 -> tokenizer vocab 크기가 큰 모델을 탐색

기존 32K 사이즈보다 큰 42K word piece 알고리즘 기반 학습모델 kykim/bert 모델 중

KorSTS 벤치마크 성능이 가장 뛰어난 electra 모델 사용

  

### 2. snunlp/KR-ELECTRA-discriminator

KorSTS 벤치마크 기준으로 상위권 모델

Mecab-Ko 형태소 분석기 기반 사전 -> 한국어 기반으로 구축된 양질의 사전을 통해 성능

향상 기대

  

### 3. monologg/koelectra-base-v3-discriminator

많이 사용되는 여러 한국어 기반 모델과 현재 프로젝트에서 선택한 모델에 비교해서 KorSTS 벤치마크를 기준으로 더 뛰어난 성능의 모델 추가


### Ensemble

(kykim(증강), snu, snu(증강), monologg, snu(aug+hanspell))

모델의 일반화 및 성능 향상을 위해 앙상블 Weighted sum 모델 적용

Test pearson 기준 0.92가 넘는 모델 후보군 설정. Inference의 다양성 확보를 위해 증강

데이터로 학습한 모델과 아닌 모델들을 앙상블(예측 결과 모든 라벨을 고르게 예측)

  

### B. 손실함수 측면

- 손실함수의 설정이 모델의 학습 방향에 큰 영향을 미치기 때문에 적절한 손실함수 사용이 성능에 영향을 줄 것이라는 가정에서 다양한 손실함수를 적용해보기로 결정.
  
- 실험에 적용한 손실함수 : SmoothL1Loss, MSELoss, pearson score, General and Adaptive Robust Loss Function, MSE + Huber, MSE + noise(CosineSimilarity), MSE+pearson

- 각 손실함수의 설정 의의와 결과는 노션의 STS Project - 실험결과 참조. 
    

  

### C. 데이터 측면 

 Inference output 결과 분석
 
 <img src="https://lh7-us.googleusercontent.com/wQ_D8dXXd9UPw7aapowEb1FugD-Eh8PzSzIerVvF9gVV1UclwK-u4MOwZR8nRFMbZuJVeKSgf0pz9cBouMH84SKo7mnYhdZ70pVEi5B0X08yyQKSdEizrLHLU6PofQgzlbK_7jnq7AqeTINWnkDMT68" width="50%" height="50%"/>

- valid 예측-정답 간의 점수 차이 분포를 비교, 분석 결과 고득점 label에 대해 낮은 예측값을 출력하는  
  경우가 다수 발생함을 확인
- [UNK] 토큰과 문장 의미상 오타가 발생하는 경우  
  예측이 제대로 이루어지지 않음
→ Py-Haspell 라이브러리를 이용한 오타 교정 및
  tokenizer 개선 모델 탐색 진행

  
 <img src="https://lh7-us.googleusercontent.com/ABf8a964DwK-1P25xaoOTC-NuXtID-k3qzWn2zX3TZP_Gc6vcYVn_v3xQuB5NpJnx6JDhNFO-81Kp5EsstQe7f8nFt-jaiBVUutRmB6maq5UQA7ic_Z3kNRTZ__9i5LbOIU5O_S7QkVqWVAbe94jK2o" width="50%" height="50%"/>

 - 각 모델 별 valid 예측 분포를 분석하여 향후  
  앙상블 과정에서 weighted sum 계산 기준 수립

  

### Preprocessing Data 및 결과

1. 데이터 정제(data cleaning)  
    - 맞춤법 검사(Hanspell)  
      맞춤법 검사 라이브러리 Py-Haspell 내 spell_checker를 활용했다. 맞춤법, 띄어쓰기 오류가 많이 개선하였으며, [UNK] 등 마스킹을 삭제하여 실험시 성능이 향상되었다.  
    - regexp  
      정규식을 활용해 특수문자를 제거했다. regexp만 사용하여 실험 시 raw와 성능이 비슷하지만 다른 전처리 기법과 같이 사용했을 때 성능이 떨어져 채택하지 않았다.
    
2. 데이터 증강(data augmentation)  
    - Bert Augmentation([https://github.com/kyle-bong/K-TACC](https://github.com/kyle-bong/K-TACC))
    

 BERT 모델을 통해 문맥을 고려하여 [MASK] 토큰을 복원하는 방식으로 데이터를 증강 하는 방법. Insertion 방법과 Replacement 2가지 방법 중 원본 문장의 단어는 그대로 보존한 채 단어나 기호를 추가하는 방식인 Replacement 방법 채택.

실험 결과 중 가장 높은 Pearson`s correlation(0.9300)

  
- label smoothing, Copied sentence  
bert 증강기법으로 증강하였으나 여전히 label 0 데이터의 비율이 높았다.  
  
<img src="https://lh7-us.googleusercontent.com/kFUowXYLDEp_flPI3glA6kk95f_3nUy9y8BsiVp0UJZOqqNZemE8s-uNRYi1tql4uHym9v9PeSpm3ZMpA0FeB1AqnMjuubs6RwdueT-K0--RPcIhSCzXLQikIC6IlvLmbzVfkHUTWocHHdA5Nn64zPk" width="50%" height="50%"/>

따라서 label을 uniform, random 분포로 만들어주기 위해, label 0의 데이터를 잘라내서 label 5의 데이터로 만들어주는 작업을 진행했다. 

  이 때, copied sentence와 hanspell을 이용하여 label 0의 데이터 중 문장2 데이터를 hanspell 적용하여 label 0 개수와 비슷해질 때까지 문장2로 copy하여 label 5로 만들어주었다. 잘라내는 label 0 데이터의 비율을 조정하며 최적의 값을 찾아보았다.

  후에 dev 분포와 비슷하게 같은 방법으로 증강이 필요한 label을 2배 증강하였다.
  
  ![](https://lh7-us.googleusercontent.com/tMqj0UcjmTCgsDmhtHqX_z9Stv2PQK0_tzRaPh2jNPHdUSlMYohxHXaIOYPHjjELcqbwxCcXXjaVsyQQBqj65vM11ylgY-DkUjgufZMTwNtFVCrYIr-TmpvNhX9BpGrK00MGszDtZTxsJxoE8cfAaeQ)

- Kolmogorov-Smirnov test : 귀무가설을 기각하지 못하므로, 분포가 비슷하다.

|   |   |
|---|---|
|KS Statistic|0.046486|
|P-value|0.192672|

  

- Swap sentence  
  두 문장의 순서를 바꾸는 방법이다. 문장 순서가 바뀌어도 유사도(label)는 동일함이 객관적으로 보장되며, 또 Bert 모델계열 사용 시에 Segmentation Embedding 값이 다르므로 유의미한 데이터 증강이 될 것이라 분석하였다. 그러나 성능이 오히려 떨어지는 결과가 나와 채택하지 않았다.

  
- K-fold  
  데이터 개수가 적을 때 일반화 성능의 향상을 위해 유용하게 쓸 수 있는 방법이다. 회귀문제는 stratified를 K-fold 쓸 수 없어 balanced하게 데이터 전처리 후 K-fold를 적용해보았으나, 데이터가 많이 증강된 후라 성능 향상에 크게 의미가 없었다.  

  
  

### 4-3. 결과

 Public Score 0.9285,Private Score 0.9349

 학습 데이터와 검증 데이터의 분포의 차이를 줄이는 것, 모델의 Inference 결과를 분석하여 편향된 Inference 경향을 줄이는 것에 집중하여 대회 진행.모델 측면에서 앙상블을 통해 일반화 성능을 높이고 편향된 Inference 경향을 줄이는 데 성공했다.하지만 더 낮은 점수였던 모델의 Private Score가 더 높았다. 데이터 측면에서는 다양한 데이터 전처리, 증강 방법에 따른 모델 결과에 대한 분석이 부족했다.![](https://lh7-us.googleusercontent.com/DkJg5A2yLYmicTIXNfuoNjgpVSmq5o-UxNBgGgafaextwkUKKQ-Ie5bjk70TylnISpD5HUjo1GV9JzjlXyZOIjoAGRn3ZeGhx4wpNWBKhO-YVj0uqX7YcXNja1yfgVK4rJAPJoLPa3-F738VBhBzzec)

![](https://lh7-us.googleusercontent.com/VmXqjdu10k6ahP2OVH15BkhWjaE37zaxtL-KLmpgDTjw6lANDux25JGj7IqH76mX6UPpeWLRp4DFrS76Qhd8uEmopwImRYSTq2Yi7CRVyZB3-24mJ8yC_JJUBBgYDt9SDK-_-gco1uKQvMds5e_UsKc)

  
