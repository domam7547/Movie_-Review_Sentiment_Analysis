# Movie Review Sentiment Analysis

영화 리뷰의 짧은 phrase를 입력받아 감성을 5개 클래스로 분류하는 딥러닝 프로젝트입니다.  
GloVe 임베딩을 기반으로 **BiLSTM + Attention**, **BiGRU + Attention**, **Transformer Encoder**를 구현하고 동일한 데이터 분할과 vocabulary 환경에서 성능을 비교했습니다.

> 딥러닝 기말 프로젝트 · Phrase-level Sentiment Classification

## 프로젝트 개요

이 프로젝트는 영화 리뷰 phrase를 다음 5개 감성 중 하나로 분류하는 **many-to-one multiclass classification** 문제를 다룹니다.

| Label | Sentiment |
|:---:|---|
| 0 | Negative |
| 1 | Somewhat Negative |
| 2 | Neutral |
| 3 | Somewhat Positive |
| 4 | Positive |

짧은 문장에서도 앞뒤 문맥과 감성 표현의 위치를 함께 반영할 수 있도록 양방향 순환 신경망과 Attention을 사용했습니다. 비교 모델로는 self-attention 기반의 Transformer Encoder를 직접 구성했습니다.

## 데이터

| 항목 | 설정 |
|---|---:|
| Train data | 156,060 rows |
| Test data | 66,292 rows |
| 클래스 수 | 5 |
| Train / Validation | 90% / 10% stratified split |
| Vocabulary size | 16,380 |
| Embedding | GloVe 100d |
| GloVe coverage | 약 92.5% |
| Maximum sequence length | 40 tokens |
| Batch size | 128 |

데이터는 neutral 클래스가 가장 많고 양 끝의 negative·positive 클래스가 상대적으로 적은 불균형 구조를 보였습니다. 이에 따라 train-validation 분할에는 stratification을 적용했습니다.

## 전처리

1. 모든 문자를 소문자로 변환
2. 알파벳, 숫자, 기본 문장부호 외 문자 제거
3. 문장부호를 별도 token으로 분리
4. vocabulary에 없는 단어는 `<UNK>`로 처리
5. sequence를 40 tokens로 padding 또는 truncation
6. vocabulary와 GloVe 100차원 벡터를 매핑

`not`, `never`와 같은 부정 표현이 감성 판단에 중요하므로 stopwords 제거는 적용하지 않았습니다. `<PAD>`는 zero vector로 설정했으며, Attention 계산에서 padding 위치를 masking했습니다.

## 모델

### 1. BiLSTM + Attention

양방향 LSTM이 각 token의 앞뒤 문맥을 학습하고, Attention layer가 감성 분류에 중요한 token에 더 큰 가중치를 부여합니다.

```text
GloVe Embedding → BiLSTM → Masked Attention → MLP Classifier → 5 Classes
```

### 2. BiGRU + Attention

BiLSTM과 동일한 embedding, attention, classifier 구조를 유지하고 recurrent module만 GRU로 교체했습니다. 이를 통해 LSTM과 GRU 자체의 차이를 비교했습니다.

```text
GloVe Embedding → BiGRU → Masked Attention → MLP Classifier → 5 Classes
```

### 3. Transformer Encoder

GloVe embedding에 positional encoding을 더한 뒤 Transformer Encoder와 Attention pooling을 적용했습니다. BERT와 같은 대형 사전학습 모델은 사용하지 않았습니다.

```text
GloVe Embedding + Positional Encoding
        → Transformer Encoder → Attention Pooling → Classifier
```

## 실험 설정

| 항목 | 설정 |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Epochs | 5 |
| Hidden dimension | 256 (BiLSTM / BiGRU) |
| Recurrent layers | 2 |
| Gradient clipping | `max_norm=1.0` |
| Classifier | Dropout → Linear → ReLU → Dropout → Linear |
| Transformer | `nhead=4`, `num_layers=2`, `dim_feedforward=256` |

하이퍼파라미터의 영향을 해석하기 쉽도록 기본 모델에서 한 번에 하나의 조건만 변경했습니다.

- Dropout: `0.5 → 0.6`
- Learning rate: `1e-3 → 5e-4`

## 실험 결과

| Version | Model | 변경 사항 | Best Validation Accuracy | Kaggle Public Score |
|:---:|---|---|---:|---:|
| ver1 | BiLSTM + Attention | dropout=0.5, lr=1e-3 | **0.6914** | 0.65505 |
| **ver2** | **BiLSTM + Attention** | **dropout=0.6** | 0.6906 | **0.65731** |
| ver3 | BiLSTM + Attention | lr=5e-4 | 0.6871 | 0.65576 |
| ver4 | BiGRU + Attention | dropout=0.5, lr=1e-3 | 0.6878 | 0.65635 |
| ver5 | BiGRU + Attention | dropout=0.6 | 0.6908 | 0.65623 |
| ver6 | BiGRU + Attention | lr=5e-4 | 0.6876 | 0.65323 |
| ver9 | Transformer Encoder | self-attention 기반 | 0.6747 | 0.64801 |

### Best Model

최종 모델은 **ver2: BiLSTM + Attention (dropout=0.6)** 입니다.

- Kaggle Public Score: **0.65731**
- Validation Accuracy: **0.6906**
- 전체 실험 중 가장 높은 Kaggle public score 기록

기본 BiLSTM(ver1)이 validation accuracy에서는 가장 높았지만, dropout을 0.6으로 높인 ver2가 test set에서 더 좋은 결과를 보였습니다. 강한 regularization이 일반화에 도움을 준 것으로 해석할 수 있으나, 점수 차이가 작기 때문에 특정 설정의 우위를 일반화하기보다는 이번 split과 실험 조건에서의 결과로 보는 것이 적절합니다.

## 주요 분석

- **RNN 계열이 Transformer보다 안정적이었습니다.** 대부분의 phrase가 짧고 모델을 처음부터 학습한 조건에서는 BiLSTM과 BiGRU가 더 높은 Kaggle score를 기록했습니다.
- **Dropout 조정이 learning rate 감소보다 효과적이었습니다.** learning rate를 `5e-4`로 낮춘 모델은 5 epochs 안에서 충분히 수렴하지 못했을 가능성이 있습니다.
- **Validation과 test 순위가 완전히 일치하지 않았습니다.** 최종 모델을 고를 때 두 지표를 함께 확인할 필요가 있었습니다.
- **양방향 문맥과 Attention이 phrase 분류에 적합했습니다.** 감성 단어와 부정 표현의 위치가 달라도 중요한 token을 반영할 수 있었습니다.

## 한계 및 개선 방향

- 클래스 불균형이 있으므로 accuracy 외에 macro F1-score, 클래스별 precision·recall을 함께 평가할 필요가 있습니다.
- class weight, weighted sampler, oversampling 등 불균형 보정 방법을 비교할 수 있습니다.
- 모든 모델을 5 epochs로 비교했기 때문에 낮은 learning rate와 Transformer가 충분히 수렴하지 못했을 수 있습니다.
- early stopping과 learning-rate scheduler를 적용해 학습 시점을 모델별로 조절할 수 있습니다.
- GloVe 기반 모델을 BERT, RoBERTa 등 사전학습 언어 모델과 비교할 수 있습니다.
- Attention weight 시각화와 confusion matrix를 활용하면 모델의 판단 근거와 오분류 패턴을 더 자세히 분석할 수 있습니다.

## 기술 스택

- Python
- PyTorch
- NumPy / pandas
- scikit-learn
- Matplotlib
- GloVe Embeddings

## 프로젝트 보고서

실험 설계와 모델 구조, 학습 곡선 및 결과 분석에 대한 자세한 내용은 [프로젝트 보고서](./Comparative Analysis of Attention-Based Models for Movie Review Sentiment Classification.pdf)에서 확인할 수 있습니다.

## 참고자료

- 수업 교안: *RNN, LSTM and GRU*, *Regularization and Generalization*, *Recent Issues in Deep Learning*
- Pang, B. & Lee, L. (2005). *Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales*. ACL.

