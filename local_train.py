# -*- coding: utf-8 -*-
"""lstm_review_pytorch.py - Local Windows PC Version"""

# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# kobert-transformers 라이브러리에서 직접 토크나이저와 모델을 가져옵니다.
from transformers import BertModel, BertTokenizer # <-- BERT 클래스를 사용
from transformers import AutoModel # <-- AutoModel은 그대로 사용 가능

import pandas as pd
import numpy as np
import urllib.request
import os # 경로 처리를 위해 os 모듈을 불러옵니다.

# =================================================================
# 1. 로컬 환경 설정 및 경로 지정
# =================================================================

# 훈련된 모델을 저장할 로컬 폴더 경로를 변수로 지정합니다.
# 현재 스크립트 파일이 실행되는 디렉터리 내에 'my_local_models/lstm/review_classify' 폴더가 생성됩니다.
# Windows에서는 상대 경로('./')를 사용하면 편리합니다.
LOCAL_PATH = './my_local_models/lstm/review_classify'

# 폴더가 없다면 생성합니다.
os.makedirs(LOCAL_PATH, exist_ok=True)
print(f"모델 저장 경로: {os.path.abspath(LOCAL_PATH)}")

# GPU 사용 설정 (KoBERT 모델 로드 및 device 설정은 이전 코드 그대로 유지)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================

# 데이터 로드 (네이버 쇼핑 리뷰 데이터 자동 다운로드)
DATA_FILE = 'shopping.txt'
print(f"데이터 {DATA_FILE} 다운로드 중...")
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt',
    DATA_FILE
)
raw = pd.read_table(DATA_FILE, names=['rating','review'])
raw['label'] = np.where(raw['rating']>3, 1, 0)
print(f"총 {len(raw)}개의 리뷰 데이터 로드 완료.")

# 토크나이저 및 모델 로드
MODEL_NAME = 'monologg/kobert' # <-- 모델 이름을 더 안정적인 monologg 버전으로 변경
# 1. BertTokenizer를 사용하여 로드
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME) # <-- BertTokenizer 사용

# 2. BertModel을 사용하여 로드
kobert_model = BertModel.from_pretrained(MODEL_NAME).to(device).eval() # <-- AutoModel 대신 BertModel 사용

EMBEDDING_DIM = kobert_model.config.hidden_size # 768

# 전체 데이터셋 인코딩
reviews = raw['review'].tolist()

print("전체 리뷰 데이터 인코딩 중...")
tokenized_data = tokenizer(
    reviews,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
labels = torch.tensor(raw['label'].values)
tokenized_data['labels'] = labels
print("인코딩 완료.")


# --- 3. 학습/검증 데이터 분리 및 DataLoader 준비 ---

from sklearn.model_selection import train_test_split
# 데이터 분할 (예: 80% 학습, 20% 검증)
input_ids = tokenized_data['input_ids']
attention_mask = tokenized_data['attention_mask']

train_indices, val_indices = train_test_split(
    range(len(labels)),
    test_size=0.2, # 검증 셋 비율 20%
    stratify=labels.cpu().numpy(),
    random_state=42
)

# 학습 데이터셋 준비
train_dataset = TensorDataset(
    input_ids[train_indices],
    attention_mask[train_indices],
    labels[train_indices]
)

# 검증 데이터셋 준비
val_dataset = TensorDataset(
    input_ids[val_indices],
    attention_mask[val_indices],
    labels[val_indices]
)

BATCH_SIZE = 32 # 배치 크기 설정

# DataLoader 생성
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =================================================================
# 4. GRU 기반 분류기 모델 정의
# =================================================================

class GRUClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GRUClassifier, self).__init__()
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, attention_mask):
        global kobert_model # 전역 변수 KoBERT 모델 사용

        # 1. KoBERT 임베딩 추출 (no_grad로 KoBERT 가중치 고정)
        with torch.no_grad():
            outputs = kobert_model(input_ids=text, attention_mask=attention_mask)
            embedded = outputs.last_hidden_state

        # 2. GRU 레이어 통과
        rnn_output, hidden = self.rnn(embedded)

        # 3. 최종 은닉 상태를 분류기에 사용 (순방향/역방향 결합)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        # 4. 선형 레이어를 통과시켜 최종 예측값 계산
        prediction = self.fc(hidden)

        return prediction


# =================================================================
# 5. 모델 학습 및 평가 함수
# =================================================================

from sklearn.metrics import accuracy_score

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(labels)

    return epoch_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval() # 평가 모드 (Dropout, BatchNorm 비활성화)
    epoch_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad(): # 평가 시에는 그래디언트 계산 불필요
        for batch in dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item() * len(labels)

            all_predictions.extend(predictions.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 정확도 계산
    accuracy = accuracy_score(all_labels, all_predictions)

    return epoch_loss / len(dataloader.dataset), accuracy


# =================================================================
# 6. 학습 실행
# =================================================================

HIDDEN_DIM = 256
OUTPUT_DIM = 2
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 1e-3
N_EPOCHS = 20

model = GRUClassifier(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


print("\n" + "=" * 60)
print(f"GRU 분류 모델 학습 시작 (학습: {len(train_dataset)}개, 검증: {len(val_dataset)}개)")
print("=" * 60)

best_val_loss = float('inf')

for epoch in range(N_EPOCHS):
    # 학습
    train_loss = train(model, train_dataloader, optimizer, criterion, device)

    # 검증
    val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)

    # 베스트 모델 저장 (손실이 가장 낮을 때)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 저장 경로 수정 (로컬 경로 사용)
        SAVE_PATH = os.path.join(LOCAL_PATH, 'gru_classifier_best.pt')
        torch.save(model.state_dict(), SAVE_PATH)

    print(f'에폭: {epoch+1:02} | 학습 손실: {train_loss:.4f} | 검증 손실: {val_loss:.4f} | 검증 정확도: {val_acc:.4f}')

print("=" * 60)
print("학습 완료.")
print(f"가장 성능이 좋았던 모델이 '{SAVE_PATH}'에 저장되었습니다. (최소 검증 손실: {best_val_loss:.4f})")