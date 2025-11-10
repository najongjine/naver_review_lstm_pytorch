# -*- coding: utf-8 -*-
"""gru_review_pytorch_tiktoken_clean.py - tiktoken 및 PyTorch nn.Embedding 기반 (최소 전처리)"""

# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import urllib.request
import os
import re

# ⭐️ tiktoken 임포트 및 인코더 로드 ⭐️
import tiktoken 
ENCODER_NAME = "cl100k_base" # GPT-4, GPT-3.5 Turbo에 사용되는 인코딩
# 인코더 로드
tokenizer = tiktoken.get_encoding(ENCODER_NAME)
# 'tok'라는 이름으로 토큰화 메서드를 alias하여 Mecab 자리에 대체 사용
def tok(text):
    return tokenizer.encode(text) 

# =================================================================
# 1. 로컬 환경 설정 및 경로 지정 (생략 없음)
# =================================================================

LOCAL_PATH = './my_local_models/gru_tiktoken_clean/review_classify' # 경로 이름 변경
os.makedirs(LOCAL_PATH, exist_ok=True)
print(f"모델 저장 경로: {os.path.abspath(LOCAL_PATH)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================

# 데이터 로드 (Naver Shopping Review)
DATA_FILE = 'shopping.txt'
print(f"데이터 {DATA_FILE} 로드 중...")

if not os.path.exists(DATA_FILE):
    print(f"데이터 {DATA_FILE} 다운로드 중...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt',
        DATA_FILE
    )
    
# ⭐️ 정규표현식(reg_txt) 정의 및 사용 코드 제거 ⭐️
#reg_txt = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z\s]") # 이 부분 제거
raw = pd.read_table(DATA_FILE, names=['rating','review'])
raw['label'] = np.where(raw['rating']>3, 1, 0)
print(f"총 {len(raw)}개의 리뷰 데이터 로드 완료.")

# 리뷰 데이터 전처리 및 정제 (최소한의 결측값 처리만 남김)
raw['review'] = raw['review'].fillna('')
# raw['review'] = raw['review'].str.replace(reg_txt, '', regex=True) # 이 부분 제거
raw = raw[raw['review'].str.strip() != '']
print(f"빈 리뷰를 제거하여 총 {len(raw)}개의 리뷰 데이터가 남았습니다. (정제 로직 최소화)")

# ------------------------------------------------------------------
# ⭐️ tiktoken 토큰화 및 단어 집합 관리 ⭐️
# ------------------------------------------------------------------

MAX_LEN = 512
PAD_TOKEN_ID = tokenizer.eot_token 

# 토큰화 및 패딩 함수
def tokenize_and_pad_tiktoken(reviews, max_len, pad_id):
    encoded_sentences = []
    
    for review in reviews: 
        # tiktoken.encode()는 띄어쓰기나 불용어 처리를 건너뛰고 BPE로 인코딩합니다.
        encoded = tokenizer.encode(review)
        
        # 패딩 또는 잘라내기
        if len(encoded) < max_len:
            encoded += [pad_id] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        
        encoded_sentences.append(encoded)
        
    return torch.tensor(encoded_sentences, dtype=torch.long)

# 1. 리뷰 텍스트 리스트 준비
review_texts = raw['review'].tolist()

print(f"tiktoken({ENCODER_NAME}) 토큰화 및 정수 인코딩 중...")

# ⭐️ 불용어 제거 및 단어 집합 구축 로직은 원래 tiktoken 코드에도 없었음 ⭐️

# 2. 리뷰를 정수 인코딩
input_ids = tokenize_and_pad_tiktoken(review_texts, MAX_LEN, PAD_TOKEN_ID)
labels = torch.tensor(raw['label'].values, dtype=torch.long)

# 3. 최종 VOCAB_SIZE 재확인 및 PAD_IDX 정의
FINAL_VOCAB_SIZE = tokenizer.n_vocab 
PAD_IDX = PAD_TOKEN_ID

print(f"tiktoken 기반 단어 집합 크기 (n_vocab): {FINAL_VOCAB_SIZE}")
print(f"정수 인코딩 완료. 데이터 형태: {input_ids.shape}")

# ------------------------------------------------------------------
# 3. 학습/검증 데이터 분리 및 DataLoader 준비 (생략 없음)
# ------------------------------------------------------------------

train_indices, val_indices = train_test_split(
    range(len(labels)),
    test_size=0.2, 
    stratify=labels.cpu().numpy(),
    random_state=42
)

train_dataset = TensorDataset(input_ids[train_indices], labels[train_indices])
val_dataset = TensorDataset(input_ids[val_indices], labels[val_indices])

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =================================================================
# 4. GRU 기반 분류기 모델 정의 (nn.Embedding 포함) (생략 없음)
# =================================================================

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, pad_idx):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx 
        )

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

    def forward(self, text):
        embedded = self.dropout(self.embedding(text)) 
        rnn_output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) 
        prediction = self.fc(hidden) 

        return prediction


# =================================================================
# 5. 모델 학습 및 평가 함수 (생략 없음)
# =================================================================

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        input_ids, labels = [t.to(device) for t in batch]

        optimizer.zero_grad()
        predictions = model(input_ids) 
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(labels)

    return epoch_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval() 
    epoch_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = [t.to(device) for t in batch]

            predictions = model(input_ids) 
            loss = criterion(predictions, labels)

            epoch_loss += loss.item() * len(labels)

            all_predictions.extend(predictions.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)

    return epoch_loss / len(dataloader.dataset), accuracy


# =================================================================
# 6. 학습 실행 (생략 없음)
# =================================================================

EMBEDDING_DIM = 300 
HIDDEN_DIM = 512 
OUTPUT_DIM = 2
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE_GRU = 1e-3 
N_EPOCHS = 20

# tiktoken 기반 모델 초기화
model = GRUClassifier(
    vocab_size=FINAL_VOCAB_SIZE, 
    embedding_dim=EMBEDDING_DIM, 
    hidden_dim=HIDDEN_DIM, 
    output_dim=OUTPUT_DIM, 
    num_layers=NUM_LAYERS, 
    dropout=DROPOUT,
    pad_idx=PAD_IDX 
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_GRU)

print("\n" + "=" * 60)
print(f"tiktoken + GRU 분류 모델 학습 시작 (학습: {len(train_dataset)}개, 검증: {len(val_dataset)}개)")
print("=" * 60)

best_val_loss = float('inf')
SAVE_PATH = os.path.join(LOCAL_PATH, 'gru_tiktoken_classifier_best.pt')

for epoch in range(N_EPOCHS):
    # 학습
    train_loss = train(model, train_dataloader, optimizer, criterion, device)

    # 검증
    val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)

    # 베스트 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)

    print(f'에폭: {epoch+1:02} | 학습 손실: {train_loss:.4f} | 검증 손실: {val_loss:.4f} | 검증 정확도: {val_acc:.4f}')

print("=" * 60)
print("학습 완료.")
print(f"가장 성능이 좋았던 모델이 '{SAVE_PATH}'에 저장되었습니다. (최소 검증 손실: {best_val_loss:.4f})")