# -*- coding: utf-8 -*-
"""gru_review_pytorch_mecab.py - Mecab 및 PyTorch nn.Embedding 기반"""

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

# Mecab 사용을 위한 import (설치되어 있어야 합니다)
from mecab import MeCab 
# from soynlp.word import WordExtractor # Mecab이 없다면 다른 토크나이저 고려 가능

# =================================================================
# 1. 로컬 환경 설정 및 경로 지정
# =================================================================

LOCAL_PATH = './my_local_models/gru_mecab/review_classify'
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

# 데이터 파일이 없으면 다운로드 (주석 해제 필요 시)
if not os.path.exists(DATA_FILE):
    print(f"데이터 {DATA_FILE} 다운로드 중...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt',
        DATA_FILE
    )
    
reg_txt = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z\s]") # 한글, 영문, 공백만 남깁니다.
raw = pd.read_table(DATA_FILE, names=['rating','review'])
raw = raw[:6000] # 데이터 6000개만 사용
raw['label'] = np.where(raw['rating']>3, 1, 0) # 4점, 5점은 1(긍정), 나머지는 0(부정)
print(f"총 {len(raw)}개의 리뷰 데이터 로드 완료.")

# 리뷰 데이터 전처리 및 정제
raw['review'] = raw['review'].fillna('')
raw['review'] = raw['review'].str.replace(reg_txt, '', regex=True)
raw = raw[raw['review'].str.strip() != '']
print(f"정규표현식 적용 후, 빈 리뷰를 제거하여 총 {len(raw)}개의 리뷰 데이터가 남았습니다.")

# ------------------------------------------------------------------
# ⭐️ Mecab 토큰화 및 불용어 제거, 단어 집합 구축 ⭐️
# ------------------------------------------------------------------

MAX_LEN = 256
VOCAB_SIZE = 10000 
UNKNOWN_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

# Mecab 토크나이저 초기화
mecab = MeCab()

# 토큰화 함수
def tokenize_and_pad(reviews, vocab, max_len, pad_token, unk_token):
    encoded_sentences = []
    
    for sentence_tokens in reviews: # reviews는 이미 토큰화된 리스트입니다 (all_tokenized_reviews)
        # 단어를 인덱스로 변환
        encoded = [vocab.get(word, vocab[unk_token]) for word in sentence_tokens]
        
        # 패딩 또는 잘라내기
        if len(encoded) < max_len:
            encoded += [vocab[pad_token]] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        
        encoded_sentences.append(encoded)
        
    return torch.tensor(encoded_sentences, dtype=torch.long)

# 사용자 정의 불용어 리스트
stop_word=["그","에서","나","에","을","를","다","의","는",
           "들","은","는","이","가","어요","과","도","하다"]

# 1. 단어 빈도수 계산을 위한 데이터 정제 (불용어 제거)
all_tokenized_reviews = []
review_texts = raw['review'].tolist()

print("Mecab 형태소 분석 및 불용어 제거 중...")
for review in review_texts:
    # 1) 형태소 분석
    tok = mecab.morphs(review) 
    
    # 2) 불용어 제거
    tmp = [w for w in tok if not w in stop_word]
    all_tokenized_reviews.append(tmp)

# 2. 단어 빈도수 계산
word_counts = {}
for token_list in all_tokenized_reviews:
    for word in token_list:
        word_counts[word] = word_counts.get(word, 0) + 1

# 3. 빈도수를 기준으로 상위 N개 단어 선택
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
top_words = [word for word, count in sorted_words[:VOCAB_SIZE - 2]] 

# 4. 단어 집합(Vocabulary) 생성: 인덱스 0은 <pad>, 1은 <unk>
word_to_index = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
for index, word in enumerate(top_words, start=2):
    word_to_index[word] = index
    
# 5. 리뷰를 정수 인코딩
input_ids = tokenize_and_pad(all_tokenized_reviews, word_to_index, MAX_LEN, PAD_TOKEN, UNKNOWN_TOKEN)
labels = torch.tensor(raw['label'].values, dtype=torch.long)

# 6. 최종 VOCAB_SIZE 재확인
FINAL_VOCAB_SIZE = len(word_to_index)
print(f"Mecab 기반 단어 집합 크기: {FINAL_VOCAB_SIZE}")
print(f"정수 인코딩 완료. 데이터 형태: {input_ids.shape}")

# ------------------------------------------------------------------
# 3. 학습/검증 데이터 분리 및 DataLoader 준비
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
# 4. GRU 기반 분류기 모델 정의 (nn.Embedding 포함)
# =================================================================

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, pad_idx):
        super(GRUClassifier, self).__init__()
        # 1. 임베딩 레이어: Mecab 기반 단어 집합 사용
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx # 패딩 토큰 인덱스를 지정하여 학습에서 제외
        )

        # 2. GRU 레이어
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        # 양방향(Bidirectional)이므로 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # 1. 임베딩 추출
        embedded = self.dropout(self.embedding(text)) # [batch size, seq len, emb dim]

        # 2. GRU 레이어 통과
        # rnn_output: [batch size, seq len, hidden dim * 2]
        # hidden: [num layers * 2, batch size, hidden dim]
        rnn_output, hidden = self.rnn(embedded)

        # 3. 최종 은닉 상태를 분류기에 사용 (순방향/역방향 결합)
        # hidden[-2,:,:]는 역방향 마지막 레이어의 최종 은닉 상태
        # hidden[-1,:,:]는 순방향 마지막 레이어의 최종 은닉 상태
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) # [batch size, hidden dim * 2]

        # 4. 선형 레이어를 통과시켜 최종 예측값 계산
        prediction = self.fc(hidden) # [batch size, output dim]

        return prediction


# =================================================================
# 5. 모델 학습 및 평가 함수 (Attention Mask 제거)
# =================================================================

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        # Mecab 기반: input_ids, labels만 받습니다.
        input_ids, labels = [t.to(device) for t in batch]

        optimizer.zero_grad()
        predictions = model(input_ids) # attention_mask 제거
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
            # Mecab 기반: input_ids, labels만 받습니다.
            input_ids, labels = [t.to(device) for t in batch]

            predictions = model(input_ids) # attention_mask 제거
            loss = criterion(predictions, labels)

            epoch_loss += loss.item() * len(labels)

            all_predictions.extend(predictions.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)

    return epoch_loss / len(dataloader.dataset), accuracy


# =================================================================
# 6. 학습 실행
# =================================================================

EMBEDDING_DIM = 300 # 임베딩 차원 설정 (KoBERT 768 대신 일반적인 크기 사용)
HIDDEN_DIM = 512 
OUTPUT_DIM = 2
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE_GRU = 1e-3 # GRU와 임베딩 레이어를 함께 학습하므로 더 높은 학습률
N_EPOCHS = 20

# Mecab 기반 모델 초기화
model = GRUClassifier(
    vocab_size=FINAL_VOCAB_SIZE, 
    embedding_dim=EMBEDDING_DIM, 
    hidden_dim=HIDDEN_DIM, 
    output_dim=OUTPUT_DIM, 
    num_layers=NUM_LAYERS, 
    dropout=DROPOUT,
    pad_idx=word_to_index[PAD_TOKEN] # 패딩 인덱스 전달
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_GRU)

print("\n" + "=" * 60)
print(f"Mecab + GRU 분류 모델 학습 시작 (학습: {len(train_dataset)}개, 검증: {len(val_dataset)}개)")
print("=" * 60)

best_val_loss = float('inf')
SAVE_PATH = os.path.join(LOCAL_PATH, 'gru_mecab_classifier_best.pt')

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