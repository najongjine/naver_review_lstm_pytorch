import torch
import torch.nn as nn
import tiktoken 
import os
import numpy as np

# =================================================================
# 1. 환경 설정 및 상수 재정의 (학습 코드와 동일하게 유지)
# =================================================================

# ⭐️ tiktoken 인코더 로드 (학습 시 사용한 인코더와 동일해야 함) ⭐️
ENCODER_NAME = "cl100k_base" # GPT-4, GPT-3.5 Turbo에 사용되는 인코딩
tokenizer = tiktoken.get_encoding(ENCODER_NAME)

# tiktoken 기반 상수
MAX_LEN = 512
PAD_TOKEN_ID = tokenizer.eot_token 
FINAL_VOCAB_SIZE = tokenizer.n_vocab 
PAD_IDX = PAD_TOKEN_ID

# GRU 모델 하이퍼파라미터 (학습 코드와 동일해야 함)
EMBEDDING_DIM = 300 
HIDDEN_DIM = 512 
OUTPUT_DIM = 2
NUM_LAYERS = 2
DROPOUT = 0.5

# 모델 저장 경로 및 파일명 (사용자 정보 기반)
LOCAL_PATH = './my_local_models/gru_tiktoken_clean/review_classify' 
SAVE_PATH = os.path.join(LOCAL_PATH, 'gru_tiktoken_classifier_best.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# =================================================================
# 2. GRU 기반 분류기 모델 정의 (학습 코드와 완전히 동일해야 함)
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
        # 양방향 RNN의 마지막 은닉 상태를 연결 (hidden[-2]와 hidden[-1])
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) 
        prediction = self.fc(hidden) 

        return prediction

# =================================================================
# 3. 예측을 위한 전처리 및 분류 함수
# =================================================================

def preprocess_and_tokenize(review, max_len, pad_id):
    """tiktoken으로 텍스트를 정수 인코딩하고 패딩/잘라내기 처리"""
    # 1. 텍스트를 tiktoken으로 인코딩
    encoded = tokenizer.encode(review)
    
    # 2. 패딩 또는 잘라내기
    if len(encoded) < max_len:
        encoded += [pad_id] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
        
    # 3. PyTorch Long Tensor 형태로 변환 (배치 크기 1 추가: [1, MAX_LEN])
    return torch.tensor([encoded], dtype=torch.long)

def classify_review(model, review_text, device):
    """주어진 텍스트를 모델로 분류하고 결과를 반환"""
    # 1. 모델을 평가 모드로 설정
    model.eval()

    # 2. 텍스트 전처리 및 텐서 변환
    input_ids = preprocess_and_tokenize(review_text, MAX_LEN, PAD_IDX).to(device)

    # 3. 예측 수행 (경사 계산 비활성화)
    with torch.no_grad():
        predictions = model(input_ids)
        
    # 4. 확률 계산 (Softmax) 및 최종 클래스 (0 또는 1) 결정
    probabilities = torch.softmax(predictions, dim=1).cpu().numpy()[0]
    predicted_class = predictions.argmax(1).item() # 0: 부정, 1: 긍정
    
    sentiment = "긍정 (Positive)" if predicted_class == 1 else "부정 (Negative)"
    
    return sentiment, probabilities

# =================================================================
# 4. 모델 로드 및 사용자 입력 처리
# =================================================================

def main(user_query):
    # 모델 초기화
    model = GRUClassifier(
        vocab_size=FINAL_VOCAB_SIZE, 
        embedding_dim=EMBEDDING_DIM, 
        hidden_dim=HIDDEN_DIM, 
        output_dim=OUTPUT_DIM, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT,
        pad_idx=PAD_IDX 
    ).to(device)
    
    # 학습된 가중치 로드
    if not os.path.exists(SAVE_PATH):
        print(f"❌ 에러: 저장된 모델 파일 '{SAVE_PATH}'을 찾을 수 없습니다.")
        print("학습 코드를 먼저 실행하여 모델을 저장했는지 확인해 주세요.")
        return
        
    try:
        # 모델 상태 사전 로드 (GPU에서 학습했더라도, 현재 장치에 맞게 로드)
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        print(f"✅ 모델 로드 성공: '{SAVE_PATH}'")
    except Exception as e:
        print(f"❌ 에러: 모델 로드 중 오류 발생: {e}")
        return

    print("\n" + "=" * 50)
    print(f"입력 리뷰: **'{user_query}'**")
    
    # 분류 실행
    sentiment, probabilities = classify_review(model, user_query, device)

    # 결과 출력
    print("-" * 50)
    print(f"⭐ 예측 결과: **{sentiment}**")
    print(f"  * 부정(0) 확률: {probabilities[0]:.4f}")
    print(f"  * 긍정(1) 확률: {probabilities[1]:.4f}")
    print("=" * 50)
    
# ------------------------------------------------------------------

# 사용자 입력
user_query = "왜 이따구임?" 

if __name__ == "__main__":
    main(user_query)