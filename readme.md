uv venv venv

.\venv\Scripts\Activate.ps1

// 3060
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

// 5060ti
uv pip uninstall torch torchvision torchaudio -y
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

stop_word=["그","에서","나","에","을","를","다","의","는",...]

# 형태소 분석 및 불용어 제거 통합:

all_tokenized_reviews = []
for review in raw['review'].tolist():
tok = mecab.morphs(review) # 1) 형태소 분석 (Mecab)
tmp = [w for w in tok if not w in stop_word] # 2) 불용어 제거
all_tokenized_reviews.append(tmp)

# 1. 단어 빈도수 계산

word_counts = {}
for token_list in all_tokenized_reviews: # <-- 정제된 데이터 사용 # ...

1. 형태소 분석, Okt (Konlpy 라이브러리), 문장(텍스트)을 의미를 가지는 가장 작은 단위인 **형태소(단어의 쪼개진 부분)**로 나눕니다.
2. 어간 추출, stem=True 옵션, 동사/형용사의 활용형(예: "먹고", "먹으니")을 기본형/원형("먹다")으로 통일시킵니다.
3. 불용어 제거, stop_word 리스트, 의미를 가지지 않고 단순히 문법적 역할만 하는 단어들(예: "은", "는", "이", "가", "들")을 목록에서 찾아 제거합니다.

원본 문장 (doc) 형태소 분석 (tok) 불용어 제거 (tmp)
"저는 쇼핑몰에서 좋은 상품들을 샀어요." "저", "는", "쇼핑몰", "에서", "좋다", "상품", "들", "을", "사다", "요" "쇼핑몰", "좋다", "상품", "사다"

#단어 출현 빈도를 계산하여 빈도수가 낮은 단어는 제거
