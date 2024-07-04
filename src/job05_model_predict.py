import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import pickle
import random
import os

# 훈련된 모델 로드
model = load_model('../models/news_category_classification_model_gru_0.69.keras')

# 데이터 로드
news_data = pd.read_csv('../crawling_data/naver_news_titles_cleaned20240704.csv')
X_data = news_data['Title']
y_data = news_data['Category']

# 전처리 함수
def preprocess_text(text):
    okt = Okt()
    stopwords = pd.read_csv('../stopwords.csv', index_col=0)
    
    text = okt.morphs(text, stem=True)
    text = [word for word in text if len(word) > 1 and word not in stopwords['stopword'].values]
    return ' '.join(text)

X_data = X_data.apply(preprocess_text)

# 전처리된 데이터 로드
with open('../models/preprocessed_data/news_data_max_30_wordsize_13129.pkl', 'rb') as f:
    X_train, X_test, Y_train, Y_test, encoder, tokenizer = pickle.load(f)

# 텍스트 데이터 토큰화 및 패딩
tokened_X = tokenizer.texts_to_sequences(X_data)
X_pad = pad_sequences(tokened_X, maxlen=30)

# 추론
y_prob = model.predict(X_pad)
y_pred = encoder.inverse_transform(np.argmax(y_prob, axis=1))

# 랜덤으로 10개의 기사 선택
random_indices = random.sample(range(len(news_data)), 10)

# 결과 출력
for i in random_indices:
    print(f"기사 제목: {news_data.iloc[i]['Title']}")
    print(f"실제 카테고리: {y_data.iloc[i]}")
    print(f"예측된 카테고리: {y_pred[i]}")
    print(f"예측 결과: {'O' if y_data.iloc[i] == y_pred[i] else 'X'}")
    print("---")
