import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
import os

# 데이터가 어떻게 출력될지 설정
pd.set_option('display.unicode.east_asian_width', True)

# 데이터 로드
input_file = r'.\crawling_data\naver_news_titles_cleaned20240701.csv'
df = pd.read_csv(input_file)
print(df.head())
df.info()

X = df['Title']  # 제목을 독립 변수로 설정
Y = df['Category']  # 카테고리를 종속 변수로 설정

# 레이블 인코딩
encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)  # 카테고리를 숫자로 변환
label = encoder.classes_  # 숫자로 변환하기 전의 카테고리를 변수에 저장
print("Labels:", label)  # 변환된 클래스의 레이블을 출력

# 인코더 저장
output_dir = '.'  # 출력 디렉토리 설정
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)  # 모델 저장 디렉토리 생성
with open(os.path.join(output_dir, 'models', 'encoder.pickle'), 'wb') as f:
    pickle.dump(encoder, f)  # 인코더를 파일로 저장

# 원-핫 인코딩
onehot_y = to_categorical(labeled_y)  # 레이블을 원-핫 인코딩 형태로 변환

# 형태소 분석
okt = Okt()
X = X.apply(lambda x: okt.morphs(x, stem=True))  # 제목의 텍스트를 형태소로 분석

# 불용어 제거
stopwords_file = './stopwords.csv'
stopwords = pd.read_csv(stopwords_file, index_col=0)
X = X.apply(lambda x: ' '.join([word for word in x if len(word) > 1 and word not in stopwords['stopword'].values]))

# 토큰화
token = Tokenizer()
token.fit_on_texts(X)  # 텍스트 데이터를 토큰화
tokened_x = token.texts_to_sequences(X)  # 텍스트를 시퀀스로 변환
wordsize = len(token.word_index) + 1
print("Vocabulary size:", wordsize)  # 어휘 사전의 크기를 출력

# 토큰화 객체 저장
with open(os.path.join(output_dir, 'models', 'news_token.pickle'), 'wb') as f:
    pickle.dump(token, f)  # 토크나이저를 파일로 저장

# 패딩
max_len = max(len(x) for x in tokened_x)  # 모든 데이터 중 최대 길이를 계산
x_pad = pad_sequences(tokened_x, max_len)  # 데이터의 길이를 맞추기 위해 패딩

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(x_pad, onehot_y, test_size=0.2)
print("Train shape:", X_train.shape, Y_train.shape)  # 학습 데이터의 형태 출력
print("Test shape:", X_test.shape, Y_test.shape)  # 테스트 데이터의 형태 출력

# 전처리된 데이터 저장
xy = (X_train, X_test, Y_train, Y_test)  # 데이터 묶기
os.makedirs(os.path.join(output_dir, 'crawling_data'), exist_ok=True)  # 데이터 저장 디렉토리 생성
with open(os.path.join(output_dir, 'crawling_data', f'news_data_max_{max_len}_wordsize_{wordsize}.pkl'), 'wb') as f:
    pickle.dump(xy, f)  # 전처리된 데이터 파일로 저장
