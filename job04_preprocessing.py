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

# 설정
pd.set_option('display.unicode.east_asian_width', True)

# 데이터 로드
input_file = r'.\crawling_data\naver_news_titles_20240626.csv'
df = pd.read_csv(input_file)
print(df.head())
df.info()

X = df['Title']
Y = df['Category']

# 레이블 인코딩
encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
label = encoder.classes_
print("Labels:", label)

# 인코더 저장
output_dir = '.'
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
with open(os.path.join(output_dir, 'models', 'encoder.pickle'), 'wb') as f:
    pickle.dump(encoder, f)

# 원-핫 인코딩
onehot_y = to_categorical(labeled_y)

# 형태소 분석
okt = Okt()
X = X.apply(lambda x: okt.morphs(x, stem=True))

# 불용어 제거
stopwords_file = './stopwords.csv'
stopwords = pd.read_csv(stopwords_file, index_col=0)
X = X.apply(lambda x: ' '.join([word for word in x if len(word) > 1 and word not in stopwords['stopword'].values]))

# 토큰화
token = Tokenizer()
token.fit_on_texts(X)
tokened_x = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
print("Vocabulary size:", wordsize)

# 토큰화 객체 저장
with open(os.path.join(output_dir, 'models', 'news_token.pickle'), 'wb') as f:
    pickle.dump(token, f)

# 패딩
max_len = max(len(x) for x in tokened_x)
x_pad = pad_sequences(tokened_x, max_len)

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(x_pad, onehot_y, test_size=0.2)
print("Train shape:", X_train.shape, Y_train.shape)
print("Test shape:", X_test.shape, Y_test.shape)

# 전처리된 데이터 저장
xy = (X_train, X_test, Y_train, Y_test)
os.makedirs(os.path.join(output_dir, 'crawling_data'), exist_ok=True)
with open(os.path.join(output_dir, 'crawling_data', f'news_data_max_{max_len}_wordsize_{wordsize}.pkl'), 'wb') as f:
    pickle.dump(xy, f)