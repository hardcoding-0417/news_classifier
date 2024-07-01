import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from konlpy.tag import Okt
from openai import OpenAI
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

# 형태소 분석
okt = Okt()
X = X.apply(lambda x: okt.morphs(x, stem=True))

# 불용어 제거
stopwords_file = './stopwords.csv'
stopwords = pd.read_csv(stopwords_file, index_col=0)
X = X.apply(lambda x: ' '.join([word for word in x if len(word) > 1 and word not in stopwords['stopword'].values]))

# 임베딩 생성
client = OpenAI()

def get_embedding(texts, model="text-embedding-3-small", batch_size=1000):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

# 학습 데이터와 테스트 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, labeled_y, test_size=0.2, random_state=42)

def generate_and_save_embeddings(texts, filename):
    texts = texts.tolist()  # 시리즈를 리스트로 변환
    embeddings = get_embedding(texts)
    np_embeddings = np.array(embeddings)
    with open(filename, 'wb') as f:
        pickle.dump(np_embeddings, f)
    print(f"Embedding complete. Saved to {filename}")

# 학습 데이터 임베딩 생성 및 저장
generate_and_save_embeddings(X_train, os.path.join(output_dir, 'crawling_data', 'X_train_embeddings.pkl'))

# 테스트 데이터 임베딩 생성 및 저장 
generate_and_save_embeddings(X_test, os.path.join(output_dir, 'crawling_data', 'X_test_embeddings.pkl'))

# 라벨 데이터 저장
with open(os.path.join(output_dir, 'crawling_data', 'labels.pkl'), 'wb') as f:
    pickle.dump((Y_train, Y_test), f)