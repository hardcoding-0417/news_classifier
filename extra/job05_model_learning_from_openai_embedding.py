import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import pickle
import os

# 데이터 로드
output_dir = '.'
with open(os.path.join(output_dir, 'crawling_data', 'X_train_embeddings.pkl'), 'rb') as f:
    X_train = pickle.load(f)
with open(os.path.join(output_dir, 'crawling_data', 'X_test_embeddings.pkl'), 'rb') as f:
    X_test = pickle.load(f)
with open(os.path.join(output_dir, 'crawling_data', 'labels.pkl'), 'rb') as f:
    Y_train, Y_test = pickle.load(f)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# OpenAI 임베딩 차원 설정
embedding_dim = 1536  # OpenAI의 text-embedding-3-small 모델의 임베딩 차원

# 입력 데이터 reshape
X_train = X_train.reshape(X_train.shape[0], 1, embedding_dim)
X_test = X_test.reshape(X_test.shape[0], 1, embedding_dim)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


model = Sequential()
model.add(InputLayer(shape=(None, embedding_dim)))
model.add(GRU(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

# 모델 컴파일 및 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_test, Y_test))

# 모델 저장
model.save('./models/news_category_classification_model_openai_gru_{}.keras'.format(fit_hist.history['val_accuracy'][-1]))

# 훈련 결과 시각화
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()