import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import pickle

# 데이터 로드
with open('.//models/preprocessed_data/news_data_max_30_wordsize_13129.pkl', 'rb') as f:
    X_train, X_test, Y_train, Y_test = pickle.load(f)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# GRU를 사용한 모델 구성
model = Sequential()
model.add(Embedding(13129, 1000, input_length=30))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(GRU(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

# 모델 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=5, validation_data=(X_test, Y_test))

# 모델 저장
accuracy = fit_hist.history['val_accuracy'][-1]  # 마지막 검증 정확도 추출
model.save(f'../models/news_category_classification_model_gru_{accuracy:.4f}.keras')  # 정확도를 파일 이름에 포함

# 훈련 결과 시각화
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()
