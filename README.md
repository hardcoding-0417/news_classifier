# news_classifier
뉴스 분류기

## RNN을 활용한 뉴스 카테고리 분류 실습

이 레포는 학생들이 RNN(Recurrent Neural Network)을 쉽게 이해하고 실습할 수 있도록 
뉴스 제목을 분류하는 모델을 제작하는 프로젝트가 담긴 자료실입니다.

천천히 따라하며 파이썬, 크롤링, 자연어 전처리, RNN 학습의 기초를 배워보세요.

## 프로젝트 목표
- 웹 크롤링을 통해 실시간 뉴스 데이터를 수집합니다.
- 수집된 데이터를 전처리하여 기계 학습 모델의 입력으로 사용할 수 있도록 준비합니다.  
- RNN 모델을 구축하고 훈련시켜 뉴스 카테고리를 자동으로 분류합니다.
- 모델의 성능을 평가하고 결과를 해석합니다.

## 라이브러리 설치
```bash
pip install pandas numpy selenium tensorflow konlpy matplotlib
```

- Pandas: 데이터 처리를 위해 필요합니다.
- NumPy: 수치 계산을 위해 사용됩니다.
- Selenium: 웹 크롤링을 위한 도구입니다.
- TensorFlow: 딥러닝 모델을 구성하고 훈련시키는 데 사용됩니다. 
- Konlpy: 한국어 자연어 처리를 위해 필요합니다.
- Matplotlib: 결과를 시각화하기 위해 사용됩니다.

Konlpy의 종속성 중 하나인 JPype1의 설치를 위해 아래의 명령어가 필요할 수도 있습니다.

```bash
sudo apt-get install g++ openjdk-11-jdk python3-dev python3-pip curl
```

## 디렉토리 구조

- `crawling_data/`: 웹 크롤링을 통해 수집된 뉴스 데이터가 저장되는 폴더입니다.
- `models/`: 훈련된 모델과 토큰화 파일이 저장되는 폴더입니다.
- `stopwords.csv`: 데이터 전처리 시 사용되는 불용어 사전입니다.
- `src/`: 
  - `crawl_news.py`: 뉴스 데이터를 웹에서 크롤링하는 작업을 수행합니다.
  - `preprocess_data.py`: 크롤링된 데이터를 정제하는 작업을 수행합니다.
  - `train_model.py`: 정제된 데이터를 사용하여 모델을 훈련시키는 스크립트입니다.
  - `evaluate_model.py`: 훈련된 모델을 평가하고 결과를 출력하는 스크립트입니다.

## 프로젝트 진행 순서
1. Selenium으로 네이버 뉴스의 헤드라인들을 크롤링합니다.  
2. 크롤링된 데이터를 정제하고, 토큰화하여 모델 학습에 적합한 형태로 변환합니다.
3. TensorFlow를 사용하여 RNN 기반의 모델을 구축하고 훈련합니다.
4. 뉴스 데이터의 카테고리를 예측하고, 모델의 성능을 평가합니다.

## 참고자료
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Konlpy Documentation](http://konlpy.org/ko/latest/)

실제 데이터로 작업해보면 딥러닝을 쉽게 이해할 수 있게 됩니다. 하나씩 따라해보세요.