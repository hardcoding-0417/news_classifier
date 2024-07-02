import pandas as pd
import glob
import datetime  # 현재 시각을 기록하기 위한 라이브러리.
import re

# '../crawling_data/' 폴더 안의 모든 파일 경로를 가져옵니다.
data_path = glob.glob('../crawling_data/*')
print(data_path)

# csv들을 읽어서 하나로 합칩니다.
df = pd.DataFrame()
for path in data_path:
    df_temp = pd.read_csv(path, skipinitialspace=True)
    df_temp = df_temp.dropna(subset=['Title'])  # Title이 없는 행 제거
    df = pd.concat([df, df_temp], ignore_index=True)  # 깨끗해진 csv를 하나로 합칩니다.

print(df.head())  # 최초의 5개 출력
print(df['Category'].value_counts())  # 카테고리별 개수 출력
df.info()

# 중복 제거
df = df.drop_duplicates()

# 제목 정제 함수 정의
def clean_title(title):
    # 왼쪽 공백 제거
    title = title.lstrip()
    # 연속된 공백을 하나의 공백으로 대체
    title = re.sub(r'\s+', ' ', title)
    return title

# 'Title' 열에 clean_title 함수 적용
df['Title'] = df['Title'].apply(clean_title)

# 정제된 데이터를 CSV 파일로 저장
df.to_csv('../crawling_data/naver_news_titles_cleaned{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)

print(df.head())  # 결과 확인
