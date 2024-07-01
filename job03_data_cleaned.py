import pandas as pd
import glob
import datetime
import re

data_path = glob.glob('./crawling_data/*')
print(data_path)

df = pd.DataFrame()
for path in data_path:
    df_temp = pd.read_csv(path, skipinitialspace=True)
    df_temp = df_temp.dropna(subset=['Title'])  # Title이 없는 행 제거
    df = pd.concat([df, df_temp], ignore_index=True)

print(df.head())
print(df['Category'].value_counts())
df.info()

# 중복 제거
df = df.drop_duplicates()

# 제목 정제
def clean_title(title):
    # 왼쪽 공백 제거
    title = title.lstrip()
    # 연속된 공백을 하나의 공백으로 대체
    title = re.sub(r'\s+', ' ', title)
    return title

df['Title'] = df['Title'].apply(clean_title)

# CSV 파일로 저장
df.to_csv('./crawling_data/naver_news_titles_cleaned{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)

print(df.head())  # 결과 확인