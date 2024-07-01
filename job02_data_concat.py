import pandas as pd
import glob
import datetime

data_path = glob.glob('./crawling_data2/*')
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

df.to_csv('./crawling_data/naver_news_titles_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)