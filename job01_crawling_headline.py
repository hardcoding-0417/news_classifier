import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import re
import time
import sys

output_dir = "crawling_data"

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# Chrome 옵션 설정
options = ChromeOptions()
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
options.add_argument('user-agent=' + user_agent)
options.add_argument('lang=ko_KR')

# WebDriver 설정
service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# 카테고리 정의
categories = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']

# 카테고리별로 크롤링
for category_num, category_name in zip(range(100, 106), categories):
    print(f"Crawling category: {category_name}")
    
    # 네이버 뉴스 페이지 열기
    driver.get(f"https://news.naver.com/section/{category_num}")
    time.sleep(2)

    # 더보기 버튼 50번 클릭
    for click_count in range(50):
        try:
            more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.section_more_inner[data-persistable='false']"))
            )
            more_button.click()
            time.sleep(1)
            print(f"Click {click_count + 1} completed.")
        except TimeoutException:
            print("더 이상 '더보기' 버튼을 찾을 수 없습니다.")
            break

    # 모든 클릭이 완료된 후 헤드라인 추출
    all_titles = set()
    headlines = driver.find_elements(By.CSS_SELECTOR, "strong.sa_text_strong")

    for headline in headlines:
        try:
            title = headline.text
            # 한글, 숫자, 공백만 남기고 나머지는 제거
            title = re.sub(r'[^가-힣0-9\s|a-z|A-Z]', ' ', title)
            all_titles.add(title)  # set에 추가하여 자동으로 중복 제거
        except (NoSuchElementException, StaleElementReferenceException):
            print(f'Error extracting headline: {headline}')

    print(f"Total unique headlines: {len(all_titles)}")

    # 카테고리별 데이터프레임 생성
    df_category = pd.DataFrame({
        'Category': [category_name] * len(all_titles),
        'Title': list(all_titles)
    })

    # CSV 파일로 저장
    csv_filename = os.path.join(output_dir, f"naver_news_{category_name}.csv")
    df_category.to_csv(csv_filename, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(csv_filename))
    print(f"Headlines saved to {csv_filename}")

# 브라우저 종료
driver.quit()
