import os  # 운영 체제와 관련된 기능을 사용하기 위한 라이브러리
import pandas as pd  # 데이터를 표 형식(data frame)으로 다루기 위한 라이브러리
from selenium import webdriver  # 웹 브라우저를 자동으로 조작하기 위한 라이브러리
from selenium.webdriver.chrome.service import Service as ChromeService # 크롬을 조작하기 위한 클래스
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By  # 웹 요소를 찾기 위한 클래스
from selenium.webdriver.support import expected_conditions as EC  # WebDriver와 연계하여 예상되는 조건들을 정의한 모듈
from selenium.webdriver.support.ui import WebDriverWait  # 원하는 게 로딩되면 바로바로 실행하기 위해 특정 요소를 기다리는 클래스. EC와 연계되면 특정 조건이 달성될 때까지 대기하도록 설정 가능
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException  # 예외 처리용
import re  # 정규 표현식을 표현하기 위한 라이브러리
import time  # sleep()을 사용하기 위한 라이브러리
import sys  # 특정 매개변수와 함수를 사용하기 위한 시스템 라이브러리

# 크롤링한 데이터를 저장할 디렉토리를 지정합니다.
output_dir = "../crawling_data"

# 출력 디렉토리가 없으면 생성합니다.
os.makedirs(output_dir, exist_ok=True)

# Chrome 브라우저 옵션을 설정합니다.
options = ChromeOptions()
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
options.add_argument('user-agent=' + user_agent)
options.add_argument('lang=ko_KR')  # 언어를 한국어로 설정합니다.

# 브라우저를 자동으로 제어하기 위해 Chrome WebDriver를 설정합니다.
service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# 크롤링할 뉴스 카테고리를 정의합니다.
categories = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']

# 각 카테고리별로 크롤링을 수행합니다.
for category_num, category_name in zip(range(100, 106), categories):
    print(f"크롤링 중인 카테고리: {category_name}")
    driver.get(f"https://news.naver.com/section/{category_num}")
    time.sleep(2)  # 페이지가 완전히 로드될 때까지 기다립니다.

    for click_count in range(50):
        try:
            more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.section_more_inner[data-persistable='false']"))
            )
            more_button.click()
            time.sleep(1)
            print(f"클릭 {click_count + 1} 완료.")
        except TimeoutException:
            print("더 이상 '더보기' 버튼을 찾을 수 없습니다.")
            break

    all_titles = set()
    headlines = driver.find_elements(By.CSS_SELECTOR, "strong.sa_text_strong")
    for headline in headlines:
        try:
            title = headline.text
            title = re.sub(r'[^가-힣0-9\s|a-z|A-Z]', ' ', title)
            all_titles.add(title)
        except (NoSuchElementException, StaleElementReferenceException):
            print(f'헤드라인 추출 중 오류 발생: {headline}')

    print(f"총 고유 헤드라인 수: {len(all_titles)}")
    df_category = pd.DataFrame({
        'Category': [category_name] * len(all_titles),
        'Title': list(all_titles)
    })
    csv_filename = os.path.join(output_dir, f"naver_news_{category_name}.csv")
    df_category.to_csv(csv_filename, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(csv_filename))
    print(f"헤드라인을 {csv_filename}에 저장했습니다.")

driver.quit()
