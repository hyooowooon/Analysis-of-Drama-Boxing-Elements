from bs4 import BeautifulSoup  # html 파싱
from selenium import webdriver as wd
import time
# import lxml.html
from selenium.webdriver.common.keys import Keys

path = r"C:\ITWILL\5_Tensorflow\workspace"
driver = wd.Chrome(path + '/chromedriver.exe')
url_list = ['https://pedia.watcha.com/ko-KR/contents/tlnN20e/comments']

all_stars = []  # 평점 저장
all_comments = []  # 리뷰 저장

for url in url_list:
    driver.get(url)
    driver.maximize_window()

    time.sleep(2)

    pause_sec = 2

    body = driver.find_element_by_tag_name('body')

    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(2)
    body.send_keys(Keys.PAGE_UP)

    while True:
        last_height = driver.execute_script('return document.documentElement.scrollHeight')
        body.send_keys(Keys.END)
        time.sleep(pause_sec)
        new_height = driver.execute_script('return document.documentElement.scrollHeight')

        if new_height == last_height:
            break;

    html_source = driver.page_source

    html = BeautifulSoup(html_source, "html.parser")  # "lxml"

    # 평점 저장
    print(html)
    stars = html.select(
        '#root > div > div.css-1xm32e0 > section > section > div > div > div > ul > div > div.css-4obf01 > div.css-yqs4xl > span')
    print(len(stars))  # 17 -> 평점 없는 댓글(4개 있음)
    # 평점 경로 : ~ div.css-4obf01 > div.css-yqs4xl > span

    # 평점 부모 경로 : ~ div.css-4obf01
    stars_parents = html.select(
        '#root > div > div.css-1xm32e0 > section > section > div > div > div > ul > div > div.css-4obf01')
    print(len(stars_parents))  # 21 : 평점 부모 경로로 tag 수집

    # 리뷰 저장 : 평점 부모 경로로 리뷰 tag 수집
    tag = html.select(
        '#root > div > div.css-1xm32e0 > section > section > div > div > div > ul > div > div.css-4tkoly > div > span')
    print(len(tag))  # 21

    for st in stars_parents:  # 평점 부모 경로로 tag 평점 수집
        span = st.select_one('div.css-yqs4xl > span')  # html.select("tr[class='odd']")
        if span:  # 평점 태그 있는 경우
            star = str(span.text)
            all_stars.append(star)
        else:  # 평점 태그 없는 경우
            all_stars.append('없음')

    for t in tag:
        comment = str(t.text)
        all_comments.append(comment)

    print('평점 개수 :', len(all_stars))  # 평점 개수 : 21
    print('리뷰 개수 :', len(all_comments))  # 리뷰 개수 : 21

driver.close()

import pandas as pd

df = pd.DataFrame({'star': all_stars, 'review': all_comments},
                  columns=['star', 'review'])

# 2) csv file save
df.to_csv('itwon_star.csv', index=False, encoding="utf-8-sig")