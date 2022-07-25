from konlpy.tag import Kkma  # class - 형태소 분석기
from wordcloud import WordCloud  # class - 단어구름 시각화

# 1. 이태원클라쓰 (흥행)
# 1-1. text file(docs) 읽기
path = r"C:\ITWILL\5_Tensorflow"

file = open(path + '/itwon.txt', encoding='utf-8')
para = file.read()  # 문자열

print(para)
type(para)  # str
file.close()

# 1-2. 문단(문자열) -> 문장(list)
kkma = Kkma()

ex_sents = kkma.sentences(para)  # list 반환
len(ex_sents)  # 636

# 문단(문자열) -> 명사(list)
ex_nouns = kkma.nouns(para)  # 유일한 명사 추출
len(ex_nouns)  # 2709

# 1-3. 문장 -> 단어(명사) 추출
nouns = []  # 중복 명사 저장

for sent in ex_sents:  # 문단 -> 문장
    for noun in kkma.nouns(sent):  # 문장 -> 명사 추출
        nouns.append(noun)

len(nouns)  # 6303

# 1-4. 전처리 & 단어 카운트 : 1음절 제외 & 서수 제외
from re import match  # 서수 제외

wc = {}  # 단어 카운트

for noun in nouns:
    if len(noun) > 1 and not (match('^[0-9]', noun)):  # 전처리
        wc[noun] = wc.get(noun, 0) + 1  # 단어 카운트

print(wc)
len(wc)  # 1645

# 1-5. 단어구름 시각화

# 1) topN word 선정
from collections import Counter  # class

counter = Counter(wc)
top25_word = counter.most_common(25)
print(top25_word)
'''
[('드라마', 375),
 ('캐릭터', 221),
 ('사람', 146),
 ('생각', 139),
 ('원작', 138),
 ('배우', 134),
 ('연기', 116),
 ('웹툰', 110),
 ('느낌', 96),
 ('전개', 94),
 ('사랑', 93),
 ('마지막', 91),
 ('이서', 87),
 ('스토리', 82),
 ('초반', 80),
 ('박새', 78),
 ('대사', 76),
 ('이태원', 75),
 ('조이', 71),
 ('후반부', 63),
 ('이야기', 62),
 ('소신', 60),
 ('인생', 59),
 ('유치', 54),
 ('성공', 53)]
'''

# word cloud
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',
               width=500, height=400,
               max_words=100, max_font_size=150,
               background_color='white')

wc_result = wc.generate_from_frequencies(dict(top25_word))

import matplotlib.pyplot as plt

plt.imshow(wc_result)
plt.axis('off')  # 축 눈금 감추기
plt.show()

#################################################################################3
# 2. 알고있지만 (비흥행)
# 2-1. text file(docs) 읽기
file = open(path + '/know2.txt', encoding='utf-8')
para = file.read()  # 문자열

print(para)
type(para)  # str
file.close()

# 2-2. 문단(문자열) -> 문장(list)
kkma = Kkma()

ex_sents = kkma.sentences(para)  # list 반환
len(ex_sents)  # 351

# 문단(문자열) -> 명사(list)
ex_nouns = kkma.nouns(para)  # 유일한 명사 추출
len(ex_nouns)  # 1940

# 2-3. 문장 -> 단어(명사) 추출
nouns = []  # 중복 명사 저장

for sent in ex_sents:  # 문단 -> 문장
    for noun in kkma.nouns(sent):  # 문장 -> 명사 추출
        nouns.append(noun)

len(nouns)  # 4464

# 2-4. 전처리 & 단어 카운트 : 1음절 제외 & 서수 제외
from re import match  # 서수 제외

wc = {}  # 단어 카운트

for noun in nouns:
    if len(noun) > 1 and not (match('^[0-9]', noun)):  # 전처리
        wc[noun] = wc.get(noun, 0) + 1  # 단어 카운트

print(wc)
len(wc)  # 1198

# 2-5. 단어구름 시각화

# 1) topN word 선정
from collections import Counter  # class

counter = Counter(wc)
top25_word = counter.most_common(25)
'''
[('드라마', 81),
 ('송강', 73),
 ('연기', 63),
 ('박재', 62),
 ('소희', 53),
 ('생각', 49),
 ('나비', 48),
 ('배우', 44),
 ('원작', 38),
 ('사람', 37),
 ('얼굴', 36),
 ('비주얼', 33),
 ('사랑', 33),
 ('유나', 29),
 ('재언', 29),
 ('웹툰', 29),
 ('연출', 26),
 ('대사', 22),
 ('결말', 22),
 ('느낌', 22),
 ('마음', 21),
 ('때문', 21),
 ('캐릭터', 21),
 ('감정', 18),
 ('내가', 18)]
'''

# word cloud
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',
               width=500, height=400,
               max_words=100, max_font_size=150,
               background_color='white')

wc_result = wc.generate_from_frequencies(dict(top25_word))

import matplotlib.pyplot as plt

plt.imshow(wc_result)
plt.axis('off')  # 축 눈금 감추기
plt.show()

