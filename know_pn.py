import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from konlpy.tag import Okt
from collections import Counter

os.chdir(r'C:\ITWILL')

know = pd.read_csv('know_star.csv', encoding='utf-8')

know = know[know.star != '보고싶어요']
know = know[know.star != '보는중']
know = know[know.star != '없음']
know.info()

know = know.astype({'star' : 'float'})

know.value_counts(know.star)
know['star'].hist()


def apply_regular_expression(text):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')  # 한글 추출 규칙: 띄어 쓰기(1 개)를 포함한 한글
    result = hangul.sub('', text)  # 위에 설정한 "hangul"규칙을 "text"에 적용(.sub)시킴
    return result

apply_regular_expression(know['review'][0])

okt = Okt()
nouns = okt.nouns(apply_regular_expression(know['review'][0]))
nouns

corpus = "".join(know['review'].tolist())
corpus

apply_regular_expression(corpus)

nouns = okt.nouns(apply_regular_expression(corpus))
print(nouns)

counter = Counter(nouns)

counter.most_common(10)

available_counter = Counter({x: counter[x] for x in counter if len(x) > 1})
available_counter.most_common(10)

stopwords = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()
stopwords[:10]

my_stopwords = ['불구', '제발', '는걸', '최고다', '건가']
for word in my_stopwords:
    stopwords.append(word)
    

from sklearn.feature_extraction.text import CountVectorizer

def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')  # 정규 표현식 처리
    result = hangul.sub('', text)
    okt = Okt()  # 형태소 추출
    nouns = okt.nouns(result)
    nouns = [x for x in nouns if len(x) > 1]  # 한글자 키워드 제거
    nouns = [x for x in nouns if x not in stopwords]  # 불용어 제거
    return nouns

vect = CountVectorizer(tokenizer = lambda x: text_cleaning(x))
bow_vect = vect.fit_transform(know['review'].tolist())
word_list = vect.get_feature_names()
count_list = bow_vect.toarray().sum(axis=0)

word_list
count_list

bow_vect.shape


word_count_dict = dict(zip(word_list, count_list))
word_count_dict

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)
print(tf_idf_vect.shape)

print(tf_idf_vect[0])

vect.vocabulary_

invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
print(str(invert_index_vectorizer)[:100]+'...')


# 감성분석 start

def rating_to_label(star):
    if star > 3:
        return 1
    else:
        return 0
    
know['y'] = know['star'].apply(lambda x: rating_to_label(x))
know.info()
know.head()

know['y'].value_counts()

# test/train set 나누기
from sklearn.model_selection import train_test_split

x = tf_idf_vect
y = know['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)
x_train.shape, y_train.shape

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# fit in training set
lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)

# predict in test set
y_pred = lr.predict(x_test)

print('accuracy: %.2f' % accuracy_score(y_test, y_pred)) # 0.61
print('precision: %.2f' % precision_score(y_test, y_pred)) # 0.63
print('recall: %.2f' % recall_score(y_test, y_pred)) # 0.22
print('F1: %.2f' % f1_score(y_test, y_pred)) # 0.32

# confusion matrix

from sklearn.metrics import confusion_matrix

confu = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(confu)

plt.figure(figsize=(4, 3))
sns.heatmap(confu, annot=True, annot_kws={'size':15}, cmap='OrRd', fmt='.10g')
plt.title('Confusion Matrix')
plt.show()



plt.figure(figsize=(10, 8))
plt.bar(range(len(lr.coef_[0])), lr.coef_[0])

print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse = True)[:5])
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse = True)[-5:])

coef_pos_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse = True)
coef_neg_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse = False)

invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
invert_index_vectorizer

for coef in coef_pos_index[:20]:
    print(invert_index_vectorizer[coef[1]], coef[0])
'''
사랑 1.1152966065087864
연애 1.0103330560003532
케미 0.7258438769468887
분위기 0.6829579425242286
나라 0.6402651894301273
마음 0.6158654014574617
노래 0.6108459229424176
이유 0.6043537873422562
태도 0.5814033919416468
감각 0.5777651784283937
나비 0.5671167558769685
완벽 0.5592586475094333
얼굴 0.5462081695969305
언니 0.5446975881107359
에피소드 0.5397305112266705
나쁜남자 0.5320057412746584
문제 0.515462899517604
소희 0.5141749043344156
이야기 0.5051344283168149
여럿 0.5006602442193597
'''


for coef in coef_neg_index[:20]:
    print(invert_index_vectorizer[coef[1]], coef[0])
'''
드라마 -0.9319993215378131
내용 -0.9111609117229933
비주 -0.8692589025712348
캐스팅 -0.8039690209830139
하차 -0.7215060261647737
중간 -0.6876035487422163
매력 -0.6402687226052151
별개 -0.6363447735868853
참고 -0.6263307476910615
송강 -0.6109747886459603
주인공 -0.6086237154511678
치명 -0.5967486948639917
무슨 -0.567657731938863
마무리 -0.5393849350543879
보지 -0.4581328840871607
분량 -0.4549167746286115
화로 -0.4214419050849403
도대체 -0.4195233674485137
소녀 -0.4177927334488438
완만 -0.41499862256952713
'''

