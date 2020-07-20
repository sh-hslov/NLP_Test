# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:01:26 2020

@author: SungHyun
"""

# =============================================================================
# 여러 클래스 중 Okt(Open Korean Text) 클래스를 이용 예제
# 클래스 참고 https://konlpy-ko.readthedocs.io/ko/v0.4.3/morph/
# =============================================================================
import konlpy
from konlpy.tag import Okt
import json
import os
from pprint import pprint

# =============================================================================
# https://konlpy-ko.readthedocs.io/ko/v0.5.2/install/#id2 참고
# JPye1 설치시 cp뒤 번호가 python 버전 cp37 = 3.7버전과 호환
# =============================================================================

# =============================================================================
# 'cp949' codec can't decode byte 0xec in position 26: illegal multibyte sequence 에러시
#  encoding='UTF8' 추가.
# =============================================================================
def read_data(filename):
    with open(filename, 'r', encoding='UTF8') as f: 
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
        
    return data

train_data = read_data('../NSMC_Movie/nsmc-master/nsmc-master/ratings_train.txt')
test_data = read_data('../NSMC_Movie/nsmc-master/nsmc-master/ratings_test.txt')

# =============================================================================
# print(len(train_data))
# print(len(train_data[0]))
# print(len(test_data))
# print(len(test_data[0]))
# 
# =============================================================================

okt = Okt()
# =============================================================================
# print(okt.pos(u'이 밤 그날의 반딧불을 당신의 창 가까이 보낼게'))
# =============================================================================

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('../NSMC_Movie/nsmc-master/nsmc-master/train_docs.json'):
    with open('../NSMC_Movie/nsmc-master/nsmc-master/train_docs.json', encoding="utf-8") as f:
        train_docs = json.load(f)
    with open('../NSMC_Movie/nsmc-master/nsmc-master/test_docs.json', encoding="utf-8") as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # JSON 파일로 저장
    with open('../NSMC_Movie/nsmc-master/nsmc-master/train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('../NSMC_Movie/nsmc-master/nsmc-master/test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")


# 예쁘게(?) 출력하기 위해서 pprint 라이브러리 사용
pprint(train_docs[0])

#데이터 토큰 개수 확인
tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))

#nltk 라이브러리 사용 전처리를 위해서
import nltk
text = nltk.Text(tokens, name='NMSC')
print(text)

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))            

# 출현 빈도가 높은 상위 토큰 10개
pprint(text.vocab().most_common(10))


#출현빈도 높은 단어 출력
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
#%matplotlib inline

#font_fname = '../NSMC_Movie/Font/AppleGothic.ttf' #깨지네?
font_fname = 'c:/windows/fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
plt.figure(figsize=(20,10))
text.plot(50)

#데이터 벡터화. 원 핫 인코딩 대신 CountVectorization 사용
selected_words = [f[0] for f in text.vocab().most_common(10000)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]

# import numpy as np

# x_train = np.asarray(train_x).astype('float32')
# train_x = np.asarray(train_x).astype('float32')
# x_test = np.asarray(test_x).astype('float32')

# y_train = np.asarray(train_y).astype('float32')
# y_test = np.asarray(test_y).astype('float32')


# from tensorflow.keras import models
# from tensorflow.keras import layers
# from tensorflow.keras import optimizers
# from tensorflow.keras import losses
# from tensorflow.keras import metrics

# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

# model.fit(x_train, y_train, epochs=10, batch_size=512)
# results = model.evaluate(x_test, y_test)

