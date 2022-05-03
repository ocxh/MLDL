import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

#데이터 준비
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

#표준화 전처리기
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

#5개의 샘플에 대한 예측
print(lr.predict(test_scaled[:5]))
#5개의 샘플에 대한 예측 확률(소수점4번째에서 반올림)
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

#클래스정보
print(lr.classes_)

#선형방정식 모습
print(lr.coef_.shape, lr.intercept_.shape)

#z값 구하기
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
#소프트맥스 함수를 사용한 확률 구하기
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))