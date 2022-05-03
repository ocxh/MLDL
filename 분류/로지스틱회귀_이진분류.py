import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

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

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

#클래스 예측
print(lr.predict(train_bream_smelt[:5]))
#확률
print(lr.predict_proba(train_bream_smelt[:5]))

#현재 클래스 보기
print(lr.classes_)

#계수 확인
print(lr.coef_, lr.intercept_)
#z값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
'''
z값을 구하는 세부식
z = lr.coef_[0]*무게 + lr.coef_[1]*길이 + lr.coef_[2]*대각선 + lr.coef_[3]*높이 + lr.coef_[4]*두께 + lr.intercept_ 
print( np.sum( train_bream_smelt[0]*lr.coef_ ) + lr.intercept_ )
'''
#시그모이드 함수를 통한 z값 확률화
print(expit(decisions))