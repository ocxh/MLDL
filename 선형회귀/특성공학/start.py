from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

#pandas를 사용하여 csv파일(농어 데이터: length, weight, width)을 읽은 후 numpy배열로 변환
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()

#타깃 데이터
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

#훈련세트와 테스트세트 나누기
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

#변환기를 통한 특성공학(기본)
poly = PolynomialFeatures(include_bias=False) #선형모델에서는 자동으로 절편을 추가하므로 False로 설정
poly.fit(train_input)
train_poly = poly.transform(train_input)
#print(train_poly.shape) #배열의 크기
#poly.get_feature_names_out() #각 특성이 만들어진 방법 출력
test_poly = poly.transform(test_input)

#다중회귀 모델 훈련하기
lr = LinearRegression()
lr.fit(train_poly, train_target)
print("[train_score]", lr.score(train_poly, train_target))
print("[test_score]", lr.score(test_poly, test_target))

#변환기를 통한 특성공학(degree 조정)
poly = PolynomialFeatures(degree=5, include_bias=False) #선형모델에서는 자동으로 절편을 추가하므로 False로 설정
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

#다중회귀 모델 훈련하기(degree조정 후)
lr = LinearRegression()
lr.fit(train_poly, train_target)
print("===degree 조정 후===")
print("[train_score]", lr.score(train_poly, train_target))
print("[test_score]", lr.score(test_poly, test_target))