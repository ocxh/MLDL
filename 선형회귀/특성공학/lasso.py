from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
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

#변환기를 통한 특성공학
poly = PolynomialFeatures(degree=5, include_bias=False) 
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

#규제 전 표준화(특성을 표준점수로 변환)
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#라쏘 회귀(alpha값 조정을 통해 규제의 강도 조절)
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
  lasso = Lasso(alpha=alpha, max_iter=10000)
  lasso.fit(train_scaled, train_target)

  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target))

#최적의 alpha값 찾기
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

#라쏘 모델의 계수 확인(0은 사용하지 않는 것)
print(np.sum(lasso.coef_ == 0))