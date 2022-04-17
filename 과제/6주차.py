import mglearn
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#샘플 데이터1
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

overfitting = 0.05 #과대적합 측정 값

#모델별 R2
knr_best = 0 
linear_best = 0
poly_best = 0

#샘플 데이터1에서 100번의 테스트로 각 모델별 성능 측정
for _ in range(100):
  train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, train_size=0.7)
  train_input = train_input.reshape(-1, 1)
  test_input = test_input.reshape(-1, 1)

  #KNR
  knr = KNeighborsRegressor()
  knr.n_neighbors = 3 #K값 설정
  knr.fit(train_input, train_target)

  train_R2 = knr.score(train_input,  train_target)
  test_R2 = knr.score(test_input,  test_target)
  diff = train_R2-test_R2

  if diff >= 0 and diff < overfitting:
    knr_best = max(knr_best, test_R2)

  #선형회귀
  lr = LinearRegression()
  lr.fit(train_input, train_target)

  train_R2 = lr.score(train_input,  train_target)
  test_R2 = lr.score(test_input,  test_target)
  diff = train_R2-test_R2

  if diff >= 0 and diff < overfitting:
    linear_best = max(linear_best, test_R2)

  #2차 다항회귀
  train_poly = np.column_stack((train_input ** 2, train_input))
  test_poly = np.column_stack((test_input **2, test_input))

  lr2 = LinearRegression()
  lr2.fit(train_poly, train_target)

  train_R2 = lr2.score(train_poly,  train_target)
  test_R2 = lr2.score(test_poly,  test_target)
  diff = train_R2-test_R2

  if overfitting > diff > 0:
    poly_best = max(poly_best, test_R2)

best1 = [knr_best, linear_best, poly_best]

#샘플 데이터2
X, y = mglearn.datasets.make_wave(n_samples=100)
knr_best = 0 #R^2
linear_best = 0
poly_best = 0

#샘플 데이터2에서 100번의 테스트로 각 모델별 성능 측정
for _ in range(100):
  train_input, test_input, train_target, test_target = train_test_split(X, y, train_size=0.7)
  train_input = train_input.reshape(-1, 1)
  test_input = test_input.reshape(-1, 1)

  #KNR
  knr = KNeighborsRegressor()
  knr.n_neighbors = 3 #K값 설정
  knr.fit(train_input, train_target)

  train_R2 = knr.score(train_input,  train_target)
  test_R2 = knr.score(test_input,  test_target)
  diff = train_R2-test_R2

  if diff >= 0 and diff < overfitting:
    knr_best = max(knr_best, test_R2)

  #선형회귀
  lr = LinearRegression()
  lr.fit(train_input, train_target)

  train_R2 = lr.score(train_input,  train_target)
  test_R2 = lr.score(test_input,  test_target)
  diff = train_R2-test_R2

  if diff >= 0 and diff < overfitting:
    linear_best = max(linear_best, test_R2)

  #2차 다항회귀
  train_poly = np.column_stack((train_input ** 2, train_input))
  test_poly = np.column_stack((test_input **2, test_input))

  lr2 = LinearRegression()
  lr2.fit(train_poly, train_target)

  train_R2 = lr2.score(train_poly,  train_target)
  test_R2 = lr2.score(test_poly,  test_target)
  diff = train_R2-test_R2

  if overfitting > diff > 0:
    poly_best = max(poly_best, test_R2)

best2 = [knr_best, linear_best, poly_best]

print("샘플 데이터1, 2에서 각 모델의 성능: ", best1, best2)

best = [x+y for x,y in zip(best1, best2)]
print("각 샘플에서의 성능을 모델별로 합친 값: ",best)

print("==최고의 모델==")
if sum(best) == 0:
  print("재시도")
elif max(best) == best[0]:
  print("KNR")
elif max(best) == best[1]:
  print("Linear")
else:
  print("Poly")