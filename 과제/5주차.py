"""
문제:
mglearn 라이브러리에 있는 wave 데이터 50개를 이용하여 K-최근접 이웃 방법(KNR)으로 회귀분석을 하고자 한다.
50개의 데이터를 랜덤하게 학습데이터 70%, 테스트데이터 30%로 나눈 후, K=1, 3, 5, 7, 9 중 BEST K를 선정하고, 이유를 적으시오. 
"""

import mglearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=50)

knr = KNeighborsRegressor()

k_li = [1,3,5,7,9]
cycle = 5 #각 K마다 5회씩 반복하여 측정
overfitting = 0.05 #과대적합 판단 기준

best_k = [0, 0] #[0]: k값, [1]: R^2

for k in k_li:
  knr.n_neighbors = k #K값 설정
  for _ in range(cycle):
    train_input, test_input, train_target, test_target = train_test_split(X, y, train_size=0.7)
    knr.fit(train_input, train_target)

    #과대적합, 과소적합에 해당되지 않으면서 테스트 세트의 결정 계수가 가장 큰 K값 찾기 
    train_R2 = knr.score(train_input,  train_target)
    test_R2 = knr.score(test_input,  test_target)
    diff = train_R2-test_R2
    if diff >= 0 and diff < overfitting and best_k[1] < test_R2:
      best_k[0] = k; best_k[1] = test_R2    

print(best_k)

"""
최종적으로 구한 최적의 K값은 과대적합과 과소적합에 해당되지 않으며,
테스트 세트의 결정 계수가 가장 크기 때문에 BEST K값 에서 가장 성능이 높다고 판단하였습니다.
"""