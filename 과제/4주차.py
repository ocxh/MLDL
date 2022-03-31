'''
(1) 이전 과제3 프로그램을 다시 잘 작성하시오.

(2) np.concatenate, np.ones, np.zeros, train_test_split 함수를 사용하여 더 효과적으로 수정하시오. train_test_split 의 test_size(또는 train_size) 옵션을 적당히 사용하시오.

(3) 이 시뮬레이션을 K=3, 5일 때, 각각 1000번 반복하여 정확도들의 평균과 표준편차를 출력하시오.
'''

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = [1,2,3,4,5,6,7,8,9,10] #특성이 1개이기 때문에 하나의 리스트로

target = [1]*5 + [0]*5 #그룹1과 그룹2를 나누는 target리스트
target = np.concatenate((np.ones(5), np.zeros(5)))

sm3 = [] #k=3일 때
sm5 = [] #k=5일 때

for i in range(1000):
  #train_test_split과정(test_size=0.4)
  train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.4,stratify=target)

  train_input = np.array(train_input).reshape(-1,1)
  test_input = np.array(test_input).reshape(-1,1)

  kn = KNeighborsClassifier(3) #k를 3으로 설정
  kn = kn.fit(train_input, train_target)

  sm3.append(kn.score(test_input, test_target))

for i in range(1000):
  #train_test_split과정(test_size=0.4)
  train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.4,stratify=target)

  train_input = np.array(train_input).reshape(-1,1)
  test_input = np.array(test_input).reshape(-1,1)

  kn = KNeighborsClassifier(5) #k를 5로 설정
  kn = kn.fit(train_input, train_target)

  sm5.append(kn.score(test_input, test_target))


mean3 = np.mean(sm3, axis=0) #평균
std3 = np.std(sm3, axis=0) #표준편차

mean5 = np.mean(sm5, axis=0) #평균
std5 = np.std(sm5, axis=0) #표준편차

print("K는 3일 때 평균과 표준편차")
print(mean3, std3)
print("K는 5일 때 평균과 표준편차")
print(mean5, std5)