'''
숫자 1, 2, 3, 4, 5가 한 그룹에 속하고, 숫자 6, 7, 8, 9, 10이 다른 한 그룹에 속한다고 할 때, K=3일 때 KNN의 분류 성능을 알아보는 문제.
이 10개의 데이터 중 랜덤하게 70%를 학습 데이터로 사용하고, 30%는 테스트 데이터로 사용하고, 테스트 데이터에 대한 KNN의 정확도를 출력(print)하시오.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = [1,2,3,4,5,6,7,8,9,10]

target = [1]*5 + [0]*5

input_arr = np.array(data)
target_arr = np.array(target)

np.random.seed(43) #seed값에 따라 정확도가 다름
index = np.arange(10)
np.random.shuffle(index) 

#훈련 데이터
train_input = input_arr[index[:7]].reshape(-1,1)
train_target = target_arr[index[:7]]
#테스트 데이터
test_input = input_arr[index[7:]].reshape(-1,1)
test_target = target_arr[index[7:]]


kn = KNeighborsClassifier()
kn = kn.fit(train_input, train_target) 
print(kn.score(test_input, test_target)) #모델 성능 측정
print(kn.predict(test_input)) #예측
print(test_target) #정답