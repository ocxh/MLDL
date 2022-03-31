#수상한 도미를 빙어로 예측하였기 때문에 실패한 학습임
#단순히 표준점수를 통한 데이터 전처리의 예
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#도미와 빙어 데이터
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target,random_state=42)


mean = np.mean(train_input, axis=0) #평균
std = np.std(train_input, axis=0) #표준편차
train_scaled = (train_input - mean) / std #표준점수로 변환(훈련 데이터)
test_scaled = (test_input - mean) / std #표준점수로 변환(테스트 데이터)

new = ([25, 150] - mean) /std #수상한 도미 한마리 표준 점수로 변환

kn = KNeighborsClassifier()
kn.fit(train_scaled, train_target)
kn.score(test_scaled, test_target)

distances, indexes = kn.kneighbors([new]) #수상한 도미의 이웃 데이터(거리와 인덱스)
plt.scatter(train_scaled[:,0], train_scaled[:,1]) #전체 데이터
plt.scatter(new[0], new[1], marker='^') #수상한 도미
plt.scatter(train_scaled[indexes,0], train_scaled[indexes, 1], marker='D') #수상한 도미의 이웃(5)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()