# 컨볼루션 신경망 CNN (Convolutional Neural Network)

컨볼루션 신경망은 입력된 이미지에서 다시 한번 특징을 추출하기 위해 커널(슬라이딩 윈도)을 도입하는 기법입니다.

![IMG_7938](https://user-images.githubusercontent.com/57530375/135735227-db180b90-41ab-4750-85b9-57c62e7a53b6.jpg)

이렇게해서 새롭게 만들어진 층을 컨볼루션(합성곱)이라고 부른다. 컨볼루션을 만들면 입력 데이터로부터 더욱 정교한 특징을 추출할 수 있다.

```
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation="relu))
```

- kernel_size 커널의 크기를 정합니다.
- input_shape Dense 층과 같이 맨 처음층에는 입력되는 값을 알려주어야 한다. input_shape(행, 열, 색상 또는 흑백) 색상이면 3, 흑백이면 1

# 맥스 풀링

컨볼루션 한 결과가 여전히 크고 복잡하면 이를 다시 한번 축소해야한다. 이 과정을 풀링(pooling) 또는 서브 샘플링(sub sampling) 이라고 한다.
정해진 구역 안에서 최댓값을 뽑아내는 <strong>맥스 풀링(max pooling)</strong>과 평균값을 뽑아내는 <strong>평균 풀링(average pooling)</strong> 등이 있다.

![IMG_7939](https://user-images.githubusercontent.com/57530375/135735309-d0c97684-7b59-4ddd-a339-8f15feeb9b93.jpg)

=맥스풀링 예시=

```
model.add(MaxPooling2D(pool_size=2))
```

## 드롭아웃, 플래튼

노드가 많아지거나 층이 많아진다고 해서 학습이 무조건 좋아지는 것이 아니기에 과적합을 효과적으로 피해가기 위해 다양한 기법이 연구되어 왔다. 그중 가장 간단하지만 효과가 큰 기법은 바로 <strong>드롭아웃(drop out)</strong> 기법이다. 드롭아웃은 은닉층에 배치된 노드 중 일부를 임의로 껴주는 것이다.

```
model.add(Dropout(0.25))
```

Dense 함수로 만들었던 기본 층과 연결할때 맥스 풀링은 2차원 배열인 채로 다루기에 1차원 배열로 바꿔줘야하고 Flatten() 함수를 이용한다.

```
model.add(Flatten())
```

![IMG_7940](https://user-images.githubusercontent.com/57530375/135735508-7b8f0b41-aa05-4637-b007-286c08165921.jpg)
