# 모델 설계하기

## 입력층, 은닉층, 출력층

```
model = Sequential()
model.add(Dense(30, input_dim=17, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
```

딥러닝이란 퍼셉트론 위에 숨겨진 퍼셉트론 층을 차곡차곡 추가하는 형태이다. 이 층들이 Sequential() 함수를 통해 쉽게 구현된다.

Sequential() 함수를 model 로 선언하고 model.add() 라는 라인을 추가하면 새로운 층이 만들어진다.

맨마지막 층은 결과를 출력하는 '출력층' 이 된다. 나머지는 모두 '은닉층'의 역할을 한다.

```
  Dense(30, input_dim=17, activation="relu")
```

Dense() 함수를 통해서 이 층에 몇개의 노드를 만들 것인지를 숫자로 써준다. 첫번째 은닉층에 input_dim 을 적어줌으로서 첫 번째 Dense 가 은닉층 + 입력층의 역할을 겸한다.

![IMG_7889](https://user-images.githubusercontent.com/57530375/134754460-fce13815-af45-44ea-a2b2-c1853d9b3695.JPG)

## 모델 컴파일

```
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
```

앞서 지정된 모델이 효과적으로 구현될 수 있게 여러가지 환경을 설정해 주면서 컴파일하는 부분이다. 먼저 어떤 오차 함수를 사용할지를 정해야 한다.

여기서는 평균 제곱 오참 함수(mean_squared_error)

## 교차 엔트로피

예측값이 참과 거짓 둘 중 하나인 형식일 때는 binary_crossentropy 를 쓴다.

### 평균 제곱 계열

mean_squared_error - 평균 제곱 오차

- 계산: mean(square(yt - yo))

mean_absolute_error - 평균 절대 오차

- 계산: mean(abs(yt-yo))

mean_absolute_percentage_error - 평균 절대 백분율 오차

- 계산: mean(abs(yt - yo) / abs(yt))

mean_squared_logarithmic_error - 평균 제곱 로그 오차가

- 계산: mean(square((log(yo) + 1) - (log(yt) + 1)))

### 교차 엔트로피 계열

categorical_crossentropy - 범주형 교차 엔트로피

binary_crossentropy - 이항 교차 엔트로피

## 모델 실행하기

```
model.fit(X, Y, epochs=100, batch_size=10)
```

batch_size 는 샘플을 한 번에 몇 개씩 처리할지를 정하는 부분이다.
