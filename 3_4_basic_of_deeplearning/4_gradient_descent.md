# 오차 수정하기 - 경사 하강법

## 경사하강법(gradient descent)

그래프에서 오차를 비교하여 가장 작은 방향으로 이동시키는 방법을 말한다.

### 개요

경사 하강법은 기울기가 0인 즉 미분값이 0인 지점을 찾는 과정을 말한다.

### 학습률 (learning rate)

경사 하강법 시 어느만큰 이동시킬지를 정해주는 것이 바로 학습률이고 적절히 바꾸면서 최적의 학습률을 찾는 것은 중요한 최적화 과정중 하나이다.

<strong>평균제곱오차</strong>

```
1/n 시그마 (y - (ax + b))^2
```

![IMG_8476](https://user-images.githubusercontent.com/57530375/142512917-66de2050-c3cb-4aae-b48c-24ed53598247.JPG)
