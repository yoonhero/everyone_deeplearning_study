# 가장 훌륭한 예측선 긋기 - 선형회귀

## 선형회귀

독립 변수 x를 사용해 종속 변수 y의 움직임을 예측하고 설명하는 작업을 말한다.

### 최소제곱법

가장 정화한 직선을 긋는 것 즉 정확한 기울기 a와 정확한 y절편의 값 b를 알아내면 된다.

<strong>최소제곱법</strong>(method of least squares)

```
a = (x-x평균)(y-y평균)의합 / (x-x평균)^2의 합
```

### 평균 제곱 오차

```
오차의 합 = 시그마(실제값 - 예측값)^2
```

```
평균 제곱 오차(MSE) = 1/n * 오차의 합
```