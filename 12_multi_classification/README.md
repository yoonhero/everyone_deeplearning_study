# 다중 분류 문제 해결하기

클래스가 2개가 아니라 3개일때, 즉 참(1)과 거짓(0) 으로 해결하는 것이 아니라, 여러개중에 어떤 것이 답인지를 예측할때를 말한다.

이항분류(binary classification) 과는 접근 방식이 조금 다르다.

### 원-핫 인코딩 (one-hot-encoding)

```
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

Y_encoded = to_categorical(Y)
```

Y 값이 숫자가 아니라 문자열이므로 숫자로 바꾸어주어야 된다. 그러므로 LabelEncoder() 함수로 숫자 형태로 바꾸어주고 활성화 함수를 적용하기 위해서는 Y 값이 0, 1로 이루어져있어야 하므로 to_categorical() 함수로 0과 1로 바꾸어준다.

## 소프트맥스

```
model = Sequential()
model.add(Dense(16, input_dim=4, activation="relu"))
model.add(Dense(3, activation="softmax"))
```

![IMG_7917](https://user-images.githubusercontent.com/57530375/135056041-f12cb71f-0b04-4a62-8cbe-001f2381c0ec.jpg)
