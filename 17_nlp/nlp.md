# 자연어 처리 Natural Language Processing

자연어란 우리가 평소에 말하는 음성이나 텍스트를 말한다. 즉 자연어 처리는 이러한 음성이나 텍스트를 컴퓨터가 인식하고 처리하는 것이다.

## 텍스트의 토큰화

먼저해야 할 일은 텍스트를 잘게 나누는 것이다. 입력할 텍스트가 준비되면 이를 단어별, 문장별, 형태소별로 나눌수 있고 이 단위를 토큰이라고 부른다. 이렇게 잘게 나누는 과정을 토큰화라고 부른다.

```
from tensorflow.keras.preprocessing.text import text_to_word_sequence

text = "해보지 않으면 해낼 수 없다."

result = text_to_word_sequence(text)

result
```

<strong>결과: ['해보지', '않으면', '해낼', '수', '없다'] </strong>

이렇게 단어 단위로 쪼개고 나면 이를 이용해서 단어가 몇번이나 중복해서 쓰였는지 알수있다. 빈도 수를 알면 텍스트에서 중요한 역할을 하는 단어를 파악할 수 있어서 텍스트를 단어 단위로 쪼개는 것은 가장 많이 쓰이는 잔처리 과정이다.

### Bag-Of-Words

단어의 가방이라는 말 뜻 그대로, 같은 단어끼리 따로따로 가방에 담은 뒤 각 가방에 몇개의 단어가 들어있는지를 세는 기법이다.

```
from tensorflow.keras.preprocessing.text import Tokenizer

docs = ["안녕하세요? 반가워요!!", "안녕히계세요... 내일 봐요", "잘지내나요?"]

token = Tokenizer()
token.fit_on_texts(docs)

token.word_counts
```

<strong>OrderedDict([('안녕하세요', 1),
('반가워요', 1),
('안녕히계세요', 1),
('내일', 1),
('봐요', 1),
('잘지내나요', 1)])</strong>

## 단어의 원-핫 인코딩

문장을 컴퓨터가 알아들을 수 있게 토큰화하고 단어의 빈도 수를 확인해 보았다. 하지만 빈도만 가지고는 해당 단어가 문장의 어디에서 왔는지 순서는 어떠했는지 등에 정보를 얻을 수 없다.

단어가 문장의 다른 요소와 어떤 관계를 가지고 있는지를 알아보는 방법이 필요하고 이러한 기법중 가장 기본적인 방법으로 <strong>원-핫 인코딩</strong> 사용된다.

```
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

text = "오랫동안 꿈꾸는 이는 그 꿈을 닮아간다"

token = Tokenizer()

token.fit_on_texts([text])
# {'오랫동안': 1, '꿈꾸는': 2, '이는': 3, '그': 4, '꿈을': 5, '닮아간다': 6}

x = token.texts_to_sequences([text])
# [[1, 2, 3, 4, 5, 6]]

# 인덱스 수에 하나를 추가해서 원-핫 인코딩 배열 만들기

word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes=word_size)
```

## 단어 임베딩

원-핫 인코딩을 그대로 사용하면 벡터의 길이가 너무 길어진다는 단점이 있다. 공간적 낭비를 해결하기 위해 등장한 것이 단어 임베딩이다.

![IMG_7966](https://user-images.githubusercontent.com/57530375/135779109-f606bd12-5065-477d-aa25-bdf2fb099e72.jpg)

```
from keras.layers import Embedding

model = Sequential()
model.add(Embedding(16, 4))
```
