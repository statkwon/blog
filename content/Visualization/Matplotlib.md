---
title: "Matplotlib"
date: 2021-02-11
weight: 1
TableOfContents: true
---

## 1. Several Options
{{< highlight py3 >}}
import matplotlib as mpl ; mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt ; plt.rcParams['font.family'] = 'AppleGothic'
{{< / highlight >}}

윈도우 유저라면 첫 번째 줄은 입력할 필요가 없다. 맥의 경우 '-' 기호가 깨져서 나타나는 경우를 해결하기 위해 첫 번째 줄의 코드를 사용할 수 있다. matplotlib에서 한글을 사용해야 할 경우, 두 번째 줄의 font.family를 지정하는 코드를 반드시 입력해야한다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.1.png">}}

빈 도화지를 만드는 방법이다. 이 상태를 기본으로 여러가지 변형을 시도해볼 것이다.

### 1) Figure Size
{{< highlight py3 >}}
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.2.png">}}

figsize라는 명령어를 통해 plot의 사이즈를 조절할 수 있다.

### 2) x & y ticks
{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
ax.set_xticks(range(10))
ax.set_yticks(range(10))
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.3.png">}}

ax.set_xticks(yticks) 명령어를 사용하여 x축과 y축의 눈금의 갯수를 조절할 수 있다. 이때 눈금의 라벨을 별도로 지정하지 않는다면 자동으로 눈금의 인덱스를 따른다.

### 3) x & y ticklabels
{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
ax.set_yticklabels(['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ'])
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.4.png">}}

ax.set_xticklabels(yticklabels) 명령어를 사용하여 눈금의 라벨을 지정할 수 있다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], rotation=90)
ax.set_yticklabels(['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ'])
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.5.png">}}

ax.set_xticklabels(yticklabels) 안에 rotation값을 추가하면 원하는 각도만큼 눈금의 라벨을 회전시킬 수 있다. 라벨의 길이가 길어 서로 겹치는 경우에 사용하면 좋다.

### 4) x & y limits
{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
ax.set_xlim([-5, 5])
ax.set_ylim([0, 10])
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.6.png">}}

ax.set_xlim(ylim) 명령어를 사용하여 눈금을 조절하는 것도 가능하다. 눈금의 양 끝값을 지정하면 자동으로 배열이 형성된다.

### 5) Grid
{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
ax.grid(True)
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.7.png">}}

ax.grid 명령어를 사용하여 격자 배경을 설정할 수 있다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
ax.grid(True, color='gray', linestyle='--', linewidth=2, axis='y')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.8.png">}}

추가적으로 격자의 색상, 선 스타일, 선 굵기, 방향 설정이 가능하다.

### 6) x & y labels and title
{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
ax.set_xlabel('x축')
ax.set_ylabel('y축')
ax.set_title('제목')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.9.png">}}

ax.set_xlabel(ylabel) 명령어로 x축과 y축의 라벨을, ax.set_title 명령어로 plot의 제목을 설정할 수 있다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot()
ax.set_xlabel('x축', fontsize=10)
ax.set_ylabel('y축', fontsize=15)
ax.set_title('제목', fontsize=20)
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.10.png">}}

fontsize를 추가하여 글자 크기를 조절할 수 있다.

### 7) Multiple Plots
{{< highlight py3 >}}
fig, ax = plt.subplots(1, 2, figsize=(8, 5))
ax[0].plot()
ax[1].plot()
ax[0].set_title('첫번째')
ax[1].set_title('두번째')
fig.suptitle('제목', fontsize=20)
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.11.png">}}

plt.subplots(x, y)의 형식으로 여러 개의 plot을 동시에 나타낼 수 있다. 이때 주의할 점은 ax[0].plot, ax[0].set_title과 같이 ax의 인덱스를 지정한 후에 각각의 plot에 대한 명령어를 입력해야 한다는 것이다. 또한, 여러 개의 plot을 그렸을 경우에 전체 제목을 추가하려면 fig.suptitle이라는 명령어를 사용해야 한다.

{{< highlight py3 >}}
fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
ax[0, 0].plot()
ax[0, 1].plot()
ax[1, 0].plot()
ax[1, 1].plot()
ax[0, 0].set_title('첫번째')
ax[0, 1].set_title('두번째')
ax[1, 0].set_title('세번째')
ax[1, 1].set_title('네번째')
fig.suptitle('제목', fontsize=20)
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.12.png">}}

plot의 갯수가 늘어나면 화면의 공간이 협소해지기 때문에 sharex와 sharey를 True로 설정함으로써 plot 간에 눈금을 공유하여 여백을 확보할 수 있다.

### 8) Line Style

{{< highlight py3 >}}
import numpy as np
{{< /highlight >}}

{{< highlight py3 >}}
x = np.linspace(0, 10, 100)
{{< /highlight >}}

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), linestyle='--', color='r')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.13.png">}}

ax.plot() 안에 x값과 y값을 입력하면 자동으로 Line Graph를 그려준다. 이때 linestyle, marker, color 등의 옵션을 추가하여 선의 형태를 조절할 수 있다.

### 9) Legend
{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), linestyle='--', color='r', label='red')
ax.plot(x, np.sin(x+1), linestyle='-', color='b', label='blue')
ax.legend()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.14.png">}}

여러 개의 plot을 한 화면에 나타내는 경우 범례를 추가하는 것이 좋다. 우선 ax.plot() 안에 label을 설정하고 ax.legend 명령어를 입력하면 설정한 label에 따라 범례가 나타난다. 범례의 위치를 직접 지정하지 않으면 자동으로 그래프와 겹치지 않는 최적의 위치로 설정된다.

### 10) twinx
{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label='red')
ax2 = ax.twinx()
ax2.plot(x, np.exp(x), color='r', label='blue')
fig.legend(bbox_to_anchor=(0.27, 0.35))
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.15.png">}}

도메인의 스케일이 다른 그래프들을 한 화면에 나타내는 경우 대부분 한 쪽의 그래프가 일그러지게 된다. 이때 ax.twinx 명령어를 사용하면 서로 다른 도메인을 좌우측에 나누어 사용함으로써 그래프의 형태를 보존하는 것이 가능하다. 주의할 점은 twinx를 사용한 경우 범례를 추가할 때 fig.legend 명령어를 사용해야 한다는 것이다. bbox_to_anchor에 (x, y) 값을 입력해서 범례의 위치를 조정할 수 있다.

### 11) Annotation
{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot(x, np.sin(x))
ax.annotate('max', xy =(1.5, 1), xytext =(2.5, 1.3), arrowprops = {'arrowstyle':'->', 'color':'black'})
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.16.png">}}

plot에 코멘트를 추가하고 싶다면 ax.annotate 명령어를 사용할 수 있다. xy로 화살표의 위치를, xytext로 텍스트의 위치를, arrowprops로 화살표의 모양과 색상을 조절한다.

## 2. Various Plots

위에서 살펴본 옵션들을 복습함과 동시에 다양한 형태의 plot을 그리는 방법을 알아보도록 하자.

### 1. Line Graph

{{< highlight py3 >}}
import pandas as pd
import requests
from bs4 import BeautifulSoup
{{< /highlight >}}

{{< highlight py3 >}}
def stock_price(code):
    url = "https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&count=100&requestType=0".format(code)
    get_url = requests.get(url)
    bs_url = BeautifulSoup(get_url.content, "html.parser")

    item = bs_url.select('item')
    columns = ['Date', 'Open' ,'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame([], columns = columns, index = range(len(item)))
    
    for i in range(len(item)):
        df.iloc[i] = str(item[i]['data']).split('|')
    
    df.index = pd.to_datetime(df['Date'])
    
    return df.drop('Date', axis=1).astype(float)

kakao = stock_price('035720')
kakao.head()
{{< /highlight >}}

Date | Open | High | Low | Close | Volume
-- | -- | -- | -- | -- | --
2020-02-28 | 174500.0 | 176500.0 | 170500.0 | 172000.0 | 1022917.0
2020-03-02 | 174500.0 | 176500.0 | 169500.0 | 175000.0 | 793495.0
2020-03-03 | 180500.0 | 181000.0 | 175000.0 | 175000.0 | 629017.0
2020-03-04 | 174000.0 | 181000.0 | 174000.0 | 179500.0 | 709188.0
2020-03-05 | 182500.0 | 183000.0 | 176500.0 | 179500.0 | 821744.0

실제 데이터를 사용하여 plot을 그려보기 위해 카카오 주가 데이터를 크롤링하였다. 위와 같은 형태의 데이터프레임을 가지고 그래프를 그려보자.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot(kakao.index, kakao['Open'])
ax.grid(True)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Kakao Stock Price')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.17.png">}}

앞서 언급했듯이, ax.plot() 안에 x, y값을 추가하면 자동으로 Line Graph가 그려진다. 일반적으로 시계열 데이터를 사용하는 경우 Line Graph로 나타내게 된다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot(kakao.index, kakao['Open'], label='Open')
ax.plot(kakao.index, kakao['Close'], label='Close')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Kakao Stock Price')
ax.legend()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.18.png">}}

이번에는 시가와 종가를 동시에 그려보았다. 쉽게 선을 구분하기 위해 범례를 추가하였다.

{{< highlight py3 >}}
kakao_diff = kakao.diff().dropna()
fig, ax = plt.subplots()
ax.plot(kakao_diff.index, kakao_diff['Open'], marker='.')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Kakao Stock Price (diff)')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.19.png">}}

시계열 분석을 하는 경우, 데이터를 차분한 그래프를 그려 정상성 조건을 확인하게 된다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.plot(kakao.index, kakao['Open'], label='default')
ax2 = ax.twinx()
ax2.plot(kakao_diff.index, kakao_diff['Open'], color='r', label='diff')
fig.legend(bbox_to_anchor=(0.41, 0.83))
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.20.png">}}

원 그래프와 차분한 그래프를 동시에 나타내는 경우 twinx를 사용하여 두 그래프 모두 명확한 형태로 확인하는 것이 가능하다.

{{< highlight py3 >}}
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,5))
ax[0, 0].plot(kakao.index, kakao['Open'], label='Open')
ax[0, 1].plot(kakao.index, kakao['High'], color='r', label='High')
ax[1, 0].plot(kakao.index, kakao['Low'], color='g', label='Low')
ax[1, 1].plot(kakao.index, kakao['Close'], color='y', label='Close')
ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.21.png">}}

### 2. Pi-Chart

{{< highlight py3 >}}
covid = pd.read_html('http://ncov.mohw.go.kr/bdBoardList_Real.do?brdId=1&brdGubun=13&ncvContSeq=&contSeq=&board_id=&gubun=', encoding='utf8')
covid = covid[0]
covid.columns = ['시도명', '합계', '해외유입', '국내발생', '확진환자', '격리중', '격리해제', '사망자', '발생률']
covid = covid.drop([0, 18])
covid = covid.set_index('시도명')
covid.head()
{{< /highlight >}}

시도명 | 합계 | 해외유입 | 국내발생 | 확진환자 | 격리중 | 격리해제 | 사망자 | 발생률
-- | -- | -- | -- | -- | -- | -- | -- | --
서울 |16 | 0 | 16 | 1514 | 157 | 1346 | 11 | 15.55
부산 | 0 | 0 | 0 | 157 | 5 | 149 | 3 | 4.60
대구 | 1 | 1 | 0 | 6937 | 17 | 6730 | 190 | 284.71
인천 | 5 | 1 | 4 | 377 | 26 | 349 | 2 | 12.75
광주 | 4 | 3 | 1 | 191 | 88 | 101 | 2 | 13.11

이번에는 코로나 바이러스 확진자 데이터를 크롤링하였다.

{{< highlight py3 >}}
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(covid.iloc[:5, 3], labels=covid.index[:5], autopct='%1.1f%%', explode=(0, 0, 0.1, 0, 0))
ax.set_title('확진환자 비율')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.22.png">}}

ax.pie 명령어를 사용하면 Pi-Chart를 그릴 수 있다. 사용할 데이터를 필수적으로 입력하고, 그 외 labels, autopct, explode 등의 추가적인 옵션을 통해 그래프의 형태를 조절할 수 있다. labels는 각 파트의 라벨을 설정, autopct는 각 파트 별 %를 소수 몇 째 자리까지 나타낼지를 설정, explode는 특정 파트를 분리하기 위해 사용하는 옵션이다.

### 3. Bar-Chart

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.bar(covid.index, covid['확진환자'])
ax.set_xticklabels(covid.index, rotation=90)
ax.set_xlabel('시도명')
ax.set_ylabel('확진환자')
ax.set_title('코로나바이러스 국내 현황')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.23.png">}}

세 번째는 막대 그래프이다. ax.bar 명령어를 사용하여 그릴 수 있다. x축의 라벨이 되는 x값과 높이가 되는 y값 모두 필수적으로 입력해야 한다. set_xticklabels의 rotation=90을 사용하여 시도명을 회전시켜 가독성을 높혀보았다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.bar(covid.index, covid['격리해제'], label='격리해제')
ax.bar(covid.index, covid['격리중'], bottom=covid['격리해제'], label='격리중')
ax.set_xticklabels(covid.index, rotation=90)
ax.set_xlabel('시도명')
ax.set_ylabel('확진환자')
ax.set_title('코로나바이러스 국내 현황')
ax.legend()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.24.png">}}

여러 개의 데이터를 누적해서 표현하고 싶은 경우, ax.bar() 안에 bottom이라는 옵션을 추가하면 된다. bottom에 할당된 기존 데이터가 새로운 데이터보다 아래에 위치하게 된다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.bar(covid.index, covid['격리해제'], label='격리해제')
ax.bar(covid.index, covid['격리중'], bottom=covid['격리해제'], label='격리중')
ax.bar(covid.index, covid['사망자'], bottom=covid['격리해제'] + covid['격리중'], label='사망자')
ax.set_xticklabels(covid.index, rotation=90)
ax.set_xlabel('시도명')
ax.set_ylabel('확진환자')
ax.set_title('코로나바이러스 국내 현황')
ax.legend()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.25.png">}}

몇 개의 데이터든 계속해서 누적하여 표현이 가능하다. 사용하는 데이터가 세 개 이상인 경우, bottom 옵션에 기존 데이터를 a + b + ~ 형식으로 입력해주면 된다. 식의 순서대로 아래서부터 쌓아 올라가게 된다. 즉, a + b 라면 a가 가장 아래, b가 그 위, c가 가장 위에 위치한다.

### 4. Histogram

{{< highlight py3 >}}
from sklearn.datasets import load_iris
{{< /highlight >}}

{{< highlight py3 >}}
load_iris = load_iris()
iris = pd.DataFrame(data=load_iris.data, columns=load_iris.feature_names)
iris['species'] = load_iris.target
setosa = iris[iris['species']==0]
versicolor = iris[iris['species']==1]
virginica = iris[iris['species']==2]
{{< /highlight >}}

이번에는 붓꽃 데이터를 활용해보자.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.hist(iris['sepal length (cm)'], bins=15)
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('count')
ax.set_title('Distribution of Sepal Length')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.26.png">}}

히스토그램을 그리고 싶은 경우, ax.hist 명령어를 사용하면 된다. ax.pie와 마찬가지로 사용하고 싶은 데이터만 필수적으로 입력해주면 된다. ax.hist() 안에 bins 값을 추가함으로써 x축을 구분하는 구간의 갯수를 조절할 수 있다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.hist(iris['sepal length (cm)'], histtype='step', label='sepal length')
ax.hist(iris['petal length (cm)'], histtype='step', label='petal length')
ax.set_xlabel('sepal length & petal length (cm)')
ax.set_ylabel('count')
ax.set_title('Distribution of Sepal Length & Petal Length')
ax.legend()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.27.png">}}

여러 개의 히스토그램을 겹쳐서 그리고 싶은 경우, 겹쳐진 부분을 확인하기 위해서 histtype을 'step'으로 설정함으로써 투명한 그래프를 그리는 것이 가능하다.

### 5. Violin-Plot

{{< highlight py3 >}}
fig, ax = plt.subplots(1, 3, sharey=True)
ax[0].violinplot(setosa['sepal length (cm)'])
ax[1].violinplot(versicolor['sepal length (cm)'])
ax[2].violinplot(virginica['sepal length (cm)'])
fig.suptitle('Mean Value of Sepal Length')
ax[0].set_xlabel('setosa')
ax[1].set_xlabel('versicolor')
ax[2].set_xlabel('virginica')
ax[0].set_ylabel('Sepal Length')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.28.png">}}

Violin-Plot은 Boxplot과 Distribution-Plot을 섞어 놓은 형태의 plot이다. ax.violinplot이라는 명령어를 사용하여 그릴 수 있으며, 사용할 데이터만 필수적으로 입력해주면 된다. sharey=True를 사용하여 종 간의 분포 비교가 가능하도록 하였다.

### 6. Boxplot

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.boxplot([setosa['sepal length (cm)'], versicolor['sepal length (cm)'], virginica['sepal length (cm)']])
ax.set_xticklabels(['setosa', 'versicolor', 'virginica'])
ax.set_ylabel('sepal length (cm)')
ax.set_title('Box-Plot of Sepal Length')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.29.png">}}

boxplot은 ax.boxplot 명령어로 그릴 수 있다. 여러 개의 boxplot을 동시에 그리고 싶은 경우, 리스트 안에 여러 데이터를 포함하여 입력해주면 된다.

### 7. Scatter-Plot

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.scatter(iris['sepal length (cm)'], iris['sepal width (cm)'])
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('sepal width (cm)')
ax.set_title('Sepal Length & Sepal Width')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.30.png">}}

산점도는 ax.scatter 명령어로 그릴 수 있다. x값과 y값으로 사용할 데이터 모두 필수적으로 입력해야한다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.scatter(iris['sepal length (cm)'], iris['sepal width (cm)'], c=iris['species'], s=iris['petal length (cm)']*10, alpha=0.3)
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('sepal width (cm)')
ax.set_title('Sepal Length & Sepal Width')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.31.png">}}

c와 s에 각각 특정 데이터를 할당하면 해당 데이터에 따라 점의 색상과 크기를 다르게 할 수 있다. s에 할당된 값에 10을 곱해준 것은 크기 구분을 조금 더 명확하게 하기 위함이다. alpha는 투명도를 조절하는 옵션이다.

## 3. Change the Style

{{< highlight py3 >}}
plt.style.use('ggplot')
{{< /highlight >}}

plt.style.use라는 명령어를 사용하면 matplotlib의 스타일을 변경할 수 있다. R에서 자주 사용하는 'ggplot' 스타일로 변경해보자.

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.bar(covid.index, covid['격리해제'], label='격리해제')
ax.bar(covid.index, covid['격리중'], bottom=covid['격리해제'], label='격리중')
ax.bar(covid.index, covid['사망자'], bottom=covid['격리해제'] + covid['격리중'], label='사망자')
ax.set_xticklabels(covid.index, rotation=90)
ax.set_xlabel('시도명')
ax.set_ylabel('확진환자')
ax.set_title('코로나바이러스 국내 현황')
ax.legend()
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.32.png">}}

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.hist(iris['sepal length (cm)'], bins=15)
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('count')
ax.set_title('Distribution of Sepal Length')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.33.png">}}

{{< highlight py3 >}}
fig, ax = plt.subplots()
ax.scatter(iris['sepal length (cm)'], iris['sepal width (cm)'], c=iris['species'], s=iris['petal length (cm)']*10)
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('sepal width (cm)')
ax.set_title('Sepal Length & Sepal Width')
plt.show()
{{< /highlight >}}
{{<figure src="/plt_1.34.png">}}

{{< highlight py3 >}}
print(plt.style.available)
{{< /highlight >}}

'ggplot' 이외에도 다양한 스타일을 사용할 수 있다. plt.style.available이라는 명령어를 사용하여 사용 가능한 스타일의 목록을 확인할 수 있다.

## 4. Quiz

아래의 코로나 발생 현황 사이트에 게재되어 있는 그래프들을 직접 만들어 보자.

[코로나바이러스감염증-19(COVID-19)](http://ncov.mohw.go.kr/bdBoardList_Real.do?brdId=1&brdGubun=11&ncvContSeq=&contSeq=&board_id=&gubun=)

quiz1.csv와 quiz2.csv는 [ESC 시각화 스터디 깃헙 저장소](https://github.com/statkwon/ESC-Visualization)에서 다운로드할 수 있다.

### Quiz 1

{{<figure src="/plt_1.35.png">}}

{{<rawhtml>}}
<details>
<summary>Answer</summary>

{{< highlight py3 >}}
quiz1 = pd.read_csv('quiz1.csv')
quiz1 = quiz1.set_index(quiz1['일'])
quiz1 = quiz1.drop(['일'], axis=1)
{{< /highlight >}}

{{< highlight py3 >}}
fig, ax = plt.subplots(figsize=(10,6))
ax.grid(axis='y', color='grey',zorder=0)
ax.bar(quiz1.index, quiz1['국내발생'], width=0.4, bottom=quiz1['해외유입'], label='국내발생', color='mediumblue',zorder=3)
ax.bar(quiz1.index, quiz1['해외유입'], width=0.4, label='해외유입',color='mediumvioletred',zorder=3)
ax.set_ylim(0,70)
ax.set_ylabel('(명)')
fig.suptitle('감염경로구분에 따른 신규확진자 현황', fontsize=17)
ax.set_xticklabels("07-%02d" %i for i in range(14,22))
ax.legend(bbox_to_anchor=(0.55, -0.08))
plt.show()
{{< /highlight >}}

</details>
{{</rawhtml>}}

### Quiz 2

{{<figure src="/plt_1.36.png">}}

{{<rawhtml>}}
<details>
<summary>Answer</summary>

{{< highlight py3 >}}
quiz2 = pd.read_csv('quiz2.csv')
quiz2 = quiz2.set_index(quiz2['일'])
quiz2 = quiz2.drop(['일'], axis=1)
{{< /highlight >}}

{{< highlight py3 >}}
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(True, axis='y', zorder=0)
ax.bar(quiz2.index, quiz2['누적 확진환자'], label='누적 확진환자', width=0.5, color='royalblue', zorder=3)
ax.set_ylabel('누적 확진환자(명)', color='royalblue')
ax2 = ax.twinx()
ax2.plot(quiz2.index, quiz2['일 확진환자'], color='orange', linewidth=3, marker='o', label='일 확진환자', mfc='orange', mec='white', mew=2, ms=10)
ax2.set_yticks([i * 10 for i in range(10)])
ax2.set_ylabel('일 확진환자(명)', color='orange')
fig.suptitle('일일 및 누적 확진환자 추세', fontsize=17)
fig.legend(bbox_to_anchor=(0.55, 0.09))
plt.show()
{{< /highlight >}}

</details>
{{</rawhtml>}}