---
title: "Seaborn"
date: 2021-02-11
weight: 2
TableOfContents: true
---

## 0. Import Seaborn and Datasets

{{< highlight py3 >}}
import pandas as pd
import matplotlib.pyplot as plt ; plt.rcParams['font.family']='AppleGothic' ; plt.rc('axes', unicode_minus=False)
import seaborn as sns
{{< / highlight >}}

Seaborn을 사용하기 위해서는 matplotlib을 먼저 import해야 한다.  두 번째 줄의 옵션들은 [1. Matplotlib]에서 설명했던 것과 같이 '-' 기호나 한글의 깨짐 현상을 방지하기 위함이다.

{{< highlight py3 >}}
epl = pd.read_csv('epl.csv')
epl.head()
{{< / highlight >}}

| name | club | age | position | position_cat | ... | new_foreign | age_cat | club_id | big_club | new_signing
| - | - | - | - | - | - | - | - | - | - | - |
| Alexis Sanchez | Arsenal | 28 | LW | 1 | ... | 0 | 4 | 1 | 1 | 0 |
| Mesut Ozil | Arsenal | 28 | AM | 1 | ... | 0 | 4 | 1 | 1 | 0 |
| Petr Cech | Arsenal | 35 | GK | 4 | ... | 0 | 6 | 1 | 1 | 0 |
| Theo Walcott | Arsenal | 28 | RW | 1 | ... | 0 | 4 | 1 | 1 | 0 |
| Laurent Koscielny | Arsenal | 31 | CB | 3 | ... | 0 | 4 | 1 | 1 | 0 |

{{< highlight py3 >}}
epl2 = pd.read_csv('epl2.csv')
epl2 = epl2[(epl2['tournament']=='Premier League') & (epl2['season'].isin(['2018/2019', '2019/2020']))].reset_index(drop=True)
epl2.head()
{{< / highlight >}}

| game_date | country | tournament | season | home_field | ... | away_team | home_team_score | away_team_score | home_team_score_extra_time | away_team_score_extra_time |
| - | - | - | - | - | - | - | - | - | - | - |
| 2018-08-10 | England | Premier League | 2018/2019 | True | ... | Leicester | 2 | 1 | NaN | NaN |
| 2018-08-11 | England | Premier League | 2018/2019 | True | ... | Tottenham | 1 | 2 | NaN | NaN |
| 2018-08-11 | England | Premier League | 2018/2019 | True | ... | Brighton | 2 | 0 | NaN | NaN |
| 2018-08-11 | England | Premier League | 2018/2019 | True | ... | Chelsea | 0 | 3 | NaN | NaN |
| 2018-08-11 | England | Premier League | 2018/2019 | True | ... | Crystal Palace | 0 | 2 | NaN | NaN |

위의 데이터는 Kaggle에서 가져온 데이터이다. 잉글랜드 프리미어리그의 선수 및 경기 데이터를 가지고 시각화를 해보자.

## 1. Relplot

### 1. Scatterplot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.scatterplot(x='page_views', y='market_value', data=epl)
ax.set_title('Page Views & Market Value')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.1.png">}}

첫 번째는 산점도이다. Seaborn 역시 matplotlib을 기반으로 하기 때문에, 이전 장에서와 같이 fig, ax = plt.subplots()로 코드를 시작한다. 이후 sns.scatterplot이라는 명령어를 사용하여 산점도를 그릴 수 있다. x와 y에 각각의 변수를 입력하고, data에는 출처가 되는 데이터를 입력하면 된다. ax.set_title 등 matplotlib에서 사용하던 옵션들을 전부 적용 가능하다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.scatterplot(x='page_views', y='market_value', data=epl, color='r', marker='+')
ax.set_title('Page Views & Market Value')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.2.png">}}

matplotlib에서와 마찬가지로 color, marker 등의 옵션으로 점의 색상과 모양을 조절할 수 있다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.scatterplot(x='page_views', y='market_value', data=epl, hue='big_club', style='big_club')
ax.set_title('Page Views & Market Value')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.3.png">}}

추가적으로 hue와 style에 특정 변수를 입력함으로써 해당 변수에 따라 점의 색상과 모양을 구분할 수 있다.

### 2. Lineplot

{{< highlight py3 >}}
MU = epl2[epl2['home_team']=='Manchester Utd'].reset_index(drop=True)
{{< / highlight >}}

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.lineplot(x='game_date', y='home_team_score', data=MU)
ax.set_xticklabels(labels=epl2['game_date'], rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.4.png">}}

두 번째는 꺾은선그래프이다. 산점도와 마찬가지로 x, y 변수와 데이터 출처를 필수적으로 입력해야 한다. matplotlib에서와 마찬가지로 color, linewidth 등의 옵션을 사용하여 선의 색상과 모양을 조절할 수 있다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.lineplot(x='game_date', y='home_team_score', data=MU, label='gain')
sns.lineplot(x='game_date', y='away_team_score', data=MU, label='loss')
ax.set_xticklabels(labels=epl2['game_date'], rotation=90)
ax.set_title('Manchester United')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.5.png">}}

Seaborn을 사용하는 경우에는 각각의 그래프에 label을 추가하였을 때 ax.legend 명령어를 사용하지 않아도 자동으로 범례가 생성된다.

### 3. Relplot

relplot은 scatterplot과 lineplot을 모두 포함하는 개념의 명령어이다. sns.relplot() 안에 kind='scatter' 또는 'line'으로 입력하여 각각의 plot을 그릴 수 있다. relplot은 Facetgrid 형태를 갖기 때문에 fig, ax = plt.subplots()로 코드를 시작하지 않고 바로 sns.relplot 명령어를 사용해야 한다. relplot은 scatterplot과 lineplot과 비교했을 때 여러 개의 그래프를 동시에 그리기 쉽다는 장점을 갖는다.

{{< highlight py3 >}}
sns.relplot(x='page_views', y='market_value', data=epl, kind='scatter', hue='big_club')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.6.png">}}

kind 옵션을 필수적으로 입력해야 한다는 점을 제외하면 scatterplot이나 lineplot과 동일한 구조를 갖는다.

{{< highlight py3 >}}
sns.relplot(x='page_views', y='market_value', data=epl, kind='scatter', hue='region', col='big_club', palette='RdBu')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.7.png">}}

특정 변수에 따라 plot을 여러 개로 구분해서 그리고 싶은 경우 col이라는 옵션에 특정 변수를 할당하면 된다. 추가적으로 palette 옵션을 사용하면 점의 색상을 구분하는 기준이 되는 색조를 변경할 수 있다.

{{< highlight py3 >}}
a = sns.relplot(x='page_views', y='market_value', data=epl, kind='scatter', style='region', col='big_club', color='y')
a.set_xticklabels(rotation=45)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.8.png">}}

scatterplot이나 lineplot에서 set_title 등의 옵션을 추가하고 싶은 경우에는 ax.set_title의 형식으로 명령어를 입력하였지만, relplot을 사용하는 경우에는 이것이 불가능하다. sns.relplot() 뒤에 .set_title의 형식으로 명령어를 붙이거나, 코드가 길어지는 것이 싫다면 sns.relplot을 하나의 변수에 할당한 후 해당 변수에 명령어를 붙여주는 방식을 사용해야 한다. 

{{< highlight py3 >}}
MUMC = epl2[(epl2['home_team']=='Manchester Utd') | (epl2['home_team']=='Manchester City')].reset_index(drop=True)
{{< / highlight >}}

{{< highlight py3 >}}
a = sns.relplot(x='game_date', y='home_team_score', data=MUMC, kind='line', col='home_team', col_wrap=1, hue='home_team')
a.set_xticklabels(rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.9.png">}}

kind='line'일 경우에도 'scatter'와 동일한 구조를 갖는다. 이때 col_wrap은 한 행에 들어가는 그래프의 갯수를 설정하는 옵션이다.

## 3. Catplot

### 1) Pointplot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.pointplot(x='club', y='market_value', data=epl)
ax.set_xticklabels(epl.club.unique(), rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.10.png">}}

pointplot을 그릴 때는 sns.pointplot 명령어를 사용한다. 마찬가지로 x, y 변수와 데이터 출처를 입력하면 된다. 아무런 옵션을 주지 않는 경우에는 화면에 보이는 것처럼 여러 개의 평균값과 신뢰구간이 직선으로 이어져서 표현된다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.pointplot(x='club', y='market_value', data=epl, hue='big_club', capsize=0.5, join=False)
ax.set_xticklabels(epl.club.unique(), rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.11.png">}}

점들을 잇는 직선을 없애고 싶다면 join=False 옵션을 추가하면 된다. 또한 신뢰구간의 끝 부분의 형태를 바꾸고 싶은 경우 capsize 옵션을 사용할 수 있다.

### 2) Barplot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.barplot(x='club', y='market_value', data=epl)
ax.set_xticklabels(epl.club.unique(), rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.12.png">}}

막대그래프를 그릴 때는 sns.barplot 명령어를 사용한다. 아무런 옵션을 주지 않는 경우 신뢰구간과 함께 표현된다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.barplot(x='club', y='market_value', data=epl, hue='big_club', ci=None)
ax.set_xticklabels(epl.club.unique(), rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.13.png">}}

신뢰구간을 없애려면 ci=None 옵션을 추가하면 된다.

### 3) Countplot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.countplot(x='club', data=epl)
ax.set_xticklabels(epl['club'].unique(), rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.14.png">}}

countplot을 그릴 때는 sns.countplot 명령어를 사용한다. 범주별로 데이터의 갯수를 파악할 때 유용하게 사용할 수 있다.

### 4) Boxplot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.boxplot(x='club', y='market_value', data=epl)
ax.set_xticklabels(epl['club'].unique(), rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.15.png">}}

boxplot 역시 이름 그대로 sns.boxplot 명령어를 사용하면 된다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.boxplot(x='club', y='market_value', data=epl, whis=[10, 90])
ax.set_xticklabels(epl['club'].unique(), rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.16.png">}}

whis 옵션을 사용하면 boxplot의 상한과 하한을 조절할 수 있다. 위의 plot에서는 10%~90% 까지의 데이터를 포함하도록 설정한 것이다.

### 5) Violinplot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.violinplot(x='position_cat', y='market_value', data=epl)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.17.png">}}

위의 plot들과 별 차이가 없으므로 그냥 넘어가겠다.

### 6) Swarmplot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.swarmplot(x='position', y='market_value', data=epl)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.18.png">}}

마찬가지로 설명 생략.

### 7) Catplot

Relplot과 마찬가지로, Catplot의 kind 옵션을 사용하여 지금까지 살펴본 plot들을 나타낼 수 있다. 역시 여러 개의 plot을 동시에 그리기 쉽다는 것이 장점이다.

{{< highlight py3 >}}
a = sns.catplot(x='club', y='market_value', data=epl, kind='point', col='big_club', hue='big_club', capsize=0.5, join=False)
a.set_xticklabels(rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.19.png">}}

{{< highlight py3 >}}
a = sns.catplot(x='club', y='market_value', data=epl, kind='bar', col='position_cat', col_wrap=2)
a.set_xticklabels(rotation=90)
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.20.png">}}

## 4. Other Plots

### 1) Kernel-Density Plot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.kdeplot(epl2['home_team_score'])
ax.set_title('Home')
ax.set_xlabel('score')
ax.set_ylabel('probability')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.21.png">}}

Kernel-Density Plot을 사용하면 데이터의 분포를 화면에 나타낼 수 있다. 분포를 알고 싶은 변수 하나만 입력해주면 된다.

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.kdeplot(epl2['home_team_score'], shade=True)
sns.kdeplot(epl2['away_team_score'], shade=True)
ax.set_title('Home & Away')
ax.set_xlabel('score')
ax.set_ylabel('probability')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.22.png">}}

share=True 옵션을 추가할 경우 분포에 색을 입힐 수 있다. 위 plot과 같이 여러 변수의 분포를 동시에 표현하는 것도 가능하다.

### 2) Distribution Plot

{{< highlight py3 >}}
fig, ax = plt.subplots()
sns.distplot(epl2['home_team_score'])
sns.distplot(epl2['away_team_score'])
ax.set_title('Home & Away')
ax.set_xlabel('score')
ax.set_ylabel('probability')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.23.png">}}

Kernel-Density Plot과 유사한 plot으로 Distribution Plot이 있다. kde=False 옵션을 사용하면 kde curve를 제거할 수 있다.

### 3) Jointplot

{{< highlight py3 >}}
sns.jointplot(x='page_views', y='market_value', data=epl, kind='reg')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.24.png">}}

Jointplot은 두 변수 사이의 분포와 상관관계를 동시에 확인할 때 유용하다. kind='reg' 옵션을 사용함으로써 회귀선을 추가할 수 있다.

### 4) Pairplot

{{< highlight py3 >}}
sns.pairplot(epl[['page_views', 'market_value', 'fpl_points', 'big_club']], hue='big_club')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.25.png">}}

Pairplot은 Jointplot을 더 많은 변수에 대해 확장한 plot이다. 원하는 변수들을 데이터프레임 형태로 입력하면 된다.

### 5) Heatmap

{{< highlight py3 >}}
fig, ax = plt.subplots(figsize=(8, 8))
a = sns.heatmap(epl.corr(), annot=True, fmt='1.1f', linewidths=1, cmap='RdBu')
ax.set_title('Correlation Matrix')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.26.png">}}

Heatmap은 변수들 간의 Correlation Matrix를 그릴 때 사용한다. 데이터프레임에 .corr()를 추가하여 입력하면 된다. annot=True로 설정할 경우 상관계수의 수치가 화면에 표시되고, fmt로 소수점 자릿수를 설정할 수 있다. 또한 linewidth나 cmap 등의 옵션으로 Matrix 간격이나 색상을 조정할 수 있다.

## 5. Change the Style

{{< highlight py3 >}}
sns.set_style('whitegrid')
sns.set_palette('RdBu')
sns.set_context('poster')
{{< / highlight >}}

{{< highlight py3 >}}
sns.relplot(x='page_views', y='market_value', data=epl, kind='scatter', hue='big_club')
plt.show()
{{< / highlight >}}
{{<figure src="/sns_2.27.png">}}

sns.set_style, sns.set_palette, sns.set_context 등의 명령어를 사용하여 Seaborn의 스타일을 변경할 수 있다. 그렇게 중요한 기능은 아니므로 자세하게 설명하지는 않도록 하겠다.