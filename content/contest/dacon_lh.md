---
title: 한국토지주택공사 주차수요 예측 AI 경진대회
date: 2021-10-27
sidebar: false
---

## 1. 대회 주제 및 목적

아파트 단지 내 주차대수는 법정주차대수와 장래주차수요 중 큰 값에 따라 결정하는 것이 원칙이다. 따라서 불필요한 공간 소요를 줄이기 위해서는 장래주차수요를 정확하게 추정하는 것이 중요하다. 현재 단지 내 장래주차수요는 주차원단위와 건축연면적을 활용하는 Rule-Based 방식을 통해 산출되고 있다. 하지만 이러한 방식은 인력 조사로 인한 오차 발생, 현장 조사 시점과 실제 건축 시점 사이의 시간차 등의 문제로 인해 장래주차수요가 과대 또는 과소 추정될 수 있다는 한계점을 가지고 있다.

> **주차원단위법**
> 
> $P=\dfrac{U\times F}{1000\times e}$
> 
> $P$: 주차 수요(대) \
> $U$: 주차원단위(대/$1,000m^2$), 건축 예정 부지 인근 유사 단지의 피크 시간대 건물연면적 $1,000m^2$당 주차발생량 \
> $F$: 건축 예정 부지의 건축연면적($m^2$) \
> $e$: 주차 이용 효율

이에 한국토지주택공사 주차수요 예측 AI 경진대회에서는 머신러닝 모형을 활용하여 유형별 임대주택 설계 시 단지 내 적정 주차 수요를 예측함으로써 기존의 추정 방식을 보완할 수 있는 방안을 제안하는 것을 목표로 하였다. 제안된 모형의 예측 정확도 평가 기준으로는 실제 단지 내 등록차량수와 예측값 사이의 MAE를 사용하였다.

## 2. 데이터 설명

|테이블명|설명|크기|
|:-:|:-:|:-:|
|TRAIN|총세대수, 공급유형, 전용면적, 자격유형, 임대료, 도보 10분거리 내 지하철역 수, 등록차량수 등 단지 내 건물별 정보 (종속변수 포함)|2,952$\times$15|
|TEST|총세대수, 전용면적, 임대료, 도보 10분거리 내 지하철역 수 등 단지 내 건물별 정보 (종속변수 미포함)|1,022$\times$14|
|AGE_GENDER_INFO|지역 임대주택 나이별, 성별 인구 분포|16$\times$23|

Sample Data: [https://dacon.io/competitions/official/235745/data](https://dacon.io/competitions/official/235745/data)

## 3. 주요 과정 및 결과

{{<figure src="/contest/lh1.jpg" width="1100">}}

{{<figure src="/contest/lh2.jpg" width="1100">}}

{{<figure src="/contest/lh3.jpg" width="1100">}}

{{<figure src="/contest/lh4.jpg" width="1100">}}

{{<figure src="/contest/lh5.jpg" width="1100">}}

{{<figure src="/contest/lh6.jpg" width="1100">}}
