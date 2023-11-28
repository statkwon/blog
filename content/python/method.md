---
title: "Method"
date: 2023-10-01
categories:
  - "Python"
tags:
  - "OOP"
  - "Method"
sidebar: false
draft: true
---

파이썬에서 객체의 method는 크게 세 가지로 구분할 수 있다.

1. instance method: 클래스의 기본적인 method. 첫 번째 인자로 항상 self를 받는다.
2. static method: 인스턴스의 attirubte에 변화를 일으키지 않는 method. 주로 클래스와 관련이 있는 유틸리티성 함수를 만들 때 사용된다.
3. class method: 임의의 클래스의 attribute에 접근할 수 있는 method. 첫 번째 인자로 항상 cls를 받는다.

```py
class MyClass:
    def __init__(self, var):
        self.var = var

    def func1(self):
        print(self.var)

    @staticmethod
    def func2():
        print("func2()")

    @classmethod
    def func3(cls):
        print(cls.name)

myclass = MyClass(0)
myclass.func()
```
