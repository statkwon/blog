---
title: "Abstract Class"
date: 2023-10-01
categories:
  - "Python"
tags:
  - "OOP"
  - "Abstraction"
sidebar: false
---

파이썬에서도 abstract class를 만들 수 있다. abstract class로부터 인스턴스를 생성하는 것은 당연히 불가능하다.

```py
from abc import ABCMeta, abstractmethod

class BaseClass(metaclass=ABCMeta):
    @abstractmethod
    def func1(self):
        pass
    
    @abstractmethod
    def func2(self):
        pass

class DerivedClass(BaseClass):
    def func1(self):
        print("func1()")
    
    def func2(self):
        print("func2()")
```

파이썬에서 abstract class를 만들기 위해서는 abc(abstract base class) 모듈을 사용해야 한다. 사용 방법은 간단하다.

1. abc 모듈로부터 `ABCMeta`와 `abstractmethod`를 import한다.
2. 만들고자 하는 abstract class의 괄호 안에 `metaclass=ABCMeta`를 추가한다.
3. abstract method 위에는 `@abstractmethod`를 추가한다. 일반적으로 구현부에는 `pass`를 적지만, 실질적인 구현부를 포함하는 것 역시 가능하다. (상속 받은 클래스에서 `super()`를 통해 해당 구현부 호출 가능)

---

**Reference**

- [https://docs.python.org/3.10/library/abc.html](https://docs.python.org/3.10/library/abc.html)
