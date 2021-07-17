# super : 부모 클래스에서 사용된 함수의 코드를 가져다가 자식 클래스 함수에서 재사용할때 사용
'''
class A:
    def plus(self):
        code1
class B(A):
    def minus(self):
        code1   # super().plus()
        code2
'''
class Marine:
    def __init__(self):
        self.health = 40
        self.attack_pow = 5

    def attack(self,unit):
        unit.health -= self.attack_pow
        if unit.health <= 0:
            unit.health = 0

class Marine2(Marine):
    def __init__(self):
        # self.health = 40
        # self.attack_pow = 5
        super().__init__()
        self.max_health = 40

marine = Marine2()

print(marine.health, marine.attack_pow, marine.max_health)