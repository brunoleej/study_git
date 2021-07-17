# 요정(Elf), 파이터(Fighter) 클래스 만들기
    # 이름을 입력받음 - Elf의 attack 메소드 : 출력 "마법으로 공격합니다."
    # Figheter의 attack 메소드 : 출력 "주먹으로 공격합니다."
    # 아래와 같이 객체 생성 후 반복문으로 공격
'''
elf1 = Elf('Dave')
fighter1 = Figheter('Anthony')
ourteam = [elf, fighter1]
for attacker in ourteam:
    attacker.attack()
'''

class Elf:
    def __init__(self, name):
        self.name = name
    
    def attack(self):
        print(self.name + '마법으로 공격합니다.')
    
class Fighter:
    def __init__(self, name):
        self.name = name
    
    def attack(self):
        print(self.name + "주먹으로 공격합니다.")

elf1 = Elf('Dave')
fighter1 = Fighter('Anthony')

ourteam = [elf1, fighter1]

for attacker in ourteam:
    attacker.attack()

'''
Dave마법으로 공격합니다.
Anthony주먹으로 공격합니다.
'''