# 위의 Person클래스를 상속받아 play의 메소드를 추가한 Child 클래스를 만들어주세요
# play: print('{} is playing.'.format(name))

class Child(Person):
    def play(self,name='Olivia'):
        print('{0} is playing'.format(name))

child = Child('Olivia', '8')
print(child.name, child.age)
print(child.play())