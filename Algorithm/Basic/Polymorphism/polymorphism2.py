# 메소드명을 동일하게 해서 같은 모양의 코드가 다른 동작을 하도록 하는 다형성의 예
class SalesWorker:
    def __init__(self, name):
        self.name = name

    def work(self):
        print(self.name + 'sells something')

class DevWorker:
    def __init__(self, name):
        self.name = name
    
    def work(self):
        print(self.name + 'develops something')
    
worker1 = SalesWorker('Dave')
worker2 = SalesWorker('David')
worker3 = SalesWorker('Andy')
worker4 = DevWorker('Aiden')
worker5 = DevWorker('Tina')
worker6 = DevWorker('Anthony')

workers = [worker1,worker2,worker3,worker4,worker5,worker6]

# 객체 타입에 따라 코드는 동일하나, 실제 호출되는 work 메소드가 다름
for worker in workers:
    worker.work()

'''
Davesells something
Davidsells something
Andysells something
Aidendevelops something
Tinadevelops something
Anthonydevelops something
'''