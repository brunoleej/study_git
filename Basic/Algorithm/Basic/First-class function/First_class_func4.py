# 4. 함수의 결과값으로 함수를 리턴할 수도 있음1

def longger(msg):
    message = msg
    def msg_creator():  # <-- 함수 안에 함수를 만들 수도 있음
        print('[HIGH LEVEL]: ', message)
    return msg_creator

log1 = longger('Dave Log-in')
print(log1) # <function longger.<locals>.msg_creator at 0x000002629D521310>
log1()  # [HIGH LEVEL]:  Dave Log-in