stack_list = list()

def push(data):
    stack_list.append(data)
    
def pop():
    data = stack_list[-1]
    del stack_list[-1]
    return data

for index in range(10):
    push(index)

print(pop())    # 9