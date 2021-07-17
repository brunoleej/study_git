import queue

data_queue2 = queue.LifoQueue()

data_queue2.put('funcoding')
data_queue2.put('1')

print(data_queue2.qsize())  # 2
print(data_queue2.get())    # 1
print(data_queue2.qsize())  # 1
