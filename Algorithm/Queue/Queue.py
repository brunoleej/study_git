import queue

data_queue = queue.Queue()          # 일반적인 FIFO정책

data_queue.put('funcoding')
data_queue.put(1)

print(data_queue.qsize())   # 2
print(data_queue.get()) # funcoding
print(data_queue.qsize())   # 1
print(data_queue.get()) # 1
print(data_queue.qsize())   # 0