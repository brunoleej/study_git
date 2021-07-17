import queue

data_queue3 = queue.PriorityQueue()

data_queue3.put((10,'korea'))
data_queue3.put((5, 1))
data_queue3.put((15,'china'))

print(data_queue3.qsize())  # 3
print(data_queue3.get())    # (5, 1)
print(data_queue3.qsize())  # 2
print(data_queue3.get())    # (10, 'korea')
print(data_queue3.qsize())  # 1
print(data_queue3.get())    # (15, 'china')
print(data_queue3.qsize())  # 0