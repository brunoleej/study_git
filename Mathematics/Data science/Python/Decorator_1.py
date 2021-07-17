# Write a decorator function that prints the execution time of the function.

# Moudule import
import time

# Decorator function Generate
def timer(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()    # code 1
        result = func(*args,**kwargs)# code 2 or code 4
        end_time = time.time()  # code 3
        print('running time : {0}'.format(end_time - start_time)) # code 3
    return wrapper

@timer
def test1(n1,n2):
    data = range(n1,n2 + 1)
    return sum(data)

@timer
def test2(n1,n2):
    result = 0
    for num in range(n1,n2+1):
        result += num
    return result

print(test1(1,10000))
print(test2(1,10000))