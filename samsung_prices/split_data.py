import numpy as np

data = np.load('./samsung_prices.npy')
# print(data)
# print(data.shape)   # (2397, 6)

def split_x(seq, size, col) :
    dataset = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    print(type(dataset))
    return np.array(dataset)

dataset = split_x(data,5, 8)
print(dataset)
# print(dataset.shape) # (2393, 5, 6)