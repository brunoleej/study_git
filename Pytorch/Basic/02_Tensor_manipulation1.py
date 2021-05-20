# 자주 사용되는 기능
# Matrix Multiplication vs Multiplication
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1],[2]])

print('Shape of Matrix 1: ', m1.shape)
print('Shape of Matrix 2: ', m2.shape)
print(m1.matmul(m2))
# print(m1 @ m2)
'''
tensor([[ 5.],
        [11.]])
'''

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

print('Shape of Matrix 1: ', m1.shape)  # torch.Size([2, 2])
print('Shape of Matrix 2: ', m2.shape)  # torch.Size([2, 1])
print(m1 * m2)
'''
tensor([[1., 2.],
        [6., 8.]])
'''
print(m1.mul(m2))
'''
tensor([[1., 2.],
        [6., 8.]])
'''

# 평균(Mean)
t = torch.FloatTensor([1,2])
print(t.mean())  # tensor(1.5000) --> 1과 2의 평균인 1.5

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
'''
tensor([[1., 2.],
        [3., 4.]])
'''

print(t.mean())  # tensor(2.5000) --> 4개의 원소의 평균인 2.5 생성
print(t.mean(dim=0))    # tensor([2., 3.]) --> dim = 0은 첫번째 차원(행)
print(t.mean(dim=1))    # tensor([1.5000, 3.5000])
print(t.mean(dim=-1))   # tensor([1.5000, 3.5000])

# 덧셈(Sum)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
'''
tensor([[1., 2.],
        [3., 4.]])
'''

print(t.sum())  # 단순히 원소 전체의 덧셈을 수행 --> tensor(10.)
print(t.sum(dim=0)) # 행을 제거 --> tensor([4., 6.])
print(t.sum(dim=1)) # 열을 제거 --> tensor([3., 7.])
print(t.sum(dim=-1)) # 열을 제거 --> tensor([3., 7.])

# 최대(Max)와 아그맥스(ArgMax)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
'''
tensor([[1., 2.],
        [3., 4.]])
'''

print(t.max())   # tensor(4.) --> 원소 중 최대값인 4를 리턴합니다.
print(t.max(dim=0)) 
'''
torch.return_types.max(
values=tensor([3., 4.]),
indices=tensor([1, 1]))
'''
# 행의 차원을 제거한다는 의미이므로 (1, 2) 텐서를 만듭니다. 결과는 [3, 4]입니다.
# 첫번째 열에서 0번 인덱스는 1, 1번 인덱스는 3입니다.
# 두번째 열에서 0번 인덱스는 2, 1번 인덱스는 4입니다.
# 다시 말해 3과 4의 인덱스는 [1, 1]입니다.

print('Max: ', t.max(dim=0)[0])  # Max:  tensor([3., 4.])
print('Argmax: ', t.max(dim=0)[1])  # Argmax:  tensor([1, 1])

print(t.max(dim=1))
'''
torch.return_types.max(
values=tensor([2., 4.]),
indices=tensor([1, 1]))
'''
print(t.max(dim=-1))
'''
torch.return_types.max(
values=tensor([2., 4.]),
indices=tensor([1, 1]))
'''
