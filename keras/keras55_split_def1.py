# x와 y 분리하는 함수
# 다 : 1
import numpy as np
dataset = np.array(range(1,20))
# print(len(dataset)) # 19

def split_xy1(dataset, time_steps) :        # time_steps : 자르고 싶은 컬럼 수
    x, y = list(), list()                   # 리턴해줄 리스트 정의
    for i in range(len(dataset)) :          # dataset 길이만큼 반복
        end_number = i + time_steps         # end_number : y 데이터가 들어갈 변수 
        if end_number > len(dataset) - 1 :  # datset 길이보다 time_steps 길이가 더 긴 경우, 함수 정지
            break
        tmp_x, tmp_y = dataset[i : end_number], dataset[end_number] 
        x.append(tmp_x)                     # x, y 데이터를 리스트에 넣어줌
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(dataset, 5)
# print(x, "\n", y)
# print(x.shape)  # (14, 5)
# print(y.shape)  # (14, )

# 다 : 다
def split_xy2(dataset, time_steps, y_column) :  # y_column : 자르고 싶은 y 컬럼 수
    x, y = list(), list()                       # 리턴해줄 리스트 정의
    for i in range(len(dataset)) :              # dataset 길이만큼 반복
        x_end_number = i + time_steps           # x_end_number : x의 끝 번호
        y_end_number = x_end_number + y_column  # y_end_number : y의 끝 번호 
        if y_end_number > len(dataset) :        # datset 길이보다 time_steps 길이가 더 긴 경우, 함수 정지
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number] 
        x.append(tmp_x)                     # x, y 데이터를 리스트에 넣어줌
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_steps = 4
y_column = 2
x, y = split_xy2(dataset, time_steps, y_column)
# print(x, '\n', y)
# print("x shape " , x.shape) # (14, 4)
# print("y shape " , y.shape) # (14, 2)

# 다입력, 다 : 1
#1. 데이터
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],\
                    [11,12,13,14,15,16,17,18,19,20],\
                    [21,22,23,24,25,26,27,28,29,30]])
# print("dataset.shape : ", dataset.shape)     # (3, 10)
dataset = np.transpose(dataset)
# print(dataset)
print("dataset.shape : ", dataset.shape)     # (10, 3)

def split_xy3(dataset, time_steps, y_column) :
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy3(dataset, 3, 1)
# print(x, "\n", y)
# print(x.shape)      # (8, 3, 2)
# print(y.shape)      # (8, 1)
y = y.reshape(y.shape[0])   # 벡터 형태로 변환
# print(y.shape)      # (8,)