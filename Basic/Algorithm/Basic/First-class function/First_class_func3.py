# 3. 함수를 다른 함수에 인자로 넣을 수도 있음

def calc_square(digit):
    return digit * digit

def calc_plus(digit):
    return digit + digit

def calc_quad(digit):
    return digit * digit * digit * digit

def list_square(function, digit_list):
    result = list()
    for digit in digit_list:
        result.append(function(digit))
    print(result)

num_list = [1, 2, 3, 4, 5]

list_square(calc_square, num_list)
list_square(calc_plus, num_list)
list_square(calc_quad, num_list)

# num_list_square