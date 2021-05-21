# Module import 
import numpy as np
import math
import random
import matplotlib.pyplot as plt
# %matplotlib inline

# 정사각형의 크기(Square_size), 원과 적사각형 안에 있는 점의 개수(points_inside_cir-cle), 임의의 점의 개수를 나타내는 샘플 크기(inside square) initialize
# 마지막으로 원의 사분면인 호(arc)를 정의
square_size = 1
points_inside_circle = 0
points_inside_square = 0
sample_size = 1000
arc = np.linspace(0, np.pi / 2, 100)


print(arc)
'''
[0.         0.01586663 0.03173326 0.04759989 0.06346652 0.07933315
 0.09519978 0.11106641 0.12693304 0.14279967 0.1586663  0.17453293
 0.19039955 0.20626618 0.22213281 0.23799944 0.25386607 0.2697327
 0.28559933 0.30146596 0.31733259 0.33319922 0.34906585 0.36493248
 0.38079911 0.39666574 0.41253237 0.428399   0.44426563 0.46013226
 0.47599889 0.49186552 0.50773215 0.52359878 0.53946541 0.55533203
 0.57119866 0.58706529 0.60293192 0.61879855 0.63466518 0.65053181
 0.66639844 0.68226507 0.6981317  0.71399833 0.72986496 0.74573159
 0.76159822 0.77746485 0.79333148 0.80919811 0.82506474 0.84093137
 0.856798   0.87266463 0.88853126 0.90439789 0.92026451 0.93613114
 0.95199777 0.9678644  0.98373103 0.99959766 1.01546429 1.03133092
 1.04719755 1.06306418 1.07893081 1.09479744 1.11066407 1.1265307
 1.14239733 1.15826396 1.17413059 1.18999722 1.20586385 1.22173048
 1.23759711 1.25346374 1.26933037 1.28519699 1.30106362 1.31693025
 1.33279688 1.34866351 1.36453014 1.38039677 1.3962634  1.41213003
 1.42799666 1.44386329 1.45972992 1.47559655 1.49146318 1.50732981
 1.52319644 1.53906307 1.5549297  1.57079633]
'''

# generate_points 함수를 정의합니다. 이 함수는 사각형 내부에 임의의 점을 생성합니다.
def generate_points(size):
    x = random.random()*size
    y = random.random()*size
    return (x,y)

# is_in_circle 함수를 정의합니다. 이 함수는 성생된 점이 원 안에 속하는지를 확인합니다.
def is_in_circle(point, size):
    return math.sqrt(point[0] ** 2 + point[1] ** 2) <= size

# pi값을 계산하는 함수를 정의합니다.
def compute_pi(points_inisde_circle, points_inside_square):
    return 4 * (points_inside_circle / points_inside_square)

# 샘플 크기만큼 사각형 내부에 임의의 점을 생성하고 points_inside_square 변수를 증가시킵니다. 생성된 점이 원 안에 있다면 points_inside_circle 변수를 증가시킵니다.
plt.axes().set_aspect('equal')
plt.plot(1 * np.cos(arc), 1 * np.sin(arc))

for i in range(sample_size):
    point = generate_points(square_size)
    plt.plot(point[0],point[1],'c')
    points_inside_square += 1
    if is_in_circle(point, square_size):
        points_inside_circle += 1

print('Approximate value of pi is {}'.format(compute_pi(points_inside_circle, points_inside_square)))
# Approximate value of pi is 3.152