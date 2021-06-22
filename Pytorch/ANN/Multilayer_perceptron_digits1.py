import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

print(digits.images[0])
'''
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
'''

print(digits.target[0]) # 0
print('전체 샘플 수 : {}'.format(len(digits.images)))   # 전체 샘플 수 : 1797

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]): # 5개의 샘플만 출력
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' % label)
plt.show()

for i in range(5):
    print(i,'번 index sample의 label: ', digits.target[i])
'''
0 번 index sample의 label:  0
1 번 index sample의 label:  1
2 번 index sample의 label:  2
3 번 index sample의 label:  3
4 번 index sample의 label:  4
'''
print(digits.data[0])
'''
[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
 15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
  0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
  0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
'''

X = digits.data
Y = digits.target
