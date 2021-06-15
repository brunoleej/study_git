# CNN은 이미지 처리에 탁월한 성능을 보이는 신경망입니다.
# CNN은 크게 Convolution layer와 Pooling layer로 구성됩니다. 

# Channel
# (height, width, channel) 3차원 텐서

# Convolution operation
# Convolution Layer는 합성곱 연산을 통해서 이미지의 특징을 추출하는 역할을 합니다.
# 커널(kernel) 또는 필터(filter)라는  n x m 크기의 행렬로 (height) x (width) 크기의 이미지를 처음부터 끝까지 겹치며 훑으면서 크기의 겹쳐지는 부분의 각 이미지와 
# 커널의 원소의 값을 곱해서 모두 더한 값을 출력으로 하는 것을 말합니다. 
# 이때, 이미지의 가장 왼쪽 위부터 가장 오른쪽까지 순차적으로 훑습니다.
# Kernel은 일반적으로 3 x 3 또는 5 x 5를 사용

# Stride
# Kernel의 이동 범위를 stride라고 합니다.

# Padding
# 합성곱 연산의 결과로 얻은 Feature map은 입력의 크기보다 작아진다는 특징이 있습니다.
# 만약, 합성곱 층을 여러개 쌓았다면 최종적으로 얻은 특성 맵은 초기 입력보다 매우 작아진 상태가 되어 버립니다.
# 합성곱 연산 이후에도 특성 맵의 크기가 입력의 크기와 동일하게 유지되도록하고 싶다면 padding을 사용하면 됨.
# padding은 입력의 가장자리에 지정된 개수의 폭만큼 행과 열을 추가해주는 것.

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Pooling
# 일반적으로 Convolution Layer(합성곱 연산 + 활성화 함수) 다음에는 Pooling Layer를 추가하는 것이 일반적이다.
# Pooling Layer에서는 Feature map을 downsampling하여 Feature map의 크기를 줄이는 pooling 연산이 이루어짐
# Maxpooling, Averagepooling이 있음