# 배열(Array)
    # 데이터를 나열하고, 각 데이터를 인덱스에 대응하도록 구성한 데이터 구조
    # 파이썬에서는 리스트 타입이 배열 기능을 제공하고 있음

# 배열이 왜 필요할까?
    # 같은 종류의 데이터를 효율적으로 관리하기 위해 사용
    # 같은 종류의 데이터를 순차적으로 저장

# 배열의 장점
    # 빠른 접근 가능

# 배열의 단점
    # 추가 / 삭제가 쉽지 않음
    # 미리 최대 길이를 지정해야 함

# 파이썬과 C언어의 배열 예제
# include <stdio.h>

# C언어 배열
'''
int main(int argc, char * argv[])
{
    char country[3] = "US";
    printf("%c%c\n", country[0],country[1]);
    printf("%s\n"m country)
    return 0;
}
'''

# 파이썬 배열
country = 'US'
print(country)  # US

country = country + 'A'
print(country)  # USA
