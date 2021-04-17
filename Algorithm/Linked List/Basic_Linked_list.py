# 1. 링크드 리스트 (Linked List) 구조
    # * 연결 리스트라고도 함
    # * 배열은 순차적으로 연결된 공간에 데이터를 나열하는 데이터 구조
    # * 링크드 리스트는 떨어진 곳에 존재하는 데이터를 화살표로 연결해서 관리하는 데이터 구조
    # * 본래 C언어에서는 주요한 데이터 구조이지만, 파이썬은 리스트 타입이 링크드 리스트의 기능을 모두 지원

# * 링크드 리스트 기본 구조와 용어
#   - 노드(Node): 데이터 저장 단위 (데이터값, 포인터) 로 구성
#   - 포인터(pointer): 각 노드 안에서, 다음이나 이전의 노드와의 연결 정보를 가지고 있는 공간

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


# Node와 Node 연결하기 (포인터 활용)
node1 = Node(1)
node2 = Node(2)
node1.next = node2
head = node1

# 링크드 리스트로 데이터 추가하기
class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

def add(data):
    node = head
    while node.next:
        node = node.next
    node.next = Node(data) 

node1 = Node(1)
head = node1
for index in range(2, 10):
    add(index)

# 링크드 리스트 데이터 출력하기(검색하기)
node = head
while node.next:
    print(node.data)
    node = node.next
print (node.data)