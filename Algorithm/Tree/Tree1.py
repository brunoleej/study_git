# 대표적인 자료구조 : 트리
# 트리(Tree) 구조

# 트리 : Node와 Branch를 이용해서, 사이클을 이루지 않도록 구성한 데이터 구조
# 실제로는 어디에 많이 사용되나?
    # 트리 중 이진 트리(Binary Tree) 형태의 구조로, 탐색(검색) 알고리즘 구현을 위해 많이 사용됨

# 알아둘 용어
    # Node : 트리에서 데이터를 저장하는 기본 요소 (데이터와 다른 연결된 노드에 대한 Branch 정보 포함)
    # Root Node : 트리 맨 위에 있는 노드
    # Level : 최상위 노드를 Level 0으로 하였을 때, 하위 Branch로 연결된 노드의 깊이를 나타냄
    # Parent Node : 어떤 노드의 상위 레벨에 연결된 노드
    # Leaf Node (Terminal Node) : Child Node가 하나도 없는 노드
    # Sibling (Brother Node) : 동일한 Parent Node를 가진 노드
    # Depth : 트리에서 Node가 가질 수 있는 최대 Level

# 이진 트리와 이진 탐색 트리 (Binary Search Tree)
    # 이진 트리 : 노드의 초대 Branch가 2인 트리
    # 이진 탐색 트리 (Binary Search Tree : BST) : 이진 트리에 다음과 같은 추가적인 조건이 있는 트리
        # 왼쪽 노드는 해당 노드보다 작은 값, 오른쪽 노드는 해당 노드보다 큰 값을 가지고 있음

# 자료 구조 이진 탐색 트리의 장점과 주요 용도
    # 주요 용도 : 데이터 검색(탐색)
    # 장점 : 탐색 속도를 개선할 수 있음
    # 단점 :  추후에 작성

# 노드 클래스 만들기
class Node:
    def __init__(self, value):
        self.value = value
        self.left, self.right = None, None  # 이렇게도 한번에 여러 변수를 초기화할 수 있음
    

