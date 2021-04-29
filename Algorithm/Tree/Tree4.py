# 이진 탐색 트리 삭제
    # 매우 복잡함. 경우를 나누어서 이해하는 것이 좋음

# Leaf Node 삭제
    # Leaf Node : Child Node가 없는 Node
    # 삭제할 Node의 Parent Node가 삭제할 Node를 가리키지 않도록 한다.

# Child Node가 하나인 Node 삭제
    # 삭제할 Node의 Parent Node가 삭제할 Node의 Child Node를 가리키도록 한다.

# Child Node가 두 개인 Node 삭제
    # 1. 삭제할 Node의 오른쪽 자식 중, 가장 작은 값을 삭제할 Node의 Parent Node가 가리키도록 한다.
    # 2. 삭제할 Node의 왼쪽 자식 중, 가장 큰 값을 삭제할 Node의 Parent Node가 가리키도록 한다.

class Node:
    def __init__(self, value):
        self.value = value
        self.left, self.right = None, None  

class NodeMgmt:
    def __init__(self, head):
        self.head = head

    def insert(self, value):
        self.current_node = self.head
        while True:
            if value < self.current_node.value:
                if self.current_node.left != None:
                    self.current_node = self.current_node.left
                else:
                    self.current_node.left = Node(value)
                    break
            else:
                if self.current_node.right != None:
                    self.current_node = self.current_node.right
                else:
                    self.current_node.right = Node(value)
                    break
    
    def search(self, value):
        self.current_node = self.head
        while self.current_node:
            if self.current_node.value == value:
                return True
            elif value < self.current_node.value:
                self.current_node = self.current_node.left
            else:
                self.current_node = self.current_node.right
        return False
    
    def delete(self, value):    # 함수 내 삭제할 Node의 데이터인 value를 입력으로 받음
        searched = False    # delete할 Node가 있는지를 판단하는 boolean 변수 선언
        self.current_node, self.parent = self.head, self.head
        while self.current_node:
            if self.current_node.value == value:
                searched = True
                break
            elif value < self.current_node.value:
                self.parent = self.current_node
                self.current_node = self.current_node.left
            else:
                self.parent = self.current_node
                self.current_node = self.current_node.right
            
        if searched == False:   # delete할 Node가 없으면, False를 리턴하고 함수 종료
            return False
# 이 라인에 오면, self.current_node 가 삭제할 Node, self.parent는 삭제할 Node의 Parent Node인 상태가 됨
