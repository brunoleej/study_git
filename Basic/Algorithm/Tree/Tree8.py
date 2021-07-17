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

        # Case1 : 삭제할 Node가 단말 노드인 경우
        # self.current_node가 삭제할 Node, self.parent는 삭제할 Node의 Parent Node인 상태가 됨
        if self.current_node.left == None and self.current_node.right == None:
            if value < self.parent.value:
                self.parent.left = None
            else:
                self.parent.right = None

        # Case2 : 삭제할 Node가 Child Node를 한 개 가지고 있을 경우
        # self.current_node가 삭제할 Node, self.parent는 삭제할 Node의 Parent Node인 상태가 됨
        if self.current_node.left != None and self.current_node.right = None:
            if value < self.parent.value:
                self.parent.left = self.current_node.left
            else:
                self.parent.right = self.current_node.left
        elif self.current_node.left == None and self.current_node.right != None:
            if value < self.parent.value:
                self.parent.left = self.current_node.right
            else:
                self.parent.right = self.current_node.right
        
        # Case3 : 삭제할 Node가 Child Node를 두개 가지고 있을 경우 (삭제할 Node가 Parent Node 왼쪽에 있을 때)
        # 기본 사용 가능 전략
            # 1. 삭제할 Node의 오른쪽 자식 중, 가장 작은 값을 삭제할 Node의 Parent Node가 가리키도록 한다.
            # 2. 삭제할 Node의 왼쪽 자식 중, 가장 큰 값을 삭제할 Node의 parent Node가 가리키도록 한다.
        # 기본 사용 가능 전략 중, 1번 전략을 사용하여 코드를 구현
            # 경우의 수가 또 다시 두가지가 있음
                # Case 3-1 : 삭제할 Node가 Parent Node의 왼쪽에 있고, 삭제할 Node의 오른쪽 자식 중, 가장 작은 값을 가진 Node의 Child Node가 없을 때
                # Case 3-2 : 삭제할 Node가 Parent Node의 왼쪽에 있고, 삭제할 Node의 오른쪽 자식 중, 가장 작은 값을 가진 Node의 오른쪽에 Child Node가 있을 때
                # 가장 작은 값을 가진 Node의 Child Node가 왼쪽에 있을 경우는 없음. 왜냐하면 왼쪽 Node가 있다는 것은 해당 Node보다 더 작은 값을 가진 Node가 있다는 뜻이기 때문.
        
        # Case 3-1
        # self.current_node가 삭제할 Node, self.parent는 삭제할 Node의 Parent Node인 상태가 됨
        if self.current_node.left != None and self.current_node.right != None:
            if value < self.parent.value:
                self.change_node = self.current_node.right
                self.change_node_parent = self.current_node.right
                while self.change_node.left != None:
                    self.change_node_parent = self.change_node
                    self.change_node = self.change_node.left
                self.change_node_parent.left = None
                if self.change_node.right != None:
                    self.change_node_parent.left = self.change_node.right
                else:
                    self.change_node_parent.left = None
                self.parent.left = self.change_node
                self.change_node.right = self.change_node_parent
                self.change_node.left = self.current_node.left

        # Case 3-2 : 삭제할 Node가 Child Node를 두 개 가지고 있을 경우 (삭제할 Node가 Parent Node 오른쪽에 있을 때)
        # self.current_node 가 삭제할 Node, self.parent는 삭제할 Node의 Parent Node인 상태가 됨
        else:
            self.change_node = self.current_node.right
            self.change_node_parent = self.current_node.right
            while self.change_node.left != None:
                self.change_node_parent = self.change_node
                self.change_node = self.change_node.left
            if self.change_node.right != None:
                self.change_node_parent.left = self.change_node.right
            else:
                self.change_node_parent.left = None
            self.parent.right = self.change_node
            self.change_node.left = self.current_node.left
            self.change_node.right = self.current_node.right
        return True