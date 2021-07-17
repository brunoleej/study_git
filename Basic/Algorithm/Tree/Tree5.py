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