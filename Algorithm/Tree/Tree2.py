# 이진 탐색 트리에 데이터 넣기
    # 이진 탐색 트리 조건에 부합하게 데이터를 넣어야 함
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
                
head = Node(1)
BST = NodeMgmt(head)
BST.insert(2)

