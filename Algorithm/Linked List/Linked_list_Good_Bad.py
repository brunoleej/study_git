# 3. 링크드 리스트의 장단점 (전통적인 C언어에서의 배열과 링크드 리스트)
    # * 장점
    #   - 미리 데이터 공간을 미리 할당하지 않아도 됨
    #     - 배열은 **미리 데이터 공간을 할당** 해야 함
    # * 단점
    #   - 연결을 위한 별도 데이터 공간이 필요하므로, 저장공간 효율이 높지 않음
    #   - 연결 정보를 찾는 시간이 필요하므로 접근 속도가 느림
    #   - 중간 데이터 삭제시, 앞뒤 데이터의 연결을 재구성해야 하는 부가적인 작업 필요

node = head
while node.next:
    print(node.data)
    node = node.next
print (node.data)

node3 = Node(1.5)

node = head
search = True
while search:
    if node.data == 1:
        search = False
    else:
        node = node.next

node_next = node.next
node.next = node3
node3.next = node_next

node = head
while node.next:
    print(node.data)
    node = node.next
print (node.data)