# Magic(special) Method
# compare
    # __eq__ : ==
    # __ne__ : !=
    # __lt__ : <(작다)
# calculate
    # __add__ : +
    # __sub__ : -
# __repr__ : 객체의 내용을 출력(개발자용)
# __str__ : 객체의 내용을 출력

print("test"=="test")   # True
print("test".__eq__('test'))    # True
print(dir(1))

class Txt:
    def __init__(self,txt):
        self.txt = txt
    
    def __eq__(self,txt_obj):
        return self.txt.lower() == txt_obj.txt.lower()
    
    def __repr__(self):
        return "Txt(txt={})".format(self.txt)
    
    def __str__(self):
        return self.txt

t1 = Txt('python')
t2 = Txt('Python')
t3 = t1

# print(t1 == t2, t1 == t3, t2 == t3) # False True False(__eq 넣기전)
print(t1 == t2, t1 == t3, t2 == t3) # True True True (__eq넣은후)

print(t1)   # python
