# 5. 함수의 결과값으로 함수를 리턴할 수도 있음2
def html_creator(tag):
    def text_wrapper(msg):
        print('<{0}>{1}<{0}>'.format(tag, msg))
    return text_wrapper

h1_html_creator = html_creator('h1')    # 1
print(h1_html_creator)  # <function html_creator.<locals>.text_wrapper at 0x0000028B1AAF1430>

h1_html_creator('H1 태그는 타이틀을 표시하는 태그입니다.')  # <h1>H1 태그는 타이틀을 표시하는 태그입니다.<h1>

p_html_creator = html_creator('p')
p_html_creator('P 태그는 문단을 표시하는 태그입니다.')  # <p>P 태그는 문단을 표시하는 태그입니다.<p>
