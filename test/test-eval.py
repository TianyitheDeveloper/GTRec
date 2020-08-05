"""
执行字符串表达式，并返回表达式的值

列表推导式
"""

regs = '3*[1,2,3,4]'
regs1 = '[i if i%2 ==0 else "当前位置的数字不能被2整除" for i in range(10) ]'
print(eval(regs))
print(sum(eval(regs)))
print(eval(regs1))

