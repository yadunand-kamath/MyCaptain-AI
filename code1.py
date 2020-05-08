print("Enter a limit:")
n = int(input())
 
a = 0
b = 1
print("Fibonacci Series:")
ls = [0,1]
if n < 1: 
    print("Incorrect input. Enter a number greater than zero.") 
elif n == 1: 
    print(a)
elif n == 2:
    print(ls)
else: 
    for i in range(2,n): 
        c = a + b 
        a = b 
        b = c 
        ls.append(c)
    print(ls)

