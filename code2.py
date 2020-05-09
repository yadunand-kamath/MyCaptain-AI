# PRINT ALL POSITIVE NUMBERS IN A RANGE

list1 = [12,-7,5,64,-14]
print("Input1:",list1)
print("Output1:")
for x in list1:
    if x > 0:
        print(x)


list2 = [12,14,-95,3]
print("Input2:",list2)
print("Output2:")
for y in list2:
    if y > 0:
        print(y)
        

list3 = []
print("User Input:-")
n = int(input("Enter the number of elements:"))
print("Enter the elements:")
for i in range(0,n):
    ele = int(input())
    list3.append(ele)
print("Input3:",list3)
print("Output3:")
for z in list3:
    if z > 0:
        print(z) 
    
