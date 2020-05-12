# LISTS, TUPLES, DICTIONARY

l1 = [ 10 , 20 , 30 , 50 ]
print("List1:\n",l1)
l1.append(40)
print("Append 40 -",l1)
del l1[3]
print("Delete 50 -",l1) 

l2 = ["c" , "c++" , "java" ]
print("List2:\n",l2)
l2.append("python")
print("appending -",l2)

t1 = ( "hi" , 10 , "mycaptain" , 20 , "AI" )
print("Tuple:\n",t1)
print("Length of tuple is:",len(t1))
print("Accessing third element of tuple -",t1[2])

d1 = { "code" : "python" , "topic" : "AI" , "source" : "MyCaptain" , "month" : "may" }
print("Dictionary:\n",d1)
del d1["month"]
print("Delete month -",d1)
