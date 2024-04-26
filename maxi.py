#maximum no of occurance of character;
a=input()
d={}
for i in a:
    d[i]=0
for i in a:
    d[i]+=1
print(max(d.values()))
