a=list(map(int,input().split()))
b=int(input())
for i in range(0,len(a)):
    for j in range(i+1,len(a)):
        if a[i]+a[j]==b:
            print(f"[{i}, {j}]")
            break
