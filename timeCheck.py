from time import process_time
ls=[i for i in range(0,10000)]
st=process_time()
ls=[i+5 for i in ls]
et=process_time()
print(et-st)
