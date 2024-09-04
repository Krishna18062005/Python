"""ML-LEARNINGS"""

import numpy as np

from time import process_time

ls=[i for i in range(10000)]
st=process_time()
ls=[i+5 for i in ls]
et=process_time()
print(et-st)

np_array=np.array([i for i in range(10000)])
st=process_time()
np_array+=5
et=process_time()
print(et-st)

list=[1,2,3,4]
print(list)
type(list)

np_array= np.array([1,2,3,4,5])
print(np_array)
type(np_array)

#creating a 1 d array
ar=np.array([1,2,3,4])
print(ar)

ar.shape

b=np.array([(7,8),(1,2)])
print(b)

b.shape

c=np.array([(1,2,3,4),(5,6,7,8)],dtype=float)
print(c)

x=np.zeros((4,5))
print(x)

y=np.ones((4,5))
print(y)

z=np.full((5,4),4)
print(z)

#identity matrix
a=np.eye(3)
print(a)

#np array with random values
b=np.random.random((4,3))
print(b)

#array with random integers
ar=np.random.randint(10,100,(3,4))
print(ar)

#array of evenly spaced values--> no of values req
d=np.linspace(10,30,5)
print(d)

# array of evenly spaced values --. specifying teh step
a=np.arange(10,30,5)
print(a)

#convert a list to a np array
lst=[1,2,34,33]
ar=np.asarray(lst)
print(ar)

#analysing the numpy array
c=np.random.randint(10,90,(5,5))
print(c)

#array dimension
print(c.shape)
#no of dim
print(c.ndim)

#chck datatype of val in the array
print(c.dtype)

#no of elemnt sin a array
print(c.size)

#mathematical operations on np array
lst=np.array([1,2,3,4,5])
lst2=np.array([6,7,8,9,10])
print(lst+lst2)
print(lst-lst2)
print(lst%lst2)

ls=np.random.randint(19,222,(2,2))
ls2=np.random.randint(19,222,(2,2))
print(ls)
print(ls2)
print(ls+ls2)

ls=np.random.randint(19,222,(2,2))
ls2=np.random.randint(19,222,(2,2))
print(ls2)
print(ls)
print(np.subtract(ls,ls2))

arra=np.random.randint(0,10,(2,3))
print(arra)

print(np.transpose(arra))

arra=np.random.randint(0,10,(2,3))
print(arra)
trans=arra.T
print(trans)

#reshaping the array
a=np.random.randint(0,10,(2,3))
print(a)
print(a.shape)
b=a.reshape(3,2)
print(b)

import pandas as pd

from sklearn.datasets import load_diabetes

ds=load_diabetes();
print(ds)

"""#pandas dataframe
df=pd.DataFrame(ds.data,columns=ds.feature_names)

#pandas dataframe
df=pd.DataFrame(ds.data,columns=ds.feature_names)
print(df)
"""

df.head()

df.shape

#csv file to dataframe
didf=pd.read_csv('/content/diabetes.csv')

print(didf)
didf.shape
type(didf)

didf.head()

didf.shape

#df=pd.read_excel('/content/diabetes.csv')
#exporting df to a csv file
df.to_csv('krish.csv')

import numpy as np
random=pd.DataFrame(np.random.rand(20,10))

print(random)

random.head()

random.shape

#inspecting dataframe
random.info()

didf.shape

didf.head()

didf.tail()#last 5 rows

#finding no of missing values
didf.isnull().sum()

# counting the values based on the values
didf.value_counts('Outcome')

#grp the values based on the mean
didf.groupby('Outcome').mean()

df.mean()#clmn

didf.std()



didf.count()

didf.min()

didf.max()

didf.describe()

#adding a colm to a df
didf['age in df']=didf['Age']+10
didf.head()

#for removing particular row
didf.drop(index=0,axis=0)

# drop a colm
didf.drop(columns='age in df',axis=1)

#locating a row using index values
didf.iloc[1]

# locate particular colmn
didf.iloc[:,0-1]

#corelation(2type)+ve & -ve
didf.corr()

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,20,100)
y=np.sin(x)
z=np.cos(x)
plt.plot(x,y)
plt.show()
plt.plot(x,z)
plt.show()

#adding title x-axis and y-axis labells
plt.plot(x,y)
plt.xlabel('angle')
plt.ylabel("sin val")
plt.title('Sine Wave')
plt.show()

x=np.linspace(-10,10,20)
y=x**2
plt.plot(x,y)
plt.title("parabola")
plt.show()

plt.plot(x,y,'r--')
plt.show()

x=np.linspace(-5,5,50)
plt.plot(x,np.sin(x),'g*')
plt.plot(x,np.cos(x),'r+')
plt.show()

flg=plt.figure()
ax1=flg.add_axes([0,0,1,1])
lan=['eng','fren','hin','span','latin']
pple=[10,23,44,55,44]
plt.bar(lan,pple)
plt.show()

fig1=plt.figure()
acx1=fig1.add_axes([0,0,1,1])
lan=['eng','fren','hin','span','latin']
pple=[10,23,44,55,44]
acx1.pie(pple,labels=lan,autopct="%1.1f%%")
plt.show()

# scatter plot
x=np.linspace(0,10,30)
y=np.sin(x)
z=np.cos(x)
fig2=plt.figure()
ax1=fig2.add_axes([0,0,1,1])
plt.scatter(x,y,color='g')
plt.scatter(x,z,color='r')
plt.show()

#3d scatter plot
fig3=plt.figure()
ax=plt.axes(projection='3d')
z=20*np.random.random(100)
x=np.sin(z)
y=np.cos(z)
ax.scatter3D(x,y,z,color='b')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tips=sns.load_dataset('tips')
print(tips)

tips.head( )

sns.relplot(data=tips,x='total_bill',y='tip',col='time',hue='smoker',style='smoker',size='size')

#setting a theme for the plt
sns.set_theme(style='darkgrid')

sns.relplot(data=tips,x='total_bill',y='tip',col='time',hue='smoker',style='smoker',size='size')

#loading iris dataset
iris=sns.load_dataset('iris')
iris.head()

#scatter plt
sns.scatterplot(x='sepal_length',y='petal_length',hue='species',data=iris)

#scatter plt
sns.scatterplot(x='sepal_length',y='petal_width',hue='species',data=iris)

#loading the titanic dataset
tit=sns.load_dataset('titanic')
tit.head()

#Count plot
sns.countplot(x='class',data=tit)

sns.countplot(x='survived',data=tit)

#bar
sns.barplot(x='sex',y='survived',hue='class',data=tit)

#house price predict
from sklearn.datasets import load_diabetes
db=load_diabetes()

hou=pd.DataFrame(db.data,columns=db.feature_names)
hou['prize']=db.target
hou.head()

sns.distplot(hou['bmi'])

plt.figure(figsize=(10,10))

sns.heatmap(hou.corr(),cbar=True,square=True,fmt='.1f', annot=True,annot_kws={'size':8},cmap='Blues')
