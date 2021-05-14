import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class kmeans:
	df : pd.core.frame.DataFrame
	k : int
	def __init__(self,k,df):
		self.k=k
		self.df=df
		self.arr=np.array(self.df.drop('Name',1))
		self.train()
		
	
	def train(self):
		self.centroids=self.initialise_points()
		for _ in range(20): #number of times you need
			self.clusters=self.classify(self.get_distance(self.centroids))
			print(self.centroids)
			print()
			self.improve()
			print(self.centroids)
			print()
		visualize(self.clusters,self.centroids)
		
	def initialise_points(self):
		df=self.arr.copy()
		points=[]
		for i in range(self.k):
			a=np.random.rand()
			b=np.random.rand()
			points.append([a,b])
		return points
		
	def get_distance(self,points):
		
		distances=[]
		
		for point in points:
			l=[]
			for i in self.arr:
				dist=np.math.dist(point,i)
				l.append(dist)
			distances.append(l)
		return distances
	
	def classify(self,dist):
		dict={x+1:[] for x in range(self.k)}
		x=0
		for _ in range(len(dist[0])):
			arr=None
			least=float('inf')
			for n,value in enumerate(dist):
				if value[x]<least:
					arr=self.arr[x],n
					least=value[x]
			dict[arr[1]+1].append(arr[0])
			x+=1
		return dict
	
	def improve(self):
		for i in range(self.k):
			sum1=0
			sum2=0
			for j in self.clusters[i+1]:
				sum1+=j[0]
				sum2+=j[1]
			
			mean1=sum1/len(self.clusters[i+1]) if len(self.clusters[i+1]) else sum1
			mean2=sum2/len(self.clusters[i+1]) if len(self.clusters[i+1]) else sum2
			self.centroids[i]=[mean1,mean2]
	
def visualize(d,centroids):
	colors=['red','green','yellow','black','green']
	for color,x in enumerate(d):
		l=d[x]
		for i in l:
			plt.scatter(i[0],i[1],color=colors[color])
	for j in centroids:
		plt.scatter(j[0],j[1],color='blue')
	plt.show()
		
df=pd.read_csv('Income.csv')

scaler=MinMaxScaler()

print(df['Income($)'].shape)
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)','Age']])
scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age','Income($)']])
print(df)
kmeans(3,df)