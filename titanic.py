from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing
from matplotlib import pyplot as plt
import numpy as np
import json
import random
from mpl_toolkits.mplot3d import Axes3D

#Function to calculate squared euclidean distance between points
def calc_distance(array1, array2):
	dist = 0.0
	for i in range(len(array1)):
		dist = dist + (array1[i] - array2[i])**2
	return (dist**0.5)
#the array survived records whether an individual survived or not
survived=[]
#Open json file
data = json.load(open('titanic.json'))

#This array will be used to store normalized input data
pass_data = []
count = 0
sum = 0.0
#initialize variables required to normalize
minage= 1000000.0
maxage=-1000000.0

minfare= 1000000.0
maxfare=-1000000.0

minsib= 1000000.0
maxsib=-1000000.0

for line in data:
	#calculate the min and max to normalize the interval variables
	if line["Age"]:
		if float(line["Age"]) < minage:
			minage=float(line["Age"])
		elif float(line["Age"]) > maxage:
			maxage=float(line["Age"])
		count= count+1
		sum = sum + float(line["Age"])
	if float(line["Fare"]) < minfare:
		minfare=float(line["Fare"])
	elif float(line["Fare"]) > maxfare:
		maxfare=float(line["Fare"])

	if float(line["SiblingsAndSpouses"]) + float(line["ParentsAndChildren"]) < minsib:
		minsib=float(line["SiblingsAndSpouses"]) + float(line["ParentsAndChildren"])
	elif float(line["SiblingsAndSpouses"])  + float(line["ParentsAndChildren"]) > maxsib:
		maxsib=float(line["SiblingsAndSpouses"]) + float(line["ParentsAndChildren"])

mean = float(sum /count)
	
for line in data:
	record = []
	if line["Age"]:
		#Interval variables are normalized
		record.append((float(line["Age"])-minage)/(maxage-minage))
	else:
		record.append((mean-minage)/(maxage-minage))
	#The feature 'Fare' has been dropped after exploring its information value
	#Hence, the following row has been commented
	#record.append((float(line["Fare"])-minfare)/(maxfare-minfare))

	record.append((float(line["SiblingsAndSpouses"])+float(line["ParentsAndChildren"]) \
			-minsib)/(maxsib-minsib))

	#The feature 'Embarked' has been dropped after exploring its information value
	#Hence, the following set of rows has been commented
#	if line["Embarked"]== "C":
#		record.append(0.0)
#	elif line["Embarked"]== 'Q':
#		record.append(0.333333)
#	elif line["Embarked"]== 'S':
#		record.append(0.666666)
#	else:
		#Assign a new category to observations with missing values
#		record.append(1.0) 

	if line["Sex"] == 'male':
		record.append(0.0)
	elif line["Sex"] == 'female':
		record.append(1.0)
	else:
		record.append("")
	pass_data.append(record)
	survived.append(int(line['Survived']))

#This code was used for checking information value of various features.
#After deciding on the features to be used for analysis, this part of 
#the code is commented and will not be used again.

#for i in range(len(pass_data)):
#	plt.scatter(pass_data[i][0], pass_data[i][1], color='blue',s=50)
#plt.xlabel('Age')
#plt.ylabel('Fare')
#plt.savefig('Age vs Fare')
#plt.cla()
#for i in range(len(pass_data)):
#	plt.scatter(pass_data[i][0], pass_data[i][2], color='blue',s=50)
#plt.xlabel('Age')
#plt.ylabel('Companions')
#plt.savefig('Age vs Companions')
#plt.cla()
#for i in range(len(pass_data)):
#	plt.scatter(pass_data[i][2], pass_data[i][3], color='blue',s=50)
#plt.xlabel('Companions')
#plt.ylabel('Embarked')
#plt.savefig('Companions vs Embarked')
#plt.cla()
#for i in range(len(pass_data)):
#	plt.scatter(pass_data[i][2], pass_data[i][4], color='blue',s=50)
#plt.xlabel('Companions')
#plt.ylabel('Sex')
#plt.savefig('Companions vs Sex')
#plt.cla()
	
#perform hierarchial clustering
#for euclidean distance between clusters and metric
Z = linkage(np.array(pass_data), method='ward', metric='euclidean') 
plt.title('Hierarchical Clustering Dendrogram using selected features')
plt.xlabel('sample index')
plt.ylabel('distance')

#create a dendrogram hierarchial plot
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
#display both figures
plt.axhline(y=5,color="blue")
plt.savefig('Hierarchical Clustering Dendrogram using selected features')

#it makes sense to threshold at which creates 2 clusters
num_clusters = 2

#this array will contain the centroids 
cluster_centroids = []
for i in range(num_clusters):
	temp = []
	for j in range(len(pass_data[0])):
		#random.uniform is used to randomly select initial set of
		#clusters with each feature value between 0 and 1		
		temp.append(random.uniform(0,1))
	cluster_centroids.append(temp)
#There will be maximum of 10 iterations 
max_iter = 10
iter_num = 0

#This flag is used to check whether centroid values are changing
flag_cluster_change = 1
cluster_number=[1000]*len(pass_data)
min_dist = [10000000.00]*len(pass_data)
pass_data = np.array(pass_data)
cluster_centroids = np.array(cluster_centroids)


#Plotting the input data against the initial random centroids
fig = plt.figure()
ax = Axes3D(fig)
colors = ("red", "green", "blue","black","orange","violet")
for i in range(len(survived)):
	ax.scatter(pass_data[i,0],pass_data[i,1],pass_data[i,2], s=20, c = 'orange')
for i in range(num_clusters):
	ax.scatter(cluster_centroids[i,0],cluster_centroids[i,1], \
		cluster_centroids[i,2], s=500, c = colors[i],marker="*")
ax.set_xlabel('Age')
ax.set_ylabel('Companions')
ax.set_zlabel('Sex')
plt.title('Clusters')	
ax.set_xlim(0,1)	
ax.set_ylim(0,1)	
ax.set_zlim(0,1)	
plt.savefig('Clusters before iteration 1')


#Performing K-means clustering
while flag_cluster_change and iter_num<max_iter:
	#For each observation, calculating the distance from the centroids 
	#and assigning the observation to the cluster with the nearest centroid
	for i in range(len(pass_data)):
		min_dist[i] = 100000.0
		cluster_number[i] = 0
		for j in range(len(cluster_centroids)):
			centr_dist = calc_distance(pass_data[i], cluster_centroids[j])
			if centr_dist < min_dist[i]:
				cluster_number[i] = j
				min_dist[i] = centr_dist
	
	areallsame = 0
	#Calculate the revised ccentroid as mean of the value of the features 
	#for observations within the cluster
	for j in range(num_clusters):
		temp_centr = [0.0]*len(cluster_centroids[0])
		membercount = 0.0		
		print "cluster ",j, "before ", cluster_centroids[j]
		for i in range(len(pass_data)):
			if cluster_number[i]==j:
				membercount+=1
				for k in range(len(pass_data[i])):						
					temp_centr[k]+=pass_data[i][k]
				
		for k in range(len(temp_centr)):
			if membercount ==0:
				cluster_centroids[j][k]=0;	
			#Check if the centroid value is changing
			elif round(cluster_centroids[j][k],6)!=round(temp_centr[k]/membercount,6):
				#Assign new centroid location 
				cluster_centroids[j][k]=temp_centr[k]/membercount
			else:
				#Track the change in values of features of centroid
				areallsame+=1
		print "cluster ",j, "after ", cluster_centroids[j], " membercount = ", membercount
	print "areallsame = ",areallsame
	#Change value of termination condition if cluster location is constant
	if areallsame==num_clusters*len(pass_data[0]):
		flag_cluster_change=0
	
	#Plot the clusters and centroids
	fig = plt.figure()
	ax = Axes3D(fig)

	for i in range(len(survived)):
		ax.scatter(pass_data[i,0],pass_data[i,1],pass_data[i,2], s=20, c = colors[cluster_number[i]+2])
	for i in range(num_clusters):
		ax.scatter(cluster_centroids[i,0],cluster_centroids[i,1],cluster_centroids[i,2], s=500, c = colors[i],marker="*")
	ax.set_xlabel('Age')
	ax.set_ylabel('Companions')
	ax.set_zlabel('Sex')
	plt.title('Clusters')	
	ax.set_xlim(0,1)	
	ax.set_ylim(0,1)	
	ax.set_zlim(0,1)	
	plt.savefig('Clusters after iteration - %s' %(iter_num+1))
	iter_num+=1

#Plot the clusters to indicate who survived within each cluster
fig = plt.figure()
ax = Axes3D(fig)
for i in range(len(survived)):
	ax.scatter(pass_data[i,0],pass_data[i,1],pass_data[i,2], s=20, c = colors[survived[i]])
for i in range(num_clusters):
	ax.scatter(cluster_centroids[i,0],cluster_centroids[i,1],cluster_centroids[i,2], s=500, c = colors[i+2],marker="*")
ax.set_xlabel('Age')
ax.set_ylabel('Companions')
ax.set_zlabel('Sex')
ax.set_xlim(0,1)	
ax.set_ylim(0,1)	
ax.set_zlim(0,1)	
plt.title('Survived/Not Survived')	
plt.savefig('Survived vs Not Survived')

