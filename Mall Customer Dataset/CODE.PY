import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

file_path = '/content/Mall_Customers.csv' 
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

features = ['Annual Income (k$)', 'Spending Score (1-100)']

# Standardizing the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Finding the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Applying K-Means with the optimal number of clusters (e.g., 3 clusters)
optimal_clusters = 5  # Replace this with the number from the elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

print(data)

