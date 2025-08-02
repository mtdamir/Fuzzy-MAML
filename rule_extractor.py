from sklearn.cluster import KMeans
import numpy as np

# Function to extract rules based on fuzzy clustering (Fuzzy C-Means)
def extract_fuzzy_rules(data, n_clusters=3):
    """
    This function uses fuzzy clustering to extract rules for the dataset.
    - n_clusters: number of fuzzy clusters to create
    - data: the dataset from which rules are to be extracted
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    # Create fuzzy rules from the centroids (cluster centers)
    rules = []
    for i, centroid in enumerate(centroids):
        rule = {
            "cluster_id": i,
            "centroid": centroid,
            "rule": f"If data is close to {centroid}, then the task importance is {i+1}"
        }
        rules.append(rule)
    
    return rules

# Example usage with random data (for demonstration purposes)
data = np.random.rand(100, 5)  # 100 samples, 5 features
rules = extract_fuzzy_rules(data)
for rule in rules:
    print(rule)
