from sklearn.cluster import KMeans
import numpy as np

# FCM دستی با NumPy (برای سازگاری با محیط – تست شده)
def fuzzy_cmeans(data, n_clusters, m=2.0, max_iter=100, error=1e-5):
    """
    پیاده‌سازی FCM: بازگشت مراکز و عضویت‌ها (u)
    data: (n_samples, n_features)
    m: فازی‌ساز (معمولاً 2)
    """
    n_samples, n_features = data.shape
    # عضویت‌های اولیه رندوم
    u = np.random.dirichlet(np.ones(n_clusters), size=n_samples)
    # مراکز اولیه رندوم
    centers = np.random.rand(n_clusters, n_features)
    
    for it in range(max_iter):
        u_old = u.copy()
        # به‌روزرسانی مراکز
        um = u ** m
        centers = (um.T @ data) / np.sum(um, axis=0)[:, np.newaxis]
        # به‌روزرسانی عضویت‌ها
        dist = np.linalg.norm(data[:, np.newaxis] - centers, axis=2) + 1e-10  # جلوگیری از تقسیم بر صفر
        u = 1.0 / (dist ** (2 / (m - 1)))
        u = (u.T / np.sum(u, axis=1)).T  # نرمالایز ردیفی
        
        # چک همگرایی
        if np.linalg.norm(u - u_old) < error:
            break
    
    return centers, u

# Function to extract rules based on fuzzy clustering (FCM حالا استفاده می‌شه)
def extract_fuzzy_rules(data, n_clusters=3, return_u=False):
    """
    This function uses fuzzy clustering (FCM) to extract rules for the dataset.
    - n_clusters: number of fuzzy clusters to create
    - data: the dataset from which rules are to be extracted (numpy array)
    بازگشت rules با centroids و میانگین عضویت‌ها؛ اگر return_u=True, (centers, u) رو هم برمی‌گردونه
    """
    # FCM اجرا کن
    centers, u = fuzzy_cmeans(data, n_clusters)
    
    # Create fuzzy rules from the centroids and memberships
    rules = []
    for i, centroid in enumerate(centers):
        avg_membership = np.mean(u[:, i])  # میانگین عضویت به این خوشه برای تمام داده‌ها
        rule = {
            "cluster_id": i,
            "centroid": centroid,
            "avg_membership": avg_membership,
            "rule": f"If data is close to {centroid} with average membership {avg_membership:.4f}, then the task importance is {i+1}"
        }
        rules.append(rule)
    
    if return_u:
        return centers, u  # برای ادغام در maml
    return rules

# Example usage with random data (for demonstration purposes) – حذف اگر لازم نیست
data = np.random.rand(100, 5)  # 100 samples, 5 features
rules = extract_fuzzy_rules(data)
for rule in rules:
    print(rule)