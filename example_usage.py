import numpy as np
import matplotlib.pyplot as plt
from som import SelfOrganizingMap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate a more complex dataset for demonstration
def generate_customer_data(n_samples=200, random_state=42):
    """Generate synthetic customer data with age and income features."""
    # Generate 4 clusters of data
    X, y = make_blobs(n_samples=n_samples, centers=4, 
                     cluster_std=[1.0, 1.5, 0.5, 2.0], 
                     random_state=random_state, n_features=2)
    
    # Scale to represent age (20-70) and income (30-120k)
    X[:, 0] = X[:, 0] * 10 + 45  # Age
    X[:, 1] = X[:, 1] * 15 + 75  # Income
    
    # Ensure values are in reasonable ranges
    X[:, 0] = np.clip(X[:, 0], 20, 70)  # Age between 20 and 70
    X[:, 1] = np.clip(X[:, 1], 30, 120)  # Income between 30k and 120k
    
    return X, y

# 1. Basic usage example
def basic_example():
    print("Running basic SOM example...")
    
    # Create sample data - same as in the original code
    data = np.array([
        [25, 40],  # Customer A
        [45, 80],  # Customer B
        [30, 60],  # Customer C
        [50, 90]   # Customer D
    ])
    
    feature_names = ['Age', 'Income (k$)']
    
    # Create and train the SOM
    som = SelfOrganizingMap(grid_shape=(2, 2), learning_rate=0.5, max_iterations=20)
    som.fit(data)
    
    # Visualize the training results
    som.plot_training_history(data, feature_names=feature_names)
    plt.show()
    
    # Display U-Matrix
    som.plot_umatrix()
    plt.show()
    
    print("Basic example completed.\n")

# 2. Advanced usage with larger dataset
def advanced_example():
    print("Running advanced SOM example with synthetic data...")
    
    # Generate synthetic customer data
    X, y = generate_customer_data(n_samples=200)
    feature_names = ['Age', 'Income (k$)']
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a larger SOM grid
    som = SelfOrganizingMap(grid_shape=(4, 4), learning_rate=0.5, 
                          max_iterations=1000, sigma_initial=2.0)
    
    # Train the SOM
    print("Training SOM on synthetic customer data...")
    som.fit(X_scaled, epochs=1)
    
    # Transform scaled data back for visualization
    weights_original = scaler.inverse_transform(som.weights)
    som.weights = weights_original  # Temporarily replace for visualization
    
    # Visualize results
    print("Generating visualizations...")
    
    # Plot training history
    som.plot_training_history(X, feature_names=feature_names, figsize=(15, 6))
    plt.show()
    
    # Plot U-Matrix
    som.plot_umatrix()
    plt.show()
    
    # Plot component planes
    som.visualize_component_planes(feature_names=feature_names)
    plt.show()
    
        # Plot cluster distribution
    # We need to map original data to the SOM neurons 
    # after transforming weights back to original scale
    som.visualize_cluster_distribution(X)
    plt.show()
    
    # Create an animation of the training process
    # First, restore scaled weights for correct animation
    som.weights = scaler.transform(weights_original)
    animation = som.plot_animation(X_scaled, feature_names=feature_names)
    
    # Save the animation as a gif
    try:
        animation.save('som_training_animation.gif', writer='pillow', fps=2)
        print("Animation saved as 'som_training_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
        
    # Restore original scale weights for further use
    som.weights = weights_original
    
    # Map new customer data
    new_customers = np.array([
        [35, 65],  # Mid-age, mid-income
        [25, 100], # Young, high-income
        [60, 45],  # Older, lower-income
        [50, 75]   # Mid-to-older, mid-to-high income
    ])
    
    # Find which cluster each new customer belongs to
    new_customers_scaled = scaler.transform(new_customers)
    clusters = som.predict(new_customers_scaled)
    
    print("\nNew customer segmentation results:")
    for i, (customer, cluster) in enumerate(zip(new_customers, clusters)):
        print(f"Customer {i+1} (Age: {customer[0]}, Income: ${customer[1]}k) -> Segment {cluster+1}")
    
    print("Advanced example completed.\n")


# 3. Customer segmentation case study
def customer_segmentation_case_study():
    print("Running customer segmentation case study...")
    
    # Create a more realistic dataset with additional features
    # Age, Income, Purchase Frequency (times per month), Avg Purchase Value ($)
    customers = np.array([
        # Young, low income, frequent small purchases
        [24, 40, 8, 20],
        [22, 35, 10, 15],
        [26, 45, 7, 25],
        [23, 38, 9, 18],
        [25, 42, 8, 22],
        
        # Young, high income, less frequent but larger purchases
        [28, 85, 3, 80],
        [27, 90, 2, 100],
        [29, 95, 3, 90],
        [26, 80, 4, 75],
        [30, 100, 2, 110],
        
        # Middle-aged, middle income, moderate frequency and value
        [42, 65, 5, 50],
        [45, 70, 6, 45],
        [40, 60, 5, 55],
        [43, 68, 4, 60],
        [47, 72, 5, 48],
        
        # Older, high income, low frequency, high value
        [58, 110, 1, 200],
        [62, 120, 1, 250],
        [55, 100, 2, 180],
        [65, 115, 1, 220],
        [60, 105, 2, 190]
    ])
    
    feature_names = ['Age', 'Income (k$)', 'Purchase Frequency', 'Avg Purchase ($)']
    
    # Scale the data
    scaler = StandardScaler()
    customers_scaled = scaler.fit_transform(customers)
    
    # Create and train a 3x2 SOM (6 customer segments)
    print("Training SOM for customer segmentation...")
    som = SelfOrganizingMap(grid_shape=(3, 2), learning_rate=0.5, 
                          max_iterations=500, sigma_initial=1.5)
    som.fit(customers_scaled, epochs=2)
    
    # Get segment assignments for each customer
    segments = som.predict(customers_scaled)
    
    # Calculate segment profiles
    segment_profiles = {}
    for segment in range(som.num_neurons):
        segment_mask = (segments == segment)
        if np.any(segment_mask):
            segment_data = customers[segment_mask]
            segment_profiles[segment] = {
                'count': np.sum(segment_mask),
                'mean': np.mean(segment_data, axis=0),
                'std': np.std(segment_data, axis=0)
            }
    
    # Print segment profiles
    print("\nCustomer Segment Profiles:")
    for segment, profile in segment_profiles.items():
        print(f"\nSegment {segment+1} ({profile['count']} customers):")
        print("  Average characteristics:")
        for i, feature in enumerate(feature_names):
            print(f"  - {feature}: {profile['mean'][i]:.1f} ± {profile['std'][i]:.1f}")
    
    # Visualize segment profiles with radar charts
    num_features = len(feature_names)
    angles = np.linspace(0, 2*np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Normalize data for radar chart
    feature_max = np.max(customers, axis=0)
    feature_min = np.min(customers, axis=0)
    
    # Define a color map for segments
    colors = plt.cm.tab10(np.linspace(0, 1, len(segment_profiles)))
    
    for i, (segment, profile) in enumerate(segment_profiles.items()):
        values = (profile['mean'] - feature_min) / (feature_max - feature_min)
        values = np.concatenate([values, [values[0]]])  # Close the polygon
        
        ax.plot(angles, values, color=colors[i], linewidth=2, 
                label=f"Segment {segment+1} (n={profile['count']})")
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Add feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    ax.set_yticks([])  # Hide radial ticks
    
    # Add chart details
    plt.title('Customer Segments - Feature Profiles', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.show()
    
    # Visualize clustering results with dimensionality reduction
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Create a figure with two subplots for PCA and t-SNE
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA for visualization
        pca = PCA(n_components=2)
        customers_pca = pca.fit_transform(customers_scaled)
        
        axes[0].scatter(customers_pca[:, 0], customers_pca[:, 1], c=segments, cmap='tab10', s=100)
        for i, segment in enumerate(np.unique(segments)):
            segment_center = np.mean(customers_pca[segments == segment], axis=0)
            axes[0].text(segment_center[0], segment_center[1], f"S{segment+1}", 
                      fontsize=12, fontweight='bold', ha='center', va='center',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        axes[0].set_title('Customer Segments (PCA)', size=14)
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        customers_tsne = tsne.fit_transform(customers_scaled)
        
        axes[1].scatter(customers_tsne[:, 0], customers_tsne[:, 1], c=segments, cmap='tab10', s=100)
        for i, segment in enumerate(np.unique(segments)):
            segment_center = np.mean(customers_tsne[segments == segment], axis=0)
            axes[1].text(segment_center[0], segment_center[1], f"S{segment+1}", 
                      fontsize=12, fontweight='bold', ha='center', va='center',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        axes[1].set_title('Customer Segments (t-SNE)', size=14)
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("PCA/t-SNE visualization requires scikit-learn. Skipping...")
    
    print("Customer segmentation case study completed.\n")


if __name__ == "__main__":
    print("SOM EXAMPLES")
    print("=" * 50)
    
    # Run examples
    basic_example()
    advanced_example()
    customer_segmentation_case_study()
    
    print("\nAll examples completed successfully!")