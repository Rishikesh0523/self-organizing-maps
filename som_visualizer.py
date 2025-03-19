import numpy as np
import pygame
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Self Organizing Map (SOM) Class
# ----------------------------
class SelfOrganizingMap:
    def __init__(self, m, n, dim, n_iterations=1000, learning_rate=0.1, sigma=None):
        """
        Initialize the SOM grid.
        :param m: Number of rows in the grid.
        :param n: Number of columns in the grid.
        :param dim: Dimensionality of the input vectors.
        :param n_iterations: Total iterations for training.
        :param learning_rate: Starting learning rate.
        :param sigma: Starting neighborhood radius; defaults to max(m, n)/2.
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(m, n) / 2.0
        # Initialize the weight vectors randomly in [0, 1]
        self.weights = np.random.rand(m, n, dim)
        self.iteration = 0

    def _decay_radius(self, iteration):
        """Exponential decay of the neighborhood radius."""
        return self.sigma * np.exp(-iteration / self.n_iterations)

    def _decay_learning_rate(self, iteration):
        """Exponential decay of the learning rate."""
        return self.learning_rate * np.exp(-iteration / self.n_iterations)

    def _find_bmu(self, input_vector):
        """
        Identify the Best Matching Unit (BMU) for the given input_vector.
        Returns the grid index (i, j) of the BMU.
        """
        diff = self.weights - input_vector
        distances = np.linalg.norm(diff, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def step(self, data):
        """
        Perform one training iteration.
        :param data: The dataset (array of input vectors).
        :return: A tuple containing:
                 (iteration number, copy of weights, BMU index,
                  current learning rate, current radius, input sample)
        """
        if self.iteration >= self.n_iterations:
            return None  # Training finished

        # Pick a random input sample from the data
        sample_idx = np.random.randint(0, data.shape[0])
        input_vector = data[sample_idx]

        # Find the BMU (Best Matching Unit)
        bmu_index = self._find_bmu(input_vector)

        # Decay learning parameters
        radius = self._decay_radius(self.iteration)
        lr = self._decay_learning_rate(self.iteration)

        # Update weights for all nodes within the neighborhood
        for i in range(self.m):
            for j in range(self.n):
                # Calculate the distance in grid space between the node and the BMU
                grid_distance = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))
                if grid_distance <= radius:
                    # Gaussian neighborhood function
                    influence = np.exp(-(grid_distance ** 2) / (2 * (radius ** 2)))
                    # Update the weight vector
                    self.weights[i, j] += lr * influence * (input_vector - self.weights[i, j])

        self.iteration += 1
        return self.iteration, self.weights.copy(), bmu_index, lr, radius, input_vector, sample_idx

    def cluster_data(self, data):
        """
        Assign each data point to its corresponding BMU.
        :param data: The dataset (array of input vectors).
        :return: List of cluster assignments.
        """
        clusters = []
        for i in range(data.shape[0]):
            input_vector = data[i]
            bmu_index = self._find_bmu(input_vector)
            clusters.append((bmu_index[0], bmu_index[1]))
        return clusters

# ----------------------------
# Helper Functions
# ----------------------------
def generate_market_research_data():
    """
    Generate synthetic market research data about consumer spending habits.
    """
    # Number of consumers
    n_consumers = 1000

    # Create random data for different spending categories
    np.random.seed(42)  # For reproducibility
    
    # Create a DataFrame with spending habits across different categories
    data = {
        'Food': np.random.gamma(5, 1000, n_consumers),  # Food spending
        'Entertainment': np.random.gamma(2, 500, n_consumers),  # Entertainment spending
        'Clothing': np.random.gamma(3, 700, n_consumers),  # Clothing spending
        'Electronics': np.random.exponential(1000, n_consumers),  # Electronics spending
        'Travel': np.random.gamma(1, 1500, n_consumers),  # Travel spending
        'Housing': np.random.gamma(7, 1500, n_consumers)  # Housing spending
    }
    
    # Add income levels
    data['Income'] = np.random.gamma(10, 5000, n_consumers)
    
    # Add age
    data['Age'] = np.random.randint(18, 80, n_consumers)
    
    # Create consumer segments (hidden variables that influence spending)
    segments = np.random.choice(['Budget Conscious', 'Luxury Shopper', 'Tech Enthusiast', 
                                 'Homebody', 'Adventure Seeker'], n_consumers)
    
    # Modify spending based on segments
    for i, segment in enumerate(segments):
        if segment == 'Budget Conscious':
            # Reduce all spending
            for category in ['Food', 'Entertainment', 'Clothing', 'Electronics', 'Travel']:
                data[category][i] *= 0.7
            data['Housing'][i] *= 0.8
        elif segment == 'Luxury Shopper':
            # Increase clothing and travel
            data['Clothing'][i] *= 1.8
            data['Food'][i] *= 1.3
            data['Travel'][i] *= 1.5
        elif segment == 'Tech Enthusiast':
            # Increase electronics
            data['Electronics'][i] *= 2.0
            data['Entertainment'][i] *= 1.4
        elif segment == 'Homebody':
            # Increase housing, reduce travel
            data['Housing'][i] *= 1.4
            data['Food'][i] *= 1.2
            data['Travel'][i] *= 0.6
        elif segment == 'Adventure Seeker':
            # Increase travel and entertainment
            data['Travel'][i] *= 2.0
            data['Entertainment'][i] *= 1.7
            data['Housing'][i] *= 0.9
    
    df = pd.DataFrame(data)
    df['Segment'] = segments
    
    return df

def map_pca_points(data, input_points=None, width=800, height=800):
    """
    Apply PCA to reduce data to 2D for visualization.
    """
    from sklearn.decomposition import PCA
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    # Scale the PCA results to be in [0, 1] for easier visualization
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(reduced_data)
    
    # Map to screen coordinates
    screen_points = [(int(x * width), int((1 - y) * height)) for x, y in scaled_data]
    
    # If input points are provided, transform them too
    if input_points is not None:
        reduced_input = pca.transform(input_points)
        scaled_input = scaler.transform(reduced_input)
        input_screen_points = [(int(x * width), int((1 - y) * height)) for x, y in scaled_input]
        return screen_points, pca, scaler, input_screen_points
    
    return screen_points, pca, scaler

def get_segment_colors(segments):
    """
    Assign colors to different market segments.
    """
    color_map = {
        'Budget Conscious': (100, 100, 255),   # Blue
        'Luxury Shopper': (255, 100, 100),     # Red
        'Tech Enthusiast': (100, 255, 100),    # Green
        'Homebody': (255, 255, 100),           # Yellow
        'Adventure Seeker': (255, 100, 255)    # Purple
    }
    
    return [color_map[segment] for segment in segments]

def visualize_clusters(som, data, feature_names, original_df, pca, scaler, width=800, height=800):
    """
    Create a visualization of the SOM clusters.
    """
    # Get cluster assignments
    clusters = som.cluster_data(data)
    
    # Count consumers in each cluster
    cluster_counts = {}
    cluster_segments = {}
    
    for i, (row, col) in enumerate(clusters):
        if (row, col) not in cluster_counts:
            cluster_counts[(row, col)] = 0
            cluster_segments[(row, col)] = {}
        
        cluster_counts[(row, col)] += 1
        
        segment = original_df.iloc[i]['Segment']
        if segment not in cluster_segments[(row, col)]:
            cluster_segments[(row, col)][segment] = 0
        cluster_segments[(row, col)][segment] += 1
    
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    # Draw the SOM grid
    for i in range(som.m):
        for j in range(som.n):
            # Calculate the size of the circle based on number of consumers
            count = cluster_counts.get((i, j), 0)
            size = max(100, count * 5)  # Minimum size 100, scale by count
            
            # Get the node's weight vector
            weight = som.weights[i, j]
            
            # Transform weight vector to 2D using the same PCA model
            weight_2d = pca.transform([weight])[0]
            weight_2d_scaled = scaler.transform([weight_2d])[0]
            
            # Map to screen coordinates
            x = weight_2d_scaled[0] * width
            y = (1 - weight_2d_scaled[1]) * height
            
            # Draw the node
            circle = plt.Circle((x, y), np.sqrt(size), alpha=0.7, 
                               fill=True if count > 0 else False, 
                               edgecolor='black', linewidth=1, 
                               facecolor='lightgray' if count > 0 else 'none')
            plt.gca().add_patch(circle)
            
            # Add count label if there are consumers in this node
            if count > 0:
                plt.text(x, y, str(count), ha='center', va='center', fontweight='bold')
                
                # Add the most common segment
                if cluster_segments.get((i, j)):
                    dominant_segment = max(cluster_segments[(i, j)].items(), key=lambda x: x[1])[0]
                    plt.text(x, y + 15, dominant_segment, ha='center', va='center', fontsize=8)
    
    # Draw connections between adjacent nodes
    for i in range(som.m):
        for j in range(som.n):
            # Get current node's weight
            weight = som.weights[i, j]
            weight_2d = pca.transform([weight])[0]
            weight_2d_scaled = scaler.transform([weight_2d])[0]
            x1, y1 = weight_2d_scaled[0] * width, (1 - weight_2d_scaled[1]) * height
            
            # Connect to right neighbor
            if j < som.n - 1:
                weight_right = som.weights[i, j+1]
                weight_right_2d = pca.transform([weight_right])[0]
                weight_right_2d_scaled = scaler.transform([weight_right_2d])[0]
                x2, y2 = weight_right_2d_scaled[0] * width, (1 - weight_right_2d_scaled[1]) * height
                plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)
            
            # Connect to bottom neighbor
            if i < som.m - 1:
                weight_bottom = som.weights[i+1, j]
                weight_bottom_2d = pca.transform([weight_bottom])[0]
                weight_bottom_2d_scaled = scaler.transform([weight_bottom_2d])[0]
                x2, y2 = weight_bottom_2d_scaled[0] * width, (1 - weight_bottom_2d_scaled[1]) * height
                plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)
    
    # Set plot properties
    plt.title('SOM Clustering of Consumer Spending Habits')
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(alpha=0.3)
    
    # Add legend for segments
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightgray', edgecolor='black', label='Consumer Cluster')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('som_clusters.png')
    plt.close()

    # Now create feature heat maps
    feature_figs = []
    for feat_idx, feature in enumerate(feature_names):
        plt.figure(figsize=(8, 6))
        plt.title(f'SOM Heatmap for {feature}')
        
        # Extract feature values from weights
        feature_grid = np.zeros((som.m, som.n))
        for i in range(som.m):
            for j in range(som.n):
                feature_grid[i, j] = som.weights[i, j, feat_idx]
        
        # Create a heatmap
        plt.imshow(feature_grid, cmap='viridis')
        plt.colorbar(label=feature)
        plt.xticks(range(som.n))
        plt.yticks(range(som.m))
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'som_heatmap_{feature}.png')
        plt.close()
    
    print(f"Visualizations saved to som_clusters.png and som_heatmap_*.png")

# ----------------------------
# Main Pygame Visualization Loop
# ----------------------------
def main():
    # Initialize Pygame and create a window
    pygame.init()
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Consumer Spending Habits SOM Visualization")
    clock = pygame.time.Clock()

    # Set up a font for rendering text
    font = pygame.font.SysFont("Arial", 20)
    small_font = pygame.font.SysFont("Arial", 16)

    # Generate market research data and prepare for SOM
    df = generate_market_research_data()
    
    # Select features for analysis
    features = ['Food', 'Entertainment', 'Clothing', 'Electronics', 'Travel', 'Housing', 'Income', 'Age']
    
    # Scale the data to [0, 1] range for SOM input
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[features].values)
    
    # Reduce dimensionality to 2D for visualization
    data_points, pca, pca_scaler = map_pca_points(data_scaled, width=width, height=height)
    
    # Get colors based on segments
    segment_colors = get_segment_colors(df['Segment'])

    # Initialize the SOM with desired parameters
    grid_rows, grid_cols = 10, 10
    iterations = 5000
    learning_rate = 0.5
    sigma = 5.0
    som = SelfOrganizingMap(grid_rows, grid_cols, len(features), n_iterations=iterations,
                            learning_rate=learning_rate, sigma=sigma)

    training_done = False
    current_sample_idx = None

    # Main loop
    while True:
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Generate final visualizations before quitting
                if som.iteration > 0:  # Only if we've done some training
                    visualize_clusters(som, data_scaled, features, df, pca, pca_scaler, width, height)
                pygame.quit()
                sys.exit()

        # If training is not finished, perform one step
        if not training_done:
            result = som.step(data_scaled)
            if result is None:
                training_done = True
                # Generate final visualizations once training is complete
                visualize_clusters(som, data_scaled, features, df, pca, pca_scaler, width, height)
            else:
                iteration, weights, bmu_index, current_lr, current_radius, input_vector, sample_idx = result
                current_sample_idx = sample_idx

                # Clear screen with white background
                screen.fill((255, 255, 255))

                # Draw the data points colored by segment
                for idx, (pt, color) in enumerate(zip(data_points, segment_colors)):
                    # Make current sample point larger
                    if idx == current_sample_idx:
                        pygame.draw.circle(screen, color, pt, 6)
                    else:
                        pygame.draw.circle(screen, color, pt, 3)

                # Transform SOM grid weights to 2D using the same PCA transformation
                grid_points = []
                for i in range(som.m):
                    for j in range(som.n):
                        weight = weights[i, j]
                        weight_2d = pca.transform([weight])[0]
                        weight_2d_scaled = pca_scaler.transform([weight_2d])[0]
                        x = int(weight_2d_scaled[0] * width)
                        y = int((1 - weight_2d_scaled[1]) * height)
                        grid_points.append((i, j, x, y))

                # Draw the SOM grid lines
                for i in range(som.m):
                    row_points = []
                    for j in range(som.n):
                        for gi, gj, gx, gy in grid_points:
                            if gi == i and gj == j:
                                row_points.append((gx, gy))
                                break
                    if len(row_points) > 1:
                        pygame.draw.lines(screen, (255, 0, 0), False, row_points, 2)
                
                for j in range(som.n):
                    col_points = []
                    for i in range(som.m):
                        for gi, gj, gx, gy in grid_points:
                            if gi == i and gj == j:
                                col_points.append((gx, gy))
                                break
                    if len(col_points) > 1:
                        pygame.draw.lines(screen, (255, 0, 0), False, col_points, 2)

                # Highlight the BMU with a blue circle
                for gi, gj, gx, gy in grid_points:
                    if (gi, gj) == bmu_index:
                        pygame.draw.circle(screen, (0, 0, 255), (gx, gy), 8)
                        break

                # Render descriptive text
                text_iter = font.render(f"Iteration: {iteration}/{som.n_iterations}", True, (0, 0, 0))
                text_lr = font.render(f"Learning Rate: {current_lr:.4f}", True, (0, 0, 0))
                text_radius = font.render(f"Radius: {current_radius:.4f}", True, (0, 0, 0))
                text_bmu = font.render(f"BMU Index: {bmu_index}", True, (0, 0, 0))
                
                # Show selected sample details
                current_customer = df.iloc[sample_idx]
                text_segment = font.render(f"Customer Segment: {current_customer['Segment']}", True, (0, 0, 0))
                
                # Display key spending values
                y_offset = 180
                for feature in features:
                    value = current_customer[feature]
                    text_feature = small_font.render(f"{feature}: ${value:.2f}" if feature not in ['Age'] else f"{feature}: {value:.0f}", True, (0, 0, 0))
                    screen.blit(text_feature, (10, y_offset))
                    y_offset += 25

                # Create a legend for segments
                legend_y = 450
                legend_title = font.render("Customer Segments:", True, (0, 0, 0))
                screen.blit(legend_title, (10, legend_y))
                
                segment_types = ['Budget Conscious', 'Luxury Shopper', 'Tech Enthusiast', 'Homebody', 'Adventure Seeker']
                segment_colors_map = get_segment_colors(segment_types)
                
                for i, (segment, color) in enumerate(zip(segment_types, segment_colors_map)):
                    pygame.draw.circle(screen, color, (20, legend_y + 30 + i*25), 6)
                    text_segment_legend = small_font.render(segment, True, (0, 0, 0))
                    screen.blit(text_segment_legend, (35, legend_y + 25 + i*25))

                # Blit the text surfaces onto the screen
                screen.blit(text_iter, (10, 10))
                screen.blit(text_lr, (10, 40))
                screen.blit(text_radius, (10, 70))
                screen.blit(text_bmu, (10, 100))
                screen.blit(text_segment, (10, 130))

                # Update the display
                pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

if __name__ == "__main__":
    main()