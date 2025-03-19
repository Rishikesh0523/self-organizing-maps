import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


class SelfOrganizingMap:
    """
    Self-Organizing Map implementation for customer segmentation.
    """
    def __init__(self, grid_shape=(2, 2), learning_rate=0.5, max_iterations=100, 
                 sigma_initial=1.0, decay_factor=0.9):
        """
        Initialize the SOM with specified parameters.
        
        Parameters:
        -----------
        grid_shape : tuple
            Shape of the SOM grid (height, width)
        learning_rate : float
            Initial learning rate for weight updates
        max_iterations : int
            Maximum number of training iterations
        sigma_initial : float
            Initial neighborhood radius
        decay_factor : float
            Decay factor for learning rate and neighborhood radius
        """
        self.grid_shape = grid_shape
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.sigma_initial = sigma_initial
        self.sigma = sigma_initial
        self.decay_factor = decay_factor
        self.num_neurons = grid_shape[0] * grid_shape[1]
        
        # Initialize neuron positions in the grid
        self.positions = np.array([
            [i, j] for i in range(grid_shape[0]) for j in range(grid_shape[1])
        ])
        
        # Weights will be initialized when data is provided
        self.weights = None
        self.data_scaler = None
        
        # Store training history
        self.history = []
        
    def _initialize_weights(self, data):
        """
        Initialize weights based on data characteristics.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data for training
        """
        # Option 1: Random initialization within data range
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        self.weights = np.random.uniform(
            low=data_min, high=data_max, 
            size=(self.num_neurons, data.shape[1])
        )
        
        # Option 2: Initialize with PCA (for larger datasets)
        # Commented out as it's overkill for the simple example
        """
        if data.shape[0] >= self.num_neurons:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pc = pca.fit_transform(data)
            pc_min, pc_max = pc.min(axis=0), pc.max(axis=0)
            
            # Create a meshgrid of points in PC space
            x = np.linspace(pc_min[0], pc_max[0], self.grid_shape[1])
            y = np.linspace(pc_min[1], pc_max[1], self.grid_shape[0])
            xx, yy = np.meshgrid(x, y)
            
            # Map these points back to input space
            mesh_pc = np.column_stack([xx.ravel(), yy.ravel()])
            self.weights = pca.inverse_transform(mesh_pc)
        """
    
    def _find_bmu(self, x):
        """
        Find the Best Matching Unit (BMU) for input x.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input data point
            
        Returns:
        --------
        int
            Index of the BMU
        """
        distances = np.linalg.norm(self.weights - x, axis=1)
        return np.argmin(distances)
    
    def _decay_parameters(self, iteration):
        """
        Update learning rate and sigma based on current iteration.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        """
        # Exponential decay
        progress = iteration / self.max_iterations
        self.learning_rate = self.initial_learning_rate * np.exp(-progress)
        self.sigma = self.sigma_initial * np.exp(-progress)
    
    def _calculate_neighborhood(self, bmu_index):
        """
        Calculate neighborhood factors for all neurons given BMU.
        
        Parameters:
        -----------
        bmu_index : int
            Index of the BMU
            
        Returns:
        --------
        numpy.ndarray
            Array of neighborhood factors for each neuron
        """
        # Calculate Gaussian neighborhood factors
        bmu_position = self.positions[bmu_index]
        squared_distances = np.sum(np.square(self.positions - bmu_position), axis=1)
        return np.exp(-squared_distances / (2 * self.sigma**2))
    
    def fit(self, data, epochs=1, shuffle=True, verbose=True):
        """
        Train the SOM model on the provided data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Training data
        epochs : int
            Number of epochs to train
        shuffle : bool
            Whether to shuffle data between epochs
        verbose : bool
            Whether to print progress information
        
        Returns:
        --------
        self
        """
        # Initialize weights if not already done
        if self.weights is None:
            self._initialize_weights(data)
        
        # Store initial state
        self.history.append({
            'iteration': 0, 
            'weights': self.weights.copy(),
            'description': "Initial weights"
        })
        
        iteration = 0
        for epoch in range(epochs):
            indices = np.arange(len(data))
            if shuffle:
                np.random.shuffle(indices)
                
            for idx in indices:
                iteration += 1
                x = data[idx]
                
                # Find BMU
                bmu_index = self._find_bmu(x)
                
                # Calculate neighborhood factors
                neighborhood = self._calculate_neighborhood(bmu_index)
                
                # Update weights
                for i in range(self.num_neurons):
                    if neighborhood[i] > 0.01:  # Skip negligible updates
                        self.weights[i] += self.learning_rate * neighborhood[i] * (x - self.weights[i])
                
                # Save state for visualization
                if iteration % (len(data) // 4 + 1) == 0 or iteration == 1:
                    self.history.append({
                        'iteration': iteration,
                        'weights': self.weights.copy(),
                        'bmu': bmu_index,
                        'input': x,
                        'neighborhood': neighborhood.copy()
                    })
                
                # Decay learning rate and neighborhood radius
                self._decay_parameters(iteration)
                
                if verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}/{epochs * len(data)}")
        
        return self
    
    def transform(self, data):
        """
        Map data points to their BMU indices.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to map
            
        Returns:
        --------
        numpy.ndarray
            BMU indices for each data point
        """
        bmu_indices = np.zeros(len(data), dtype=int)
        for i, x in enumerate(data):
            bmu_indices[i] = self._find_bmu(x)
        return bmu_indices
    
    def predict(self, data):
        """
        Predict cluster labels using the trained SOM.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to cluster
            
        Returns:
        --------
        numpy.ndarray
            Cluster labels (BMU indices)
        """
        return self.transform(data)
    
    def plot_training_history(self, data, feature_names=None, figsize=(10, 6)):
        """
        Plot the training history of the SOM.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Original training data
        feature_names : list
            Names of features (for axis labels)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if feature_names is None:
            if data.shape[1] == 2:
                feature_names = ['Feature 1', 'Feature 2']
            else:
                feature_names = [f'Feature {i+1}' for i in range(data.shape[1])]
        
        num_steps = len(self.history)
        fig, axes = plt.subplots(1, num_steps, figsize=figsize, 
                                 sharex=True, sharey=True)
        
        if num_steps == 1:
            axes = [axes]
        
        for i, snapshot in enumerate(self.history):
            ax = axes[i]
            weights = snapshot['weights']
            
            # Plot data points
            ax.scatter(data[:, 0], data[:, 1], c='blue', s=100, 
                       label='Data Points', marker='o')
            
            # Plot neurons
            for j, w in enumerate(weights):
                ax.scatter(w[0], w[1], c='red', s=200, marker='s')
                ax.text(w[0] + 0.5, w[1] + 0.5, f'N{j+1}', 
                        fontsize=12, color='darkred')
            
            # Draw grid connections
            for y in range(self.grid_shape[0]):
                for x in range(self.grid_shape[1]):
                    idx = y * self.grid_shape[1] + x
                    
                    # Connect horizontally if not at right edge
                    if x < self.grid_shape[1] - 1:
                        right_idx = idx + 1
                        ax.plot([weights[idx, 0], weights[right_idx, 0]],
                                [weights[idx, 1], weights[right_idx, 1]],
                                'k--', lw=1)
                    
                    # Connect vertically if not at bottom edge
                    if y < self.grid_shape[0] - 1:
                        down_idx = idx + self.grid_shape[1]
                        ax.plot([weights[idx, 0], weights[down_idx, 0]],
                                [weights[idx, 1], weights[down_idx, 1]],
                                'k--', lw=1)
            
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_title(f"Iteration: {snapshot['iteration']}")
            ax.grid(True)
            
            if i == 0:
                ax.legend()
        
        fig.suptitle("SOM Training Process", fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_animation(self, data, feature_names=None, figsize=(8, 6), interval=500):
        """
        Create an animated visualization of the SOM training process.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Original training data
        feature_names : list
            Names of features (for axis labels)
        figsize : tuple
            Figure size
        interval : int
            Animation interval in milliseconds
            
        Returns:
        --------
        matplotlib.animation.FuncAnimation
        """
        if feature_names is None:
            if data.shape[1] == 2:
                feature_names = ['Feature 1', 'Feature 2']
            else:
                feature_names = [f'Feature {i+1}' for i in range(data.shape[1])]
                
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set limits with some padding
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        padding = 0.1 * (data_max - data_min)
        ax.set_xlim(data_min[0] - padding[0], data_max[0] + padding[0])
        ax.set_ylim(data_min[1] - padding[1], data_max[1] + padding[1])
        
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.grid(True)
        
        # Plot data points (static)
        ax.scatter(data[:, 0], data[:, 1], c='blue', s=100, label='Data Points', marker='o')
        ax.legend()
        
        # Initialize neurons (will be updated)
        neuron_points = ax.scatter([], [], c='red', s=200, marker='s')
        neuron_labels = []
        
        # Initialize grid lines (will be updated)
        grid_lines = []
        for _ in range(self.grid_shape[0] * self.grid_shape[1] * 2):
            line, = ax.plot([], [], 'k--', lw=1)
            grid_lines.append(line)
        
        # Initialize title
        title = ax.set_title("Iteration: 0")
        
        def init():
            neuron_points.set_offsets(np.empty((0, 2)))
            for line in grid_lines:
                line.set_data([], [])
            return [neuron_points] + grid_lines
        
        def update(frame_idx):
            if frame_idx >= len(self.history):
                return [neuron_points] + grid_lines
            
            snapshot = self.history[frame_idx]
            weights = snapshot['weights']
            
            # Update neuron positions
            neuron_points.set_offsets(weights[:, :2])
            
            # Clear old text labels
            for label in neuron_labels:
                label.remove()
            neuron_labels.clear()
            
            # Add new text labels
            for j, w in enumerate(weights):
                label = ax.text(w[0] + 0.5, w[1] + 0.5, f'N{j+1}', 
                               fontsize=12, color='darkred')
                neuron_labels.append(label)
            
            # Update grid lines
            line_idx = 0
            for y in range(self.grid_shape[0]):
                for x in range(self.grid_shape[1]):
                    idx = y * self.grid_shape[1] + x
                    
                    # Connect horizontally if not at right edge
                    if x < self.grid_shape[1] - 1:
                        right_idx = idx + 1
                        grid_lines[line_idx].set_data(
                            [weights[idx, 0], weights[right_idx, 0]],
                            [weights[idx, 1], weights[right_idx, 1]]
                        )
                        line_idx += 1
                    
                    # Connect vertically if not at bottom edge
                    if y < self.grid_shape[0] - 1:
                        down_idx = idx + self.grid_shape[1]
                        grid_lines[line_idx].set_data(
                            [weights[idx, 0], weights[down_idx, 0]],
                            [weights[idx, 1], weights[down_idx, 1]]
                        )
                        line_idx += 1
            
            # Update title
            title.set_text(f"Iteration: {snapshot['iteration']}")
            
            return [neuron_points] + grid_lines + neuron_labels
        
        ani = FuncAnimation(fig, update, frames=len(self.history),
                             init_func=init, blit=True, interval=interval)
        plt.close()  # Prevent display of the static figure
        return ani
    
    def plot_umatrix(self, figsize=(8, 6)):
        """
        Plot the U-Matrix (Unified Distance Matrix) showing neuron distances.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.weights is None:
            raise ValueError("SOM must be trained before plotting U-Matrix")
        
        # Calculate distance between each neuron and its neighbors
        umatrix = np.zeros(self.grid_shape)
        
        for y in range(self.grid_shape[0]):
            for x in range(self.grid_shape[1]):
                idx = y * self.grid_shape[1] + x
                distances = []
                
                # Check neighbors (up, down, left, right)
                neighbors = []
                if y > 0:  # up
                    neighbors.append((y-1) * self.grid_shape[1] + x)
                if y < self.grid_shape[0] - 1:  # down
                    neighbors.append((y+1) * self.grid_shape[1] + x)
                if x > 0:  # left
                    neighbors.append(y * self.grid_shape[1] + (x-1))
                if x < self.grid_shape[1] - 1:  # right
                    neighbors.append(y * self.grid_shape[1] + (x+1))
                
                # Calculate average distance to neighbors
                for neighbor_idx in neighbors:
                    distances.append(np.linalg.norm(
                        self.weights[idx] - self.weights[neighbor_idx]))
                
                umatrix[y, x] = np.mean(distances) if distances else 0
        
        # Plot the U-Matrix
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(umatrix, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Distance to Neighbors')
        
        # Add neuron indices
        for y in range(self.grid_shape[0]):
            for x in range(self.grid_shape[1]):
                idx = y * self.grid_shape[1] + x
                ax.text(x, y, f'{idx+1}', ha='center', va='center', 
                        color='white', fontweight='bold')
        
        ax.set_xticks(np.arange(self.grid_shape[1]))
        ax.set_yticks(np.arange(self.grid_shape[0]))
        ax.set_title('U-Matrix (Unified Distance Matrix)')
        
        plt.tight_layout()
        return fig
    
    def visualize_component_planes(self, feature_names=None, figsize=(12, 8)):
        """
        Visualize the component planes of the SOM.
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.weights is None:
            raise ValueError("SOM must be trained before visualizing component planes")
        
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(self.weights.shape[1])]
        
        num_features = self.weights.shape[1]
        n_cols = 2
        n_rows = (num_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i in range(num_features):
            # Reshape weights for feature i to grid shape
            component = np.zeros(self.grid_shape)
            for y in range(self.grid_shape[0]):
                for x in range(self.grid_shape[1]):
                    idx = y * self.grid_shape[1] + x
                    component[y, x] = self.weights[idx, i]
            
            ax = axes[i]
            im = ax.imshow(component, cmap='coolwarm')
            plt.colorbar(im, ax=ax)
            
            # Add neuron indices
            for y in range(self.grid_shape[0]):
                for x in range(self.grid_shape[1]):
                    idx = y * self.grid_shape[1] + x
                    ax.text(x, y, f'{idx+1}', ha='center', va='center', 
                            color='black', fontweight='bold')
            
            ax.set_title(f'Component Plane: {feature_names[i]}')
            ax.set_xticks(np.arange(self.grid_shape[1]))
            ax.set_yticks(np.arange(self.grid_shape[0]))
            
        # Hide any unused subplots
        for i in range(num_features, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        return fig
    
    def visualize_cluster_distribution(self, data, figsize=(10, 6)):
        """
        Visualize the distribution of data points across SOM neurons.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to map
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.weights is None:
            raise ValueError("SOM must be trained before visualizing clusters")
        
        # Find BMU for each data point
        bmu_indices = self.transform(data)
        
        # Count data points per neuron
        counts = np.zeros(self.num_neurons, dtype=int)
        for bmu in bmu_indices:
            counts[bmu] += 1
        
        # Reshape counts to grid
        count_grid = counts.reshape(self.grid_shape)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(count_grid, cmap='YlOrRd')
        
        # Add counts as text
        for y in range(self.grid_shape[0]):
            for x in range(self.grid_shape[1]):
                idx = y * self.grid_shape[1] + x
                ax.text(x, y, f'N{idx+1}\n({counts[idx]})', 
                       ha='center', va='center', color='black', fontweight='bold')
        
        ax.set_title('Data Distribution Across SOM Neurons')
        ax.set_xticks(np.arange(self.grid_shape[1]))
        ax.set_yticks(np.arange(self.grid_shape[0]))
        
        plt.colorbar(im, ax=ax, label='Number of Data Points')
        plt.tight_layout()
        return fig
    
    def save(self, filename):
        """
        Save the SOM model to a file.
        
        Parameters:
        -----------
        filename : str
            File name to save the model
        """
        model_data = {
            'grid_shape': self.grid_shape,
            'learning_rate': self.initial_learning_rate,
            'sigma': self.sigma_initial,
            'weights': self.weights,
            'positions': self.positions,
            'decay_factor': self.decay_factor
        }
        np.save(filename, model_data)
    
    @classmethod
    def load(cls, filename):
        """
        Load a SOM model from a file.
        
        Parameters:
        -----------
        filename : str
            File name to load the model from
            
        Returns:
        --------
        SelfOrganizingMap
            Loaded SOM model
        """
        model_data = np.load(filename, allow_pickle=True).item()
        
        # Create new model with loaded parameters
        model = cls(
            grid_shape=model_data['grid_shape'],
            learning_rate=model_data['learning_rate'],
            sigma_initial=model_data['sigma'],
            decay_factor=model_data['decay_factor']
        )
        
        # Restore weights
        model.weights = model_data['weights']
        model.positions = model_data['positions']
        
        return model


# Example usage function to demonstrate the improved SOM
def demonstrate_improved_som():
    """
    Demonstrate the usage of the improved SelfOrganizingMap class.
    """
    # Create sample data - same as in the original code
    data = np.array([
        [25, 40],  # Customer A
        [45, 80],  # Customer B
        [30, 60],  # Customer C
        [50, 90]   # Customer D
    ])
    
    # Create and train the SOM
    som = SelfOrganizingMap(
        grid_shape=(2, 2),
        learning_rate=0.5,
        max_iterations=20,  # Increased to demonstrate learning
        sigma_initial=1.0
    )
    
    # Optional: Set initial weights manually for demonstration
    initial_weights = np.array([
        [30, 50],  # Neuron 1 (top-left)
        [40, 70],  # Neuron 2 (top-right)
        [20, 30],  # Neuron 3 (bottom-left)
        [50, 80]   # Neuron 4 (bottom-right)
    ], dtype=float)
    som.weights = initial_weights.copy()
    
    # Train the SOM
    print("Training SOM...")
    som.fit(data, epochs=2, verbose=True)
    
    # Visualize the training process as static plots
    feature_names = ['Age', 'Income (k$)']
    som.plot_training_history(data, feature_names=feature_names)
    plt.show()
    
    # Create an animation of the training process
    print("Generating training animation...")
    animation = som.plot_animation(data, feature_names=feature_names)
    
    # Save the animation (uncomment if you want to save)
    # animation.save('som_training.gif', writer='pillow', fps=2)
    
    # Show the U-Matrix
    print("Generating U-Matrix...")
    som.plot_umatrix()
    plt.show()
    
    # Visualize component planes
    print("Generating component planes...")
    som.visualize_component_planes(feature_names=feature_names)
    plt.show()
    
    # Visualize cluster distribution
    print("Generating cluster distribution...")
    som.visualize_cluster_distribution(data)
    plt.show()
    
    # Map new data points
    new_data = np.array([
        [27, 45],  # Similar to Customer A
        [48, 85],  # Similar to Customer B
        [35, 55]   # Between Customer A and C
    ])
    
    print("Mapping new data points...")
    cluster_indices = som.predict(new_data)
    
    for i, (point, cluster) in enumerate(zip(new_data, cluster_indices)):
        print(f"New customer {i+1} ({point[0]} yrs, ${point[1]}k): Mapped to Neuron {cluster+1}")
    
    # Save the trained model
    som.save("customer_som_model.npy")
    print("Model saved to 'customer_som_model.npy'")
    
    # Load the model back
    loaded_som = SelfOrganizingMap.load("customer_som_model.npy")
    print("Model loaded successfully!")
    
    # Verify the loaded model works
    test_prediction = loaded_som.predict(new_data)
    print("Prediction using loaded model matches:", np.array_equal(cluster_indices, test_prediction))


if __name__ == "__main__":
    demonstrate_improved_som()