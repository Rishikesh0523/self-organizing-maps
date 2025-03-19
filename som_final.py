import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
import os
import pandas as pd
import kagglehub  # Make sure to install this package

# --- Download and load the customer dataset from Kaggle ---
try:
    path = kagglehub.dataset_download("shrutimechlearn/customer-data")
    csv_file = os.path.join(path, "Mall_Customers.csv")
    print(f"Dataset downloaded to: {csv_file}")
    customer_df = pd.read_csv(csv_file)
    print(f"Successfully loaded dataset with {customer_df.shape[0]} customers")
except Exception as e:
    print(f"Error loading Kaggle dataset: {e}")
    print("Using fallback data instead")
    # Fallback data if Kaggle download fails
    customer_df = pd.DataFrame({
        'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30, 67, 35, 58, 24, 37, 22, 35, 20, 52, 35],
        'Annual_Income_(k$)': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 23, 23]
    })

# Use only 'Age' and 'Annual_Income_(k$)' columns for SOM
data = customer_df[['Age', 'Annual_Income_(k$)']].values

# Print basic statistics about the data
print(f"Data shape: {data.shape}")
print(f"Age range: {data[:, 0].min():.1f} to {data[:, 0].max():.1f}")
print(f"Annual Income range: ${data[:, 1].min():.1f}k to ${data[:, 1].max():.1f}k")

# --- Define the SOM parameters ---
grid_shape = (3, 3)  # Increased from (2,2) to (3,3) for better representation
num_neurons = grid_shape[0] * grid_shape[1]

# Initialize weights randomly within the data range
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)
weights = np.random.uniform(
    low=min_vals,
    high=max_vals,
    size=(num_neurons, 2)
)

# Positions for neurons in the grid (for neighborhood calculations)
positions = np.array([[i, j] for i in range(grid_shape[0]) for j in range(grid_shape[1])])

# --- SOM Learning parameters ---
initial_learning_rate = 0.5
learning_rate_decay = 0.9  # Learning rate will decay over epochs
initial_neighborhood_radius = 1.0
radius_decay = 0.9  # Neighborhood radius will decay over epochs

def neighborhood_factor(bmu_index, neuron_index, current_radius):
    # Use Gaussian neighborhood function instead of step function
    pos_bmu = positions[bmu_index]
    pos_neuron = positions[neuron_index]
    distance = np.sum((pos_bmu - pos_neuron) ** 2)  # Squared Euclidean distance
    return np.exp(-distance / (2 * (current_radius ** 2)))

def plot_som_state(weights, iteration_info, description):
    """
    Plot the current state of the SOM along with the provided description text.
    """
    plt.figure(figsize=(10, 8))
    # Plot data points - using Age and Income
    plt.scatter(data[:, 0], data[:, 1], c='blue', s=50, label='Customers', marker='o', alpha=0.5)
    # Plot each neuron (weight vector)
    plt.scatter(weights[:, 0], weights[:, 1], c='red', s=200, marker='s')
    for i, w in enumerate(weights):
        plt.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='darkred')
    
    # Draw grid connections
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            # Only connect adjacent neurons in the grid
            pos_i = positions[i]
            pos_j = positions[j]
            if np.sum(np.abs(pos_i - pos_j)) == 1:  # Manhattan distance = 1
                plt.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)
    
    plt.xlabel('Age')
    plt.ylabel('Annual Income (k$)')
    plt.title(f"SOM Learning: {iteration_info}", fontsize=14)
    plt.legend()
    plt.grid(True)
    
    # Add the descriptive text in the figure (at the bottom)
    plt.gcf().text(0.05, 0.02, description, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()

def plot_som_state_summary(ax, weights, iteration_info):
    """
    Plot the current state of the SOM on the provided axis (summary without detailed text).
    """
    # Plot data points with small markers and some transparency
    ax.scatter(data[:, 0], data[:, 1], c='blue', s=20, label='Customers', marker='o', alpha=0.3)
    
    # Plot neurons with larger markers
    ax.scatter(weights[:, 0], weights[:, 1], c='red', s=150, marker='s')
    for i, w in enumerate(weights):
        ax.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=10, color='darkred')
    
    # Draw grid connections between adjacent neurons only
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            pos_i = positions[i]
            pos_j = positions[j]
            if np.sum(np.abs(pos_i - pos_j)) == 1:  # Manhattan distance = 1
                ax.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_title(f"{iteration_info}", fontsize=12)
    ax.grid(True)

# --- Initial Descriptive Plot ---
initial_description = (
    "Self-Organizing Map (SOM) for Customer Segmentation\n\n"
    f"Dataset: {customer_df.shape[0]} customers with features Age and Annual Income.\n"
    "Blue circles represent customer data points.\n"
    "Red squares represent the SOM neurons with initial random weights.\n\n"
    "Process Overview:\n"
    "1. For each customer, compute Euclidean distances to each neuron:\n"
    "   d_i = ||x - w_i|| = sqrt(Σ(x_dim - w_dim,i)^2)\n"
    "2. Identify the BMU (Best Matching Unit - neuron with minimum distance).\n"
    "3. Update weights for BMU and its neighbors:\n"
    "   w(new) = w(old) + α * h(BMU,i) * (x - w(old))\n"
    "   Where h(BMU,i) is the neighborhood function\n\n"
    "This visualization demonstrates how customers can be segmented based on\n"
    "age and income profiles using a self-organizing neural network."
)

plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], c='blue', s=50, label='Customers', marker='o', alpha=0.5)
for i, w in enumerate(weights):
    plt.scatter(w[0], w[1], c='red', s=200, marker='s')
    plt.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='darkred')

# Draw grid connections (only between adjacent neurons)
for i in range(num_neurons):
    for j in range(i+1, num_neurons):
        pos_i = positions[i]
        pos_j = positions[j]
        if np.sum(np.abs(pos_i - pos_j)) == 1:  # Manhattan distance = 1
            plt.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)

plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title("Mall Customer Data and Initial SOM Grid", fontsize=14)
plt.gcf().text(0.05, 0.02, initial_description, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.legend()
plt.grid(True)
plt.show()

# --- Step-by-Step Training with Multiple Epochs ---
num_epochs = 8  # Number of epochs for training
snapshots = []  # To store snapshots: (epoch, iteration, copy of weights, detailed description)
iteration_total = 0

current_learning_rate = initial_learning_rate
current_radius = initial_neighborhood_radius

for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch} starting...")
    # Shuffle the data for each epoch
    np.random.shuffle(data)
    
    for i, x in enumerate(data):
        iteration_total += 1
        
        # Build the detailed description text for this iteration
        step_description = f"Epoch {epoch}, Customer {i+1}: {x}\n\n"
        step_description += "Step 1: Compute Euclidean distances:\n"
        step_description += r"$d_i = ||x - w_i|| = \sqrt{\Sigma(x_{dim} - w_{dim,i})^2}$" + "\n"
        
        # Calculate distances from this data point to each neuron
        distances = np.linalg.norm(weights - x, axis=1)
        for j, d in enumerate(distances):
            step_description += f"Distance to Neuron {j+1}: {d:.2f}\n"
        
        # Find the Best Matching Unit (BMU)
        step_description += "\nStep 2: Identify the BMU (neuron with the smallest distance):\n"
        bmu_index = np.argmin(distances)
        step_description += f"BMU is Neuron {bmu_index+1} (min distance = {distances[bmu_index]:.2f})\n"
        
        # Update the weights
        step_description += "\nStep 3: Update weights using:\n"
        step_description += r"$w_{new} = w_{old} + \alpha \cdot h(BMU,i) \cdot (x - w_{old})$" + "\n"
        step_description += f"Where α = {current_learning_rate:.3f} and h(BMU,i) is the neighborhood function\n\n"
        
        for j in range(num_neurons):
            h = neighborhood_factor(bmu_index, j, current_radius)
            if h > 0.01:  # Only update if neighborhood factor is significant
                update = current_learning_rate * h * (x - weights[j])
                step_description += f"Updating Neuron {j+1}: Change = {update}\n"
                weights[j] += update

        # Save a snapshot every n iterations (to avoid too many snapshots)
        if i % max(1, len(data) // 10) == 0 or i == len(data) - 1:
            snapshots.append((epoch, iteration_total, weights.copy(), step_description))
    
    # Decay learning rate and neighborhood radius after each epoch
    current_learning_rate *= learning_rate_decay
    current_radius *= radius_decay
    print(f"Epoch {epoch} completed. New learning rate: {current_learning_rate:.3f}, New radius: {current_radius:.3f}")

print("Training complete. Now showing summary visualizations.")

# --- Create a scrollable grid visualization ---
class ScrollableGrid:
    def __init__(self, snapshots, cols=3, rows_per_page=3):
        self.snapshots = snapshots
        self.cols = cols
        self.rows_per_page = rows_per_page
        self.current_page = 0
        self.total_pages = (len(snapshots) + (cols * rows_per_page) - 1) // (cols * rows_per_page)
        
        self.create_figure()
        
    def create_figure(self):
        # Create figure with extra space at bottom for navigation buttons
        self.fig = plt.figure(figsize=(16, self.rows_per_page * 4 + 1), constrained_layout=False)
        self.gs = GridSpec(self.rows_per_page + 1, self.cols, figure=self.fig)
        
        # Create navigation buttons
        prev_button_ax = self.fig.add_subplot(self.gs[self.rows_per_page, 0])
        next_button_ax = self.fig.add_subplot(self.gs[self.rows_per_page, self.cols - 1])
        page_info_ax = self.fig.add_subplot(self.gs[self.rows_per_page, 1:self.cols - 1])
        
        self.prev_button = Button(prev_button_ax, 'Previous')
        self.next_button = Button(next_button_ax, 'Next')
        self.prev_button.on_clicked(self.prev_page)
        self.next_button.on_clicked(self.next_page)
        
        # Page info text
        self.page_info = page_info_ax.text(0.5, 0.5, f"Page {self.current_page + 1}/{self.total_pages}",
                                          ha='center', va='center')
        page_info_ax.axis('off')
        
        self.fig.suptitle("SOM Training Process: Mall Customer Segmentation", fontsize=16)
        self.update_page()
        
    def update_page(self):
        # Clear previous plots
        for i in range(self.rows_per_page):
            for j in range(self.cols):
                if hasattr(self, f'ax_{i}_{j}'):
                    self.fig.delaxes(getattr(self, f'ax_{i}_{j}'))
        
        # Calculate indices for current page
        start_idx = self.current_page * (self.cols * self.rows_per_page)
        end_idx = min(start_idx + (self.cols * self.rows_per_page), len(self.snapshots))
        
        # Create new subplots for current page
        plot_idx = 0
        for i in range(self.rows_per_page):
            for j in range(self.cols):
                idx = start_idx + plot_idx
                if idx < end_idx:
                    setattr(self, f'ax_{i}_{j}', self.fig.add_subplot(self.gs[i, j]))
                    ax = getattr(self, f'ax_{i}_{j}')
                    epoch, it, snap_weights, _ = self.snapshots[idx]
                    plot_som_state_summary(ax, snap_weights, f"Epoch {epoch}, Iteration {it}")
                    plot_idx += 1
        
        # Update page info
        self.page_info.set_text(f"Page {self.current_page + 1}/{self.total_pages}")
        self.fig.canvas.draw_idle()
        
    def prev_page(self, event):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_page()
            
    def next_page(self, event):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_page()

# --- Single Page Summary Visualization ---
def create_final_visualization(weights_initial, weights_final):
    """
    Create a comprehensive visualization showing initial and final states of the SOM.
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Initial state
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(data[:, 0], data[:, 1], c='blue', s=50, label='Customers', marker='o', alpha=0.5)
    
    # Plot neurons with grid connections
    ax1.scatter(weights_initial[:, 0], weights_initial[:, 1], c='red', s=150, marker='s')
    for i, w in enumerate(weights_initial):
        ax1.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=10, color='darkred')
    
    # Draw grid connections
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            pos_i = positions[i]
            pos_j = positions[j]
            if np.sum(np.abs(pos_i - pos_j)) == 1:
                ax1.plot([weights_initial[i, 0], weights_initial[j, 0]],
                        [weights_initial[i, 1], weights_initial[j, 1]], 'k--', lw=1)
    
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Annual Income (k$)')
    ax1.set_title('Initial SOM State', fontsize=12)
    ax1.grid(True)
    
    # Plot 2: Final state
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(data[:, 0], data[:, 1], c='blue', s=50, label='Customers', marker='o', alpha=0.5)
    
    # Plot neurons with grid connections
    ax2.scatter(weights_final[:, 0], weights_final[:, 1], c='red', s=150, marker='s')
    for i, w in enumerate(weights_final):
        ax2.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=10, color='darkred')
    
    # Draw grid connections
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            pos_i = positions[i]
            pos_j = positions[j]
            if np.sum(np.abs(pos_i - pos_j)) == 1:
                ax2.plot([weights_final[i, 0], weights_final[j, 0]],
                        [weights_final[i, 1], weights_final[j, 1]], 'k--', lw=1)
    
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Annual Income (k$)')
    ax2.set_title('Final SOM State', fontsize=12)
    ax2.grid(True)
    
    # Plot 3: Customer Segments based on SOM
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Assign each customer to nearest neuron
    customer_segments = []
    for x in data:
        distances = np.linalg.norm(weights_final - x, axis=1)
        bmu = np.argmin(distances)
        customer_segments.append(bmu)
    
    # Create a scatter plot with points colored by their segment
    segment_colors = plt.cm.tab10(np.linspace(0, 1, num_neurons))
    
    for segment in range(num_neurons):
        segment_data = data[np.array(customer_segments) == segment]
        if len(segment_data) > 0:
            ax3.scatter(segment_data[:, 0], segment_data[:, 1], 
                      c=[segment_colors[segment]], label=f'Segment {segment+1}',
                      s=50, marker='o', alpha=0.7)
    
    # Plot the neurons as centroids
    ax3.scatter(weights_final[:, 0], weights_final[:, 1], c='black', s=200, marker='X')
    for i, w in enumerate(weights_final):
        ax3.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='white')
    
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Annual Income (k$)')
    ax3.set_title('Customer Segments', fontsize=12)
    ax3.grid(True)
    ax3.legend()
    
    # Plot 4: Segment Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate statistics for each segment
    segment_stats = []
    for segment in range(num_neurons):
        segment_data = data[np.array(customer_segments) == segment]
        if len(segment_data) > 0:
            avg_age = np.mean(segment_data[:, 0])
            avg_income = np.mean(segment_data[:, 1])
            count = len(segment_data)
            segment_stats.append((segment+1, count, avg_age, avg_income))
    
    # Display statistics as text
    stats_text = "Customer Segment Statistics:\n\n"
    stats_text += "Segment | Count | Avg Age | Avg Income\n"
    stats_text += "-" * 40 + "\n"
    
    for segment, count, avg_age, avg_income in segment_stats:
        stats_text += f"   {segment:<6} | {count:^5} | {avg_age:^7.1f} | ${avg_income:^7.1f}k\n"
    
    # Add explanation of the method
    stats_text += "\n\nSOM Customer Segmentation:\n\n"
    stats_text += "The Self-Organizing Map has grouped customers\n"
    stats_text += "with similar age and income profiles together,\n"
    stats_text += "creating natural segments in the data.\n\n"
    stats_text += "Each segment represents customers with similar\n"
    stats_text += "characteristics, which can be targeted with\n"
    stats_text += "specific marketing strategies."
    
    ax4.text(0, 1, stats_text, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    fig.suptitle('Mall Customer Data: SOM Segmentation Results', fontsize=16)
    plt.tight_layout()
    return fig

# --- Create a visualization showing weight changes during training ---
def plot_weight_trajectories():
    """
    Create a visualization showing how the weights of each neuron change during training
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot customer data points for reference
    ax.scatter(data[:, 0], data[:, 1], c='blue', s=15, marker='o', alpha=0.2, label='Customers')
    
    # Track each neuron's weight changes with lines and markers
    neuron_colors = plt.cm.tab10(np.linspace(0, 1, num_neurons))
    
    for neuron_idx in range(num_neurons):
        # Extract weight trajectories for this neuron
        weights_history = [snap_weights[neuron_idx] for _, _, snap_weights, _ in snapshots]
        weights_history = np.array(weights_history)
        
        # Plot the trajectory path
        ax.plot(weights_history[:, 0], weights_history[:, 1], 
               c=neuron_colors[neuron_idx], linestyle='-',
               label=f'Neuron {neuron_idx+1} Path')
        
        # Plot specific points along the trajectory (start, end, and some intermediate)
        ax.scatter(weights_history[0, 0], weights_history[0, 1], 
                  c=[neuron_colors[neuron_idx]], marker='o', s=100, alpha=0.7)
        ax.scatter(weights_history[-1, 0], weights_history[-1, 1], 
                  c=[neuron_colors[neuron_idx]], marker='s', s=100, alpha=0.7)
        
        # Add arrows to show direction of movement
        for i in range(0, len(weights_history)-1, max(1, len(weights_history)//5)):
            ax.annotate('',
                      xy=(weights_history[i+1, 0], weights_history[i+1, 1]),
                      xytext=(weights_history[i, 0], weights_history[i, 1]),
                      arrowprops=dict(arrowstyle='->', color=neuron_colors[neuron_idx], lw=1))
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_title('SOM Neuron Weight Trajectories During Training', fontsize=14)
    ax.grid(True)
    
    # Legend with smaller font and outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
             fancybox=True, shadow=True, ncol=5, fontsize=8)
    
    plt.tight_layout()
    return fig

# --- Execute the visualizations ---
print("Creating scrollable training visualization...")
grid = ScrollableGrid(snapshots, cols=3, rows_per_page=3)
plt.show()

print("Creating weight trajectory visualization...")
trajectory_fig = plot_weight_trajectories()
plt.show()

print("Creating final summary visualization...")
# Get initial and final weights
_, _, initial_weights, _ = snapshots[0]
_, _, final_weights, _ = snapshots[-1]
final_fig = create_final_visualization(initial_weights, final_weights)
plt.show()

# --- Create a 3D interactive visualization with Customer Spending Score ---
try:
    from mpl_toolkits.mplot3d import Axes3D
    
    # Check if the dataset has the 'Spending Score (1-100)' column
    if 'Spending Score (1-100)' in customer_df.columns:
        print("Creating 3D visualization with Spending Score...")
        
        # Create a new 3D visualization
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract data with age, income, and spending score
        data_3d = customer_df[['Age', 'Annual_Income_(k$)', 'Spending Score (1-100)']].values
        
        # Assign each customer to a segment based on the 2D SOM results
        customer_segments = []
        for i in range(len(data)):
            distances = np.linalg.norm(final_weights - data[i], axis=1)
            bmu = np.argmin(distances)
            customer_segments.append(bmu)
        
        # Plot points with colors based on segments
        segment_colors = plt.cm.tab10(np.linspace(0, 1, num_neurons))
        
        for segment in range(num_neurons):
            segment_indices = np.array(customer_segments) == segment
            if np.any(segment_indices):
                segment_data_3d = data_3d[segment_indices]
                ax.scatter(segment_data_3d[:, 0], segment_data_3d[:, 1], segment_data_3d[:, 2],
                          c=[segment_colors[segment]], label=f'Segment {segment+1}',
                          s=50, marker='o', alpha=0.7)
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Annual Income (k$)')
        ax.set_zlabel('Spending Score (1-100)')
        ax.set_title('Customer Segments in 3D: Age, Income, and Spending Score', fontsize=14)
        
        # Add a legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                 fancybox=True, shadow=True, ncol=5)
        
        plt.tight_layout()
        plt.show()
    else:
        print("Spending Score column not found in dataset, skipping 3D visualization.")
except ImportError as e:
    print(f"Could not create 3D visualization: {e}")

# --- Create an enhanced clustered plot with U-Matrix visualization ---
def create_umatrix_visualization(weights_final):
    """
    Create a U-Matrix visualization to show the distance between adjacent neurons,
    which helps in identifying cluster boundaries in the SOM.
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Regular customer segmentation
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Assign each customer to nearest neuron
    customer_segments = []
    for x in data:
        distances = np.linalg.norm(weights_final - x, axis=1)
        bmu = np.argmin(distances)
        customer_segments.append(bmu)
    
    # Create a scatter plot with points colored by their segment
    segment_colors = plt.cm.tab10(np.linspace(0, 1, num_neurons))
    
    for segment in range(num_neurons):
        segment_data = data[np.array(customer_segments) == segment]
        if len(segment_data) > 0:
            ax1.scatter(segment_data[:, 0], segment_data[:, 1], 
                      c=[segment_colors[segment]], label=f'Segment {segment+1}',
                      s=50, marker='o', alpha=0.7)
    
    # Plot the neurons as centroids
    ax1.scatter(weights_final[:, 0], weights_final[:, 1], c='black', s=200, marker='X')
    for i, w in enumerate(weights_final):
        ax1.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='white')
    
    # Draw grid connections
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            pos_i = positions[i]
            pos_j = positions[j]
            if np.sum(np.abs(pos_i - pos_j)) == 1:
                ax1.plot([weights_final[i, 0], weights_final[j, 0]],
                        [weights_final[i, 1], weights_final[j, 1]], 'k--', lw=1.5)
    
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Annual Income (k$)')
    ax1.set_title('Customer Segments', fontsize=14)
    ax1.grid(True)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3)
    
    # Plot 2: U-Matrix Visualization
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Calculate distances between adjacent neurons (U-Matrix values)
    rows, cols = grid_shape
    u_matrix = np.zeros((rows, cols))
    
    # Rearrange weights to match grid layout for easier visualization
    grid_weights = np.zeros((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            neuron_idx = i * cols + j
            grid_weights[i, j] = weights_final[neuron_idx]
    
    # Calculate average distance to neighbors for each neuron
    for i in range(rows):
        for j in range(cols):
            neighbors_dist = []
            # Check all adjacent neurons (horizontal, vertical)
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    dist = np.linalg.norm(grid_weights[i, j] - grid_weights[ni, nj])
                    neighbors_dist.append(dist)
            
            # Average distance to neighbors
            if neighbors_dist:
                u_matrix[i, j] = np.mean(neighbors_dist)
    
    # Create a meshgrid for visualization
    x = np.arange(0, cols, 1)
    y = np.arange(0, rows, 1)
    X, Y = np.meshgrid(x, y)
    
    # Plot U-Matrix as a heatmap
    c = ax2.pcolormesh(X, Y, u_matrix, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax2, label='Average Distance to Neighbors')
    
    # Overlay grid
    for i in range(rows+1):
        ax2.axhline(y=i-0.5, color='w', linestyle='-', alpha=0.3)
    for j in range(cols+1):
        ax2.axvline(x=j-0.5, color='w', linestyle='-', alpha=0.3)
    
    # Add neuron indices
    for i in range(rows):
        for j in range(cols):
            neuron_idx = i * cols + j + 1
            ax2.text(j, i, f'N{neuron_idx}', ha='center', va='center', color='white')
    
    ax2.set_xticks(np.arange(cols))
    ax2.set_yticks(np.arange(rows))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title('U-Matrix: Distance Between Adjacent Neurons', fontsize=14)
    
    # Add explanation text
    ax2.text(cols/2, rows+0.5, 
            "The U-Matrix shows distances between adjacent neurons.\n"
            "Darker colors indicate closer neurons (similar profiles).\n"
            "Lighter colors indicate larger distances (potential cluster boundaries).",
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7))
    
    fig.suptitle('SOM Analysis of Mall Customer Data', fontsize=16)
    plt.tight_layout()
    return fig

# --- Create analysis of customer distribution by segment ---
def create_customer_distribution_analysis():
    """
    Create visualizations showing the distribution of customers across segments
    and the characteristics of each segment.
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Get the final weights and assign customers to segments
    _, _, final_weights, _ = snapshots[-1]
    
    # Assign each customer to nearest neuron
    customer_segments = []
    for x in data:
        distances = np.linalg.norm(final_weights - x, axis=1)
        bmu = np.argmin(distances)
        customer_segments.append(bmu)
    
    # Plot 1: Segment Size Distribution
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Count customers per segment
    segment_counts = np.zeros(num_neurons)
    for segment in range(num_neurons):
        segment_counts[segment] = np.sum(np.array(customer_segments) == segment)
    
    # Create bar chart
    bars = ax1.bar(range(1, num_neurons + 1), segment_counts, color=plt.cm.tab10(np.linspace(0, 1, num_neurons)))
    
    # Add count labels on top of bars
    for i, count in enumerate(segment_counts):
        ax1.text(i + 1, count + 1, str(int(count)), ha='center', fontsize=10)
    
    ax1.set_xlabel('Segment Number')
    ax1.set_ylabel('Number of Customers')
    ax1.set_title('Customer Distribution Across Segments')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Average Age by Segment
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Calculate average age per segment
    segment_avg_ages = []
    for segment in range(num_neurons):
        segment_data = data[np.array(customer_segments) == segment]
        if len(segment_data) > 0:
            avg_age = np.mean(segment_data[:, 0])
            segment_avg_ages.append(avg_age)
        else:
            segment_avg_ages.append(0)
    
    # Create bar chart
    bars = ax2.bar(range(1, num_neurons + 1), segment_avg_ages, color=plt.cm.tab10(np.linspace(0, 1, num_neurons)))
    
    ax2.set_xlabel('Segment Number')
    ax2.set_ylabel('Average Age')
    ax2.set_title('Average Customer Age by Segment')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Average Income by Segment
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Calculate average income per segment
    segment_avg_incomes = []
    for segment in range(num_neurons):
        segment_data = data[np.array(customer_segments) == segment]
        if len(segment_data) > 0:
            avg_income = np.mean(segment_data[:, 1])
            segment_avg_incomes.append(avg_income)
        else:
            segment_avg_incomes.append(0)
    
    # Create bar chart
    bars = ax3.bar(range(1, num_neurons + 1), segment_avg_incomes, color=plt.cm.tab10(np.linspace(0, 1, num_neurons)))
    
    ax3.set_xlabel('Segment Number')
    ax3.set_ylabel('Average Income (k$)')
    ax3.set_title('Average Customer Income by Segment')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Plot 4: Segment Profiles Table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create a table with segment statistics
    segment_stats = []
    column_labels = ['Segment', 'Count', 'Avg Age', 'Avg Income', 'Profile']
    
    for segment in range(num_neurons):
        segment_data = data[np.array(customer_segments) == segment]
        if len(segment_data) > 0:
            count = len(segment_data)
            avg_age = np.mean(segment_data[:, 0])
            avg_income = np.mean(segment_data[:, 1])
            
            # Create a simple profile based on age and income
            age_profile = "Young" if avg_age < 30 else "Middle-aged" if avg_age < 50 else "Senior"
            income_profile = "Low" if avg_income < 40 else "Medium" if avg_income < 70 else "High"
            profile = f"{age_profile}, {income_profile} income"
            
            segment_stats.append([
                f"Segment {segment+1}",
                f"{count}",
                f"{avg_age:.1f}",
                f"${avg_income:.1f}k",
                profile
            ])
    
    # Create the table
    table = ax4.table(
        cellText=segment_stats,
        colLabels=column_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.1, 0.1, 0.15, 0.5]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax4.set_title('Customer Segment Profiles')
    
    fig.suptitle('Mall Customer Segmentation Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    return fig

# --- Create a class for interactive segment exploration ---
class SegmentExplorer:
    def __init__(self, data, weights):
        self.data = data
        self.weights = weights
        self.current_segment = 0
        self.num_segments = len(weights)
        
        # Assign customers to segments
        self.customer_segments = []
        for x in data:
            distances = np.linalg.norm(weights - x, axis=1)
            bmu = np.argmin(distances)
            self.customer_segments.append(bmu)
        
        self.create_figure()
        
    def create_figure(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = GridSpec(2, 2, figure=self.fig)
        
        # Create navigation buttons
        prev_button_ax = self.fig.add_axes([0.2, 0.01, 0.1, 0.05])
        next_button_ax = self.fig.add_axes([0.7, 0.01, 0.1, 0.05])
        self.prev_button = Button(prev_button_ax, 'Previous')
        self.next_button = Button(next_button_ax, 'Next')
        self.prev_button.on_clicked(self.prev_segment)
        self.next_button.on_clicked(self.next_segment)
        
        # Add segment counter text
        counter_ax = self.fig.add_axes([0.4, 0.01, 0.2, 0.05])
        counter_ax.axis('off')
        self.counter_text = counter_ax.text(0.5, 0.5, f"Segment {self.current_segment + 1}/{self.num_segments}", 
                                         ha='center', va='center', fontsize=12)
        
        self.fig.suptitle(f"Exploring Segment {self.current_segment + 1}", fontsize=16)
        self.update_display()
        
    def update_display(self):
        # Clear previous plots
        self.fig.clf()
        
        # Recreate the navigation elements
        prev_button_ax = self.fig.add_axes([0.2, 0.01, 0.1, 0.05])
        next_button_ax = self.fig.add_axes([0.7, 0.01, 0.1, 0.05])
        self.prev_button = Button(prev_button_ax, 'Previous')
        self.next_button = Button(next_button_ax, 'Next')
        self.prev_button.on_clicked(self.prev_segment)
        self.next_button.on_clicked(self.next_segment)
        
        counter_ax = self.fig.add_axes([0.4, 0.01, 0.2, 0.05])
        counter_ax.axis('off')
        self.counter_text = counter_ax.text(0.5, 0.5, f"Segment {self.current_segment + 1}/{self.num_segments}", 
                                         ha='center', va='center', fontsize=12)
        
        # Create the main subplot grid
        self.gs = GridSpec(2, 2, figure=self.fig, height_ratios=[3, 1], hspace=0.3, top=0.9, bottom=0.15)
        
        # Plot 1: Customers in this segment highlighted
        ax1 = self.fig.add_subplot(self.gs[0, :])
        
        # Plot all data with low alpha
        ax1.scatter(self.data[:, 0], self.data[:, 1], c='gray', s=50, marker='o', alpha=0.2)
        
        # Highlight segment customers
        segment_mask = np.array(self.customer_segments) == self.current_segment
        if np.any(segment_mask):
            segment_data = self.data[segment_mask]
            ax1.scatter(segment_data[:, 0], segment_data[:, 1], 
                      c='red', s=80, marker='o', alpha=0.8, 
                      label=f'Segment {self.current_segment+1} Customers')
            
            # Draw convex hull around segment if there are enough points
            if len(segment_data) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(segment_data)
                    hull_points = segment_data[hull.vertices]
                    # Close the polygon
                    hull_points = np.append(hull_points, [hull_points[0]], axis=0)
                    ax1.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.2, color='red')
                except Exception as e:
                    print(f"Could not draw convex hull: {e}")
        
        # Plot the neurons
        ax1.scatter(self.weights[:, 0], self.weights[:, 1], c='blue', s=150, marker='s', alpha=0.5)
        
        # Highlight the current segment's neuron
        ax1.scatter(self.weights[self.current_segment, 0], self.weights[self.current_segment, 1], 
                  c='red', s=250, marker='X', label='Segment Centroid')
        
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Annual Income (k$)')
        ax1.set_title(f'Customers in Segment {self.current_segment+1}', fontsize=14)
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Segment statistics
        ax2 = self.fig.add_subplot(self.gs[1, 0])
        
        # Calculate segment statistics
        segment_data = self.data[segment_mask] if np.any(segment_mask) else np.array([])
        
        if len(segment_data) > 0:
            count = len(segment_data)
            avg_age = np.mean(segment_data[:, 0])
            avg_income = np.mean(segment_data[:, 1])
            min_age = np.min(segment_data[:, 0])
            max_age = np.max(segment_data[:, 0])
            min_income = np.min(segment_data[:, 1])
            max_income = np.max(segment_data[:, 1])
            
            # Create a simple profile based on age and income
            age_profile = "Young" if avg_age < 30 else "Middle-aged" if avg_age < 50 else "Senior"
            income_profile = "Low" if avg_income < 40 else "Medium" if avg_income < 70 else "High"
            
            # Display statistics
            stats_text = (
                f"Segment {self.current_segment+1} Statistics:\n\n"
                f"Number of customers: {count}\n\n"
                f"Age range: {min_age:.1f} - {max_age:.1f}\n"
                f"Average age: {avg_age:.1f}\n\n"
                f"Income range: ${min_income:.1f}k - ${max_income:.1f}k\n"
                f"Average income: ${avg_income:.1f}k\n\n"
                f"Profile: {age_profile}, {income_profile} income"
            )
        else:
            stats_text = f"No customers in Segment {self.current_segment+1}"
        
        ax2.axis('off')
        ax2.text(0.05, 0.95, stats_text, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Plot 3: Marketing recommendations
        ax3 = self.fig.add_subplot(self.gs[1, 1])
        ax3.axis('off')
        
        if len(segment_data) > 0:
            # Provide marketing recommendations based on segment profile
            if avg_age < 30:
                if avg_income < 40:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• Target with budget-friendly products\n"
                        "• Mobile app promotions\n"
                        "• Social media campaigns\n"
                        "• Student discounts\n"
                        "• Loyalty programs with gradual benefits"
                    )
                elif avg_income < 70:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• Focus on quality-to-price ratio\n"
                        "• Trendy, mid-range products\n"
                        "• Career growth related offers\n"
                        "• Digital marketing campaigns\n"
                        "• Premium loyalty program"
                    )
                else:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• Premium products for young professionals\n"
                        "• Exclusive experience offerings\n"
                        "• Status-oriented marketing\n"
                        "• Early access to new products\n"
                        "• Luxury brands partnerships"
                    )
            elif avg_age < 50:
                if avg_income < 40:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• Family bundle offers\n"
                        "• Essential products with discounts\n"
                        "• Value-oriented messaging\n"
                        "• Practical loyalty benefits\n"
                        "• Physical store promotions"
                    )
                elif avg_income < 70:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• Family-oriented premium products\n"
                        "• Work-life balance messaging\n"
                        "• Mixed digital and traditional marketing\n"
                        "• Convenience-focused services\n"
                        "• Membership programs with family benefits"
                    )
                else:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• High-end family products\n"
                        "• Luxury home and lifestyle focus\n"
                        "• Exclusive family experiences\n"
                        "• Personal shopping assistance\n"
                        "• Premium brand partnerships"
                    )
            else:
                if avg_income < 40:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• Senior discounts\n"
                        "• Health and wellness products\n"
                        "• Traditional media campaigns\n"
                        "• In-store assistance emphasis\n"
                        "• Value and reliability messaging"
                    )
                elif avg_income < 70:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• Quality health and leisure products\n"
                        "• Retirement planning tie-ins\n"
                        "• Mixed digital and print marketing\n"
                        "• Travel and experience packages\n"
                        "• Customer service excellence focus"
                    )
                else:
                    recommendations = (
                        "Marketing Recommendations:\n\n"
                        "• Luxury health and wellness products\n"
                        "• High-end travel packages\n"
                        "• Exclusive retirement community partnerships\n"
                        "• Personal concierge services\n"
                        "• Legacy and heritage messaging"
                    )
        else:
            recommendations = "No recommendations for empty segment"
        
        ax3.text(0.05, 0.95, recommendations, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        self.fig.suptitle(f"Exploring Segment {self.current_segment + 1}", fontsize=16)
        self.fig.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjust layout to account for buttons
        self.fig.canvas.draw_idle()
        
    def prev_segment(self, event):
        if self.current_segment > 0:
            self.current_segment -= 1
            self.update_display()
            
    def next_segment(self, event):
        if self.current_segment < self.num_segments - 1:
            self.current_segment += 1
            self.update_display()

# --- Final Code Execution Section ---
# After training (snapshots have been collected)

try:
    # Create U-Matrix visualization
    print("Creating U-Matrix visualization...")
    _, _, final_weights, _ = snapshots[-1]
    umatrix_fig = create_umatrix_visualization(final_weights)
    plt.show()

    # Create customer distribution analysis
    print("Creating customer distribution analysis...")
    dist_fig = create_customer_distribution_analysis()
    plt.show()

    # Create interactive segment explorer
    print("Creating interactive segment explorer...")
    segment_explorer = SegmentExplorer(data, final_weights)
    plt.show()

    # Additional visualization if dataset has Gender information
    if 'Gender' in customer_df.columns:
        print("Creating gender-based analysis...")
        # Assign each customer to nearest neuron
        customer_segments = []
        for x in data:
            distances = np.linalg.norm(final_weights - x, axis=1)
            bmu = np.argmin(distances)
            customer_segments.append(bmu)
        
        # Create figure for gender-based analysis
        gender_fig = plt.figure(figsize=(14, 7))
        
        # Plot 1: Gender distribution by segment
        ax1 = gender_fig.add_subplot(1, 2, 1)
        
        # Count gender distribution in each segment
        segments = np.unique(customer_segments)
        male_counts = []
        female_counts = []
        
        for segment in segments:
            segment_indices = np.array(customer_segments) == segment
            segment_df = customer_df.iloc[segment_indices]
            
            male_count = len(segment_df[segment_df['Gender'] == 'Male'])
            female_count = len(segment_df[segment_df['Gender'] == 'Female'])
            
            male_counts.append(male_count)
            female_counts.append(female_count)
        
        # Create grouped bar chart
        x = np.arange(len(segments))
        width = 0.35
        
        ax1.bar(x - width/2, male_counts, width, label='Male', color='skyblue')
        ax1.bar(x + width/2, female_counts, width, label='Female', color='pink')
        
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Number of Customers')
        ax1.set_title('Gender Distribution by Segment')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Segment {s+1}' for s in segments])
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Plot 2: Segment scatter with gender encoding
        ax2 = gender_fig.add_subplot(1, 2, 2)
        
        # Plot data points with gender encoding
        males = customer_df['Gender'] == 'Male'
        females = customer_df['Gender'] == 'Female'
        
        # Plot male points
        ax2.scatter(data[males, 0], data[males, 1], c='skyblue', marker='s', 
                   s=60, label='Male', alpha=0.7)
        
        # Plot female points
        ax2.scatter(data[females, 0], data[females, 1], c='pink', marker='o', 
                   s=60, label='Female', alpha=0.7)
        
        # Plot the neurons
        ax2.scatter(final_weights[:, 0], final_weights[:, 1], c='red', s=200, marker='X', label='Segment Centers')
        
        # Draw grid connections
        for i in range(num_neurons):
            for j in range(i+1, num_neurons):
                pos_i = positions[i]
                pos_j = positions[j]
                if np.sum(np.abs(pos_i - pos_j)) == 1:
                    ax2.plot([final_weights[i, 0], final_weights[j, 0]],
                            [final_weights[i, 1], final_weights[j, 1]], 'k--', lw=1)
        
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Annual Income (k$)')
        ax2.set_title('Customer Segments with Gender Encoding')
        ax2.grid(True)
        ax2.legend()
        
        gender_fig.suptitle('Gender-Based Analysis of Customer Segments', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # Export a summary report if matplotlib has savefig
    # try:
    #     print("Generating summary report...")
    #     # Create a summary figure
    #     summary_fig = plt.figure(figsize=(11, 8.5))  # US Letter size
        
    #     # Create a text summary
    #     ax = summary_fig.add_subplot(1, 1, 1)
    #     ax.axis('off')
        
    #     # Calculate basic statistics for the report
    #     segment_count = {}
    #     segment_avg_age = {}
    #     segment_avg_income = {}
        
    #     # Assign each customer to nearest neuron
    #     customer_segments = []
    #     for x in data:
    #         distances = np.linalg.norm(final_weights - x, axis=1)
    #         bmu = np.argmin(distances)
    #         customer_segments.append(bmu)
        
    #     for segment in range(num_neurons):
    #         segment_mask = np.array(customer_segments) == segment
    #         segment_data = data[segment_mask]
            
    #         if len(segment_data) > 0:
    #             segment_count[segment] = len(segment_data)
    #             segment_avg_age[segment] = np.mean(segment_data[:, 0])
    #             segment_avg_income[segment] = np.mean(segment_data[:, 1])
    #         else:
    #             segment_count[segment] = 0
    #             segment_avg_age[segment] = 0
    #             segment_avg_income[segment] = 0
        
    #     # Create the report text
    #     report_text = "MALL CUSTOMER SEGMENTATION ANALYSIS\n"
    #     report_text += "==================================\n\n"
    #     report_text += f"Dataset: {customer_df.shape[0]} customers\n"
    #     report_text += f"Features: Age and Annual Income\n"
    #     report_text += f"SOM Grid Size: {grid_shape[0]}x{grid_shape[1]} ({num_neurons} segments)\n\n"
    #     report_text += "SEGMENT SUMMARY\n"
    #     report_text += "--------------\n\n"
        
    #     # Add segment statistics
    #     for segment in range(num_neurons):
    #         if segment_count[segment] > 0:
    #             age_profile = "Young" if segment_avg_age[segment] < 30 else "Middle-aged" if segment_avg_age[segment] < 50 else "Senior"
    #             income_profile = "Low" if segment_avg_income[segment] < 40 else "Medium" if segment_avg_income[segment] < 70 else "High"
                
    #             report_text += f"Segment {segment+1}:\n"
    #             report_text += f"  Customers: {segment_count[segment]}\n"
    #             report_text += f"  Average Age: {segment_avg_age[segment]:.1f}\n"
    #             report_text += f"  Average Income: ${segment_avg_income[segment]:.1f}k\n"
    #             report_text += f"  Profile: {age_profile}, {income_profile} income\n\n"
        
    #     # Add methodology explanation
    #     report_text += "\nMETHODOLOGY\n"
    #     report_text += "-----------\n\n"
    #     report_text += "Self-Organizing Maps (SOM) were used to identify natural segments in customer data.\n"
    #     report_text += "The algorithm works as follows:\n"
    #     report_text += "1. Initialize a grid of neurons with random weights\n"
    #     report_text += "2. For each customer data point:\n"
    #     report_text += "   a. Find the Best Matching Unit (BMU) - the neuron closest to the data point\n"
    #     report_text += "   b. Update the BMU and its neighbors to move closer to the data point\n"
    #     report_text += "3. Repeat for multiple epochs with decreasing learning rates\n"
    #     report_text += "4. Assign customers to their closest neurons to form segments\n\n"
        
    #     # Add conclusion
    #     report_text += "CONCLUSION\n"
    #     report_text += "----------\n\n"
    #     report_text += "The SOM algorithm successfully identified distinct customer segments based on\n"
    #     report_text += "age and income profiles. These segments can be used for targeted marketing\n"
    #     report_text += "strategies. The U-Matrix visualization helps identify segment boundaries and\n"
    #     report_text += "similarities between adjacent segments.\n\n"
        
    #     # Add date
    #     from datetime import datetime
    #     report_text += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
    #     ax.text(0.05, 0.95, report_text, va='top', fontsize=10, family='monospace')
        
    #     # Save the report
    #     summary_fig.tight_layout()
    #     summary_fig.savefig('mall_customer_som_report.pdf')
    #     summary_fig.savefig('mall_customer_som_report.png', dpi=300)
    #     print("Summary report saved as 'mall_customer_som_report.pdf' and 'mall_customer_som_report.png'")
    # except Exception as e:
    #     print(f"Could not save summary report: {e}")

except Exception as e:
    print(f"Error in visualization: {e}")
    import traceback
    traceback.print_exc()