import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

# --- Define our customer dataset (Age, Income, Purchase Frequency) ---
data = np.array([
    [25, 40, 3],  # Customer A: age 25, income 40k, purchases 3 times/month
    [45, 80, 1],  # Customer B: age 45, income 80k, purchases 1 time/month
    [30, 60, 5],  # Customer C: age 30, income 60k, purchases 5 times/month
    [50, 90, 2],   # Customer D: age 50, income 90k, purchases 2 times/month
    [60, 100, 1]   # Customer D: age 50, income 90k, purchases 2 times/month
])

# --- Define the SOM parameters ---
grid_shape = (2, 2)
num_neurons = grid_shape[0] * grid_shape[1]

# Initialize weights for the neurons (manually set for clarity)
weights = np.array([
    [30, 50, 3],  # Neuron 1 (top-left)
    [40, 70, 2],  # Neuron 2 (top-right)
    [20, 30, 4],  # Neuron 3 (bottom-left)
    [50, 80, 1]   # Neuron 4 (bottom-right)
], dtype=float)

# Positions for neurons in a 2x2 grid (for neighborhood calculations)
positions = np.array([
    [0, 0],   # Neuron 1
    [0, 1],   # Neuron 2
    [1, 0],   # Neuron 3
    [1, 1]    # Neuron 4
])

# --- SOM Learning parameters ---
learning_rate = 0.5

def neighborhood_factor(bmu_index, neuron_index):
    # Use Manhattan distance on the grid for this simple example.
    pos_bmu = positions[bmu_index]
    pos_neuron = positions[neuron_index]
    distance = np.sum(np.abs(pos_bmu - pos_neuron))
    if distance == 0:
        return 1.0  # BMU itself
    elif distance == 1:
        return 0.5  # direct neighbor
    else:
        return 0.0  # no update outside immediate neighborhood

def plot_som_state(weights, iteration_info, description):
    """
    Plot the current state of the SOM along with the provided description text.
    """
    plt.figure(figsize=(8, 6))
    # Plot data points - now we use Age and Income for the 2D visualization
    plt.scatter(data[:, 0], data[:, 1], c='blue', s=100, label='Data Points', marker='o')
    # Plot each neuron (weight vector) - projected to 2D space (Age, Income)
    for i, w in enumerate(weights):
        plt.scatter(w[0], w[1], c='red', s=200, marker='s')
        plt.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='darkred')
    # Draw grid connections for the 2x2 structure
    neuron_lines = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for i, j in neuron_lines:
        plt.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)
    plt.xlabel('Age')
    plt.ylabel('Income (k$)')
    plt.title(f"Iteration: {iteration_info}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xlim(10, 60)
    plt.ylim(20, 100)
    # Add the descriptive text in the figure (at the bottom)
    plt.gcf().text(0.05, 0.02, description, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    # plt.show()

def plot_som_state_summary(ax, weights, iteration_info):
    """
    Plot the current state of the SOM on the provided axis (summary without detailed text).
    """
    ax.scatter(data[:, 0], data[:, 1], c='blue', s=100, label='Data Points', marker='o')
    for i, w in enumerate(weights):
        ax.scatter(w[0], w[1], c='red', s=200, marker='s')
        ax.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='darkred')
    neuron_lines = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for i, j in neuron_lines:
        ax.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)
    ax.set_xlabel('Age')
    ax.set_ylabel('Income (k$)')
    ax.set_title(f"Iteration: {iteration_info}", fontsize=14)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(10, 60)
    ax.set_ylim(20, 100)

# --- Initial Descriptive Plot ---
initial_description = (
    "Initial Data Visualization\n\n"
    "Dataset: 4 customer data points with features Age, Income, and Purchase Frequency.\n"
    "Blue circles represent the data points (projected to 2D using Age and Income).\n"
    "Red squares represent the SOM neurons with initial weights (projected to 2D).\n\n"
    "Dimensionality Reduction: 3D data (Age, Income, Purchase Frequency) → 2D grid\n\n"
    "Process Overview:\n"
    "1. For each input, compute Euclidean distances to each neuron in 3D space:\n"
    "   d_i = ||x - w_i|| = sqrt(Σ(x_dim - w_dim,i)^2)\n"
    "2. Identify the BMU (neuron with minimum distance).\n"
    "3. Update weights:\n"
    "   w(new) = w(old) + α * h(BMU,i) * (x - w(old))\n"
    "   (where h(BMU,i)=1 for BMU, 0.5 for neighbors, 0 otherwise)\n\n"
    "Press Enter to start the training process."
)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c='blue', s=100, label='Data Points (2D projection)', marker='o')
for i, w in enumerate(weights):
    plt.scatter(w[0], w[1], c='red', s=200, marker='s')
    plt.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='darkred')
for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
    plt.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)
plt.xlabel('Age')
plt.ylabel('Income (k$)')
plt.title("Initial Data and SOM Setup (3D → 2D)", fontsize=14)
plt.gcf().text(0.05, 0.02, initial_description, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.legend()
plt.grid(True)
plt.xlim(10, 60)
plt.ylim(20, 100)
plt.show()

# --- Step-by-Step Training with Multiple Epochs ---
num_epochs = 8  # Set the desired number of epochs
snapshots = []  # To store snapshots: (epoch, iteration, copy of weights, detailed description)
iteration_total = 0

for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch} starting...")
    for x in data:
        iteration_total += 1
        # Build the detailed description text for this iteration including epoch info
        step_description = f"Epoch {epoch}, Input: {x}\n\n"
        step_description += "Step 1: Compute Euclidean distances in 3D space:\n"
        step_description += r"$d_i = ||x - w_i|| = \sqrt{\Sigma(x_{dim} - w_{dim,i})^2}$" + "\n"
        
        distances = np.linalg.norm(weights - x, axis=1)
        for i, d in enumerate(distances):
            step_description += f"Distance to Neuron {i+1}: {d:.2f}\n"
        
        step_description += "\nStep 2: Identify the BMU (neuron with the smallest distance):\n"
        bmu_index = np.argmin(distances)
        step_description += f"BMU is Neuron {bmu_index+1} (min distance = {distances[bmu_index]:.2f})\n"
        
        step_description += "\nStep 3: Update weights using:\n"
        step_description += r"$w_{new} = w_{old} + \alpha \cdot h(BMU,i) \cdot (x - w_{old})$" + "\n"
        step_description += "Where α = 0.5 and h(BMU,i) = 1 for BMU, 0.5 for neighbors, 0 otherwise.\n\n"
        
        for i in range(num_neurons):
            h = neighborhood_factor(bmu_index, i)
            if h > 0:
                update = learning_rate * h * (x - weights[i])
                step_description += f"Updating Neuron {i+1}: Change = {update}\n"
                weights[i] += update

        # Save a snapshot (store epoch, iteration, copy of weights, and detailed description)
        snapshots.append((epoch, iteration_total, weights.copy(), step_description))
        
        # Show the current state with detailed description
        # plot_som_state(weights, f"Epoch {epoch}, Iteration {iteration_total}", step_description)

print("Training complete. Now showing a summary of all iterations (without detailed descriptions).")

# --- Create a scrollable visualization ---
class ScrollablePlot:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.current_idx = 0
        self.total_snapshots = len(snapshots)
        
        # Create the main figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)  # Make room for buttons
        
        # Create navigation buttons
        self.prev_ax = plt.axes([0.2, 0.05, 0.1, 0.075])
        self.next_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.prev_button = Button(self.prev_ax, 'Previous')
        self.next_button = Button(self.next_ax, 'Next')
        self.prev_button.on_clicked(self.prev_snapshot)
        self.next_button.on_clicked(self.next_snapshot)
        
        # Add page counter text
        self.counter_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.counter_ax.axis('off')
        self.counter_text = self.counter_ax.text(0.5, 0.5, f"{self.current_idx+1}/{self.total_snapshots}", 
                                               ha='center', va='center', fontsize=12)
        
        # Display the first snapshot
        self.update_display()
        
    def update_display(self):
        # Clear the current axes
        self.ax.clear()
        
        # Get current snapshot data
        epoch, it, weights, _ = self.snapshots[self.current_idx]
        
        # Plot the current snapshot
        plot_som_state_summary(self.ax, weights, f"Epoch {epoch}, Iteration {it}")
        
        # Update the counter text
        self.counter_text.set_text(f"{self.current_idx+1}/{self.total_snapshots}")
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
    
    def prev_snapshot(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_snapshot(self, event):
        if self.current_idx < self.total_snapshots - 1:
            self.current_idx += 1
            self.update_display()

# After your training loop completes and snapshots are collected
print("Training complete. Now showing an interactive visualization.")

# Create the scrollable visualization with navigation buttons
# scrollable = ScrollablePlot(snapshots)
# plt.show()

class ScrollableGrid:
    def __init__(self, snapshots, cols=4, rows_per_page=3):
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
        
        self.fig.suptitle("SOM Training Process: Step-by-Step Summary (3D → 2D)", fontsize=16)
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

# Create and display the scrollable grid
# grid = ScrollableGrid(snapshots, cols=3, rows_per_page=3)
# plt.show()

# Add a 3D-2D comparison visualization
def plot_3d_2d_comparison(weights, iteration_info):
    """
    Create a figure with both 3D data visualization and 2D SOM grid
    """
    fig = plt.figure(figsize=(15, 7))
    
    # 3D subplot for the original data space
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot data points in 3D
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', s=100, label='Data Points', marker='o')
    
    # Plot neuron weights in 3D
    ax1.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='red', s=150, marker='s', label='Neurons')
    
    # Add labels for points
    for i, point in enumerate(data):
        ax1.text(point[0], point[1], point[2], f'C{i+1}', size=10, color='blue')
    
    for i, w in enumerate(weights):
        ax1.text(w[0], w[1], w[2], f'N{i+1}', size=10, color='red')
    
    # Draw lines between neighboring neurons in 3D to show topology
    neuron_lines = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for i, j in neuron_lines:
        ax1.plot([weights[i, 0], weights[j, 0]], 
                [weights[i, 1], weights[j, 1]], 
                [weights[i, 2], weights[j, 2]], 'k--', lw=1)
    
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Income (k$)')
    ax1.set_zlabel('Purchase Frequency')
    ax1.set_title('Original 3D Data Space')
    ax1.legend()
    
    # 2D subplot for the SOM grid
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Plot SOM grid
    grid_pos = np.array([
        [0, 0],  # Neuron 1 position
        [1, 0],  # Neuron 2 position
        [0, 1],  # Neuron 3 position
        [1, 1]   # Neuron 4 position
    ])
    
    ax2.scatter(grid_pos[:, 0], grid_pos[:, 1], c='red', s=200, marker='s')
    
    # Label neurons and show their weights
    for i, pos in enumerate(grid_pos):
        ax2.text(pos[0], pos[1], f'N{i+1}', ha='center', va='center', fontsize=12, color='white')
        # Show the 3D weights as a label near each neuron
        weight_text = f"({weights[i, 0]:.1f}, {weights[i, 1]:.1f}, {weights[i, 2]:.1f})"
        ax2.annotate(weight_text, (pos[0], pos[1]), 
                   xytext=(0, 20), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Draw grid connections
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        ax2.plot([grid_pos[i, 0], grid_pos[j, 0]], 
               [grid_pos[i, 1], grid_pos[j, 1]], 'k-', lw=1.5)
    
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_title('2D SOM Grid (Dimensionality Reduction)')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['0', '1'])
    ax2.set_yticklabels(['0', '1'])
    ax2.grid(True)
    
    fig.suptitle(f"3D Data to 2D SOM Mapping - {iteration_info}", fontsize=14)
    plt.tight_layout()
    return fig

class ScrollableComparisonView:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.current_idx = 0
        self.total_snapshots = len(snapshots)
        
        # Create the control figure
        self.control_fig = plt.figure(figsize=(8, 1))
        self.control_fig.suptitle('Navigation Controls', fontsize=10)
        
        # Create navigation buttons
        self.prev_ax = plt.axes([0.2, 0.2, 0.1, 0.6])
        self.next_ax = plt.axes([0.7, 0.2, 0.1, 0.6])
        self.prev_button = Button(self.prev_ax, 'Previous')
        self.next_button = Button(self.next_ax, 'Next')
        self.prev_button.on_clicked(self.prev_snapshot)
        self.next_button.on_clicked(self.next_snapshot)
        
        # Add page counter text
        self.counter_ax = plt.axes([0.4, 0.2, 0.2, 0.6])
        self.counter_ax.axis('off')
        self.counter_text = self.counter_ax.text(0.5, 0.5, f"{self.current_idx+1}/{self.total_snapshots}", 
                                               ha='center', va='center', fontsize=12)
        
        # Figure for the plots
        self.fig = None
        
        # Display the first snapshot
        self.update_display()
        
    def update_display(self):
        # Close previous figure if it exists
        if self.fig is not None:
            plt.close(self.fig)
        
        # Get current snapshot data
        epoch, it, weights, _ = self.snapshots[self.current_idx]
        
        # Create new comparison figure
        self.fig = plot_3d_2d_comparison(weights, f"Epoch {epoch}, Iteration {it}")
        
        # Update the counter text
        self.counter_text.set_text(f"{self.current_idx+1}/{self.total_snapshots}")
        
        # Show the figures
        plt.figure(self.fig.number)
        plt.show(block=False)
        
        # Bring control figure to front
        plt.figure(self.control_fig.number)
        self.control_fig.canvas.draw_idle()
    
    def prev_snapshot(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_snapshot(self, event):
        if self.current_idx < self.total_snapshots - 1:
            self.current_idx += 1
            self.update_display()

# Create the 3D-2D comparison view
# print("Showing 3D data to 2D SOM mapping visualization...")
# comparison_view = ScrollableComparisonView(snapshots)
# plt.show()

# Create a comparison of the last iteration of each epoch in a single 3D plot
def plot_epochs_comparison():
    """
    Create a 3D plot showing the progression of neuron weights across epochs
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points in 3D
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', s=100, label='Data Points', marker='o')
    
    # Add labels for data points
    for i, point in enumerate(data):
        ax.text(point[0], point[1], point[2], f'C{i+1}', size=10, color='blue')
    
    # Create a colormap for the epochs
    cmap = plt.cm.viridis
    epoch_colors = cmap(np.linspace(0, 1, num_epochs))
    
    # Find the last iteration of each epoch
    last_iterations = []
    current_epoch = 1
    
    for idx, (epoch, _, _, _) in enumerate(snapshots):
        if epoch > current_epoch:
            # The previous iteration was the last of the previous epoch
            last_iterations.append(idx - 1)
            current_epoch = epoch
    
    # Add the last iteration of the final epoch
    last_iterations.append(len(snapshots) - 1)
    
    # Plot neurons for each epoch's last iteration with different colors
    for i, idx in enumerate(last_iterations):
        epoch, iteration, weights, _ = snapshots[idx]
        
        # Plot neuron weights with epoch-specific color
        ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], 
                  c=[epoch_colors[i]], s=150, marker='s', 
                  label=f'Epoch {epoch}')
        
        # Draw lines between neighboring neurons to show topology
        neuron_lines = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for j, k in neuron_lines:
            ax.plot([weights[j, 0], weights[k, 0]], 
                   [weights[j, 1], weights[k, 1]], 
                   [weights[j, 2], weights[k, 2]], 
                   c=epoch_colors[i], linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Income (k$)')
    ax.set_zlabel('Purchase Frequency')
    ax.set_title('Neuron Weight Evolution Across Epochs', fontsize=14)
    ax.legend()
    
    # Add explanation text
    plt.figtext(0.02, 0.02, 
               "This visualization shows the progression of SOM neuron weights across epochs.\n" + 
               "Each color represents the final state of the neurons after an epoch.\n" +
               "The neurons adapt to better represent the data distribution while maintaining their 2D grid topology.",
               fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    return fig

# Create a comparison of 2D SOM grids showing weight adaptation across epochs
def plot_2d_som_evolution():
    """
    Create a grid of 2D SOM visualizations showing the evolution across epochs
    """
    num_plots = num_epochs
    fig = plt.figure(figsize=(15, 10))
    
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    grid_pos = np.array([
        [0, 0],  # Neuron 1 position
        [1, 0],  # Neuron 2 position
        [0, 1],  # Neuron 3 position
        [1, 1]   # Neuron 4 position
    ])
    
    # Get the last iteration of each epoch
    last_iterations = []
    current_epoch = 1
    
    for idx, (epoch, _, _, _) in enumerate(snapshots):
        if epoch > current_epoch:
            # The previous iteration was the last of the previous epoch
            last_iterations.append(idx - 1)
            current_epoch = epoch
    
    # Add the last iteration of the final epoch
    last_iterations.append(len(snapshots) - 1)
    
    # Create a subplot for each epoch
    for i, idx in enumerate(last_iterations):
        epoch, iteration, weights, _ = snapshots[idx]
        
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        
        # Plot SOM grid
        ax.scatter(grid_pos[:, 0], grid_pos[:, 1], c='red', s=200, marker='s')
        
        # Label neurons and show their weights
        for j, pos in enumerate(grid_pos):
            ax.text(pos[0], pos[1], f'N{j+1}', ha='center', va='center', fontsize=10, color='white')
            # Show the 3D weights in a compact form
            weight_text = f"({weights[j, 0]:.0f},{weights[j, 1]:.0f},{weights[j, 2]:.1f})"
            ax.annotate(weight_text, (pos[0], pos[1]), 
                      xytext=(0, 20), textcoords='offset points',
                      ha='center', va='bottom', fontsize=8, 
                      bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        # Draw grid connections
        for j, k in [(0, 1), (0, 2), (1, 3), (2, 3)]:
            ax.plot([grid_pos[j, 0], grid_pos[k, 0]], 
                  [grid_pos[j, 1], grid_pos[k, 1]], 'k-', lw=1.5)
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(f'Epoch {epoch}')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(True)
    
    fig.suptitle('SOM Evolution Across Epochs - Weight Adaptation', fontsize=16)
    plt.tight_layout()
    return fig

# Add comparative visualization at the end of your script
print("\nCreating epoch comparison visualizations...")

# Plot the 3D comparison
epochs_comparison_fig = plot_epochs_comparison()
plt.show()

# Plot the 2D SOM evolution
som_evolution_fig = plot_2d_som_evolution()
plt.show()

# Create a final unified visualization showing both 3D data and SOM evolution
def create_final_comparison_visualization():
    """
    Create a comprehensive visualization showing:
    1. Original 3D data space
    2. Initial SOM state
    3. Final SOM state
    4. Dimensionality reduction explanation
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 3D Data Space with final SOM state
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Plot data points
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', s=100, marker='o')
    
    # Get final weights
    _, _, final_weights, _ = snapshots[-1]
    
    # Plot final neuron positions
    ax1.scatter(final_weights[:, 0], final_weights[:, 1], final_weights[:, 2], 
               c='red', s=150, marker='s')
    
    # Connect neurons to show grid topology
    neuron_lines = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for i, j in neuron_lines:
        ax1.plot([final_weights[i, 0], final_weights[j, 0]], 
                [final_weights[i, 1], final_weights[j, 1]], 
                [final_weights[i, 2], final_weights[j, 2]], 
                'r--', lw=1.5)
    
    ax1.set_title('Final SOM State in 3D Data Space', fontsize=12)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Income (k$)')
    ax1.set_zlabel('Purchase Frequency')
    
    # 2D SOM Grid - Initial State
    ax2 = fig.add_subplot(2, 2, 2)
    grid_pos = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    
    # Get initial weights
    _, _, initial_weights, _ = snapshots[0]
    
    ax2.scatter(grid_pos[:, 0], grid_pos[:, 1], c='blue', s=200, marker='s')
    
    # Label neurons and show their initial weights
    for i, pos in enumerate(grid_pos):
        ax2.text(pos[0], pos[1], f'N{i+1}', ha='center', va='center', fontsize=10, color='white')
        # Show the 3D weights
        weight_text = f"({initial_weights[i, 0]:.1f},{initial_weights[i, 1]:.1f},{initial_weights[i, 2]:.1f})"
        ax2.annotate(weight_text, (pos[0], pos[1]), 
                   xytext=(0, 20), textcoords='offset points',
                   ha='center', va='bottom', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Draw grid connections
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        ax2.plot([grid_pos[i, 0], grid_pos[j, 0]], 
               [grid_pos[i, 1], grid_pos[j, 1]], 'b-', lw=1.5)
    
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_title('Initial 2D SOM Grid', fontsize=12)
    ax2.grid(True)
    ax2.set_xticklabels([])  # Remove x-axis tick labels
    ax2.set_yticklabels([])  # Remove y-axis tick labels
    
    # 2D SOM Grid - Final State
    ax3 = fig.add_subplot(2, 2, 3)
    
    ax3.scatter(grid_pos[:, 0], grid_pos[:, 1], c='red', s=200, marker='s')
    
    # Label neurons and show their final weights
    for i, pos in enumerate(grid_pos):
        ax3.text(pos[0], pos[1], f'N{i+1}', ha='center', va='center', fontsize=10, color='white')
        # Show the 3D weights
        weight_text = f"({final_weights[i, 0]:.1f},{final_weights[i, 1]:.1f},{final_weights[i, 2]:.1f})"
        ax3.annotate(weight_text, (pos[0], pos[1]), 
                   xytext=(0, 20), textcoords='offset points',
                   ha='center', va='bottom', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Draw grid connections
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        ax3.plot([grid_pos[i, 0], grid_pos[j, 0]], 
               [grid_pos[i, 1], grid_pos[j, 1]], 'r-', lw=1.5)
    
    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_title('Final 2D SOM Grid', fontsize=12)
    ax3.grid(True)
    ax3.set_xticklabels([])  # Remove x-axis tick labels
    ax3.set_yticklabels([])  # Remove y-axis tick labels
    
    # Text explanation
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    explanation = (
        "Dimensionality Reduction in Self-Organizing Maps\n\n"
        "This visualization demonstrates how SOMs achieve dimensionality reduction:\n\n"
        "1. Original Data: 3D space (Age, Income, Purchase Frequency)\n"
        "2. SOM Representation: 2D grid of neurons\n\n"
        "The SOM neurons adapt their 3D weight vectors to represent\n"
        "the data distribution while maintaining a 2D grid topology.\n\n"
        "Key aspects of SOM dimensionality reduction:\n"
        "• Similar high-dimensional data points map to nearby neurons\n"
        "• Topological relationships from original space are preserved\n"
        "• The 2D grid provides an interpretable representation of 3D data\n\n"
        "This makes SOMs effective for visualizing high-dimensional data\n"
        "in a way that preserves meaningful relationships."
    )
    ax4.text(0.02, 0.98, explanation, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    fig.suptitle('SOM Dimensionality Reduction: 3D → 2D', fontsize=16)
    plt.tight_layout()
    return fig

# Create and show the final visualization
final_visualization = create_final_comparison_visualization()
plt.show()