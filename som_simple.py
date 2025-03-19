import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
import kagglehub  # Ensure you have this module installed

# --- Download the latest version of the customer dataset from Haggle (Kaggle) ---
path = kagglehub.dataset_download("shrutimechlearn/customer-data")
# --- Load the dataset using pandas ---
# Assuming the downloaded file is named 'Mall_Customers.csv' inside the given path.
csv_file = os.path.join(path, "Mall_Customers.csv")
customer_df = pd.read_csv(csv_file)
# Convert the DataFrame to a NumPy array for the SOM (selecting 'Age' and 'Annual_Income_(k$)' columns)
data = customer_df[['Age', 'Annual_Income_(k$)']].values

# --- Define the SOM grid parameters (change grid_shape to adjust the grid size) ---
grid_shape = (2, 2)  # For example, (rows, cols) = (2,2) or (3,3) or (4,4), etc.
num_neurons = grid_shape[0] * grid_shape[1]

# Dynamically initialize weights within the data range (for clarity, random initialization)
min_age, max_age = data[:, 0].min(), data[:, 0].max()
min_income, max_income = data[:, 1].min(), data[:, 1].max()
weights = np.random.uniform(low=[min_age, min_income], high=[max_age, max_income], 
                            size=(num_neurons, 2))

# Dynamically create neuron positions in the grid
positions = np.array([[i, j] for i in range(grid_shape[0]) for j in range(grid_shape[1])])

def generate_neuron_lines(weights):
    """
    Generate list of connections (as index pairs) between neurons based on the grid structure.
    Connect neurons horizontally and vertically.
    """
    neuron_lines = []
    rows, cols = grid_shape
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            # Horizontal connection to right neighbor
            if j < cols - 1:
                neuron_lines.append((index, index + 1))
            # Vertical connection to neighbor below
            if i < rows - 1:
                neuron_lines.append((index, index + cols))
    return neuron_lines

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
    # Plot data points
    plt.scatter(data[:, 0], data[:, 1], c='blue', s=100, label='Data Points', marker='o')
    # Plot each neuron (weight vector)
    for i, w in enumerate(weights):
        plt.scatter(w[0], w[1], c='red', s=200, marker='s')
        plt.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='darkred')
    # Draw grid connections for the dynamic grid structure
    neuron_lines = generate_neuron_lines(weights)
    for i, j in neuron_lines:
        plt.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)
    plt.xlabel('Age')
    plt.ylabel('Annual Income (k$)')
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
    neuron_lines = generate_neuron_lines(weights)
    for i, j in neuron_lines:
        ax.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_title(f"Iteration: {iteration_info}", fontsize=14)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(10, 60)
    ax.set_ylim(20, 100)

# --- Initial Descriptive Plot ---
initial_description = (
    "Initial Data Visualization\n\n"
    "Dataset: Customer data points with features Age and Annual Income (k$).\n"
    "Blue circles represent the data points.\n"
    "Red squares represent the SOM neurons with initial weights.\n\n"
    "Process Overview:\n"
    "1. For each input, compute Euclidean distances to each neuron:\n"
    "   d_i = ||x - w_i|| = sqrt((x_age - w_age,i)^2 + (x_income - w_income,i)^2)\n"
    "2. Identify the BMU (neuron with minimum distance).\n"
    "3. Update weights:\n"
    "   w(new) = w(old) + α * h(BMU,i) * (x - w(old))\n"
    "   (where h(BMU,i)=1 for BMU, 0.5 for neighbors, 0 otherwise)\n\n"
    "Press Enter to start the training process."
)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c='blue', s=100, label='Data Points', marker='o')
for i, w in enumerate(weights):
    plt.scatter(w[0], w[1], c='red', s=200, marker='s')
    plt.text(w[0] + 0.5, w[1] + 0.5, f'N{i+1}', fontsize=12, color='darkred')
for i, j in generate_neuron_lines(weights):
    plt.plot([weights[i, 0], weights[j, 0]], [weights[i, 1], weights[j, 1]], 'k--', lw=1)
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title("Initial Data and SOM Setup", fontsize=14)
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
        step_description += "Step 1: Compute Euclidean distances:\n"
        step_description += r"$d_i = ||x - w_i|| = \sqrt{(x_{age} - w_{age,i})^2 + (x_{income} - w_{income,i})^2}$" + "\n"
        
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
        
        # Optionally, show the current state with detailed description:
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
        epoch, it, weights_snapshot, _ = self.snapshots[self.current_idx]
        
        # Plot the current snapshot
        plot_som_state_summary(self.ax, weights_snapshot, f"Epoch {epoch}, Iteration {it}")
        
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
scrollable = ScrollablePlot(snapshots)
plt.show()

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
        
        self.fig.suptitle("SOM Training Process: Step-by-Step Summary", fontsize=16)
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
# grid = ScrollableGrid(snapshots, cols=5, rows_per_page=5)
# plt.show()
