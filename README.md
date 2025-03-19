# Mall Customer Segmentation using Self-Organizing Maps

## Overview

This project implements a Self-Organizing Map (SOM) for customer segmentation using mall customer data. The implementation analyzes customer age and income profiles to identify natural segments, which can be used for targeted marketing strategies.

## Set Up a Virtual Environment (recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## Install Required Packages

```bash
# Install all dependencies
pip install numpy matplotlib pandas kagglehub scipy

# If you need exact versions
pip install numpy==1.24.3 matplotlib==3.7.2 pandas==2.0.3 kagglehub==0.2.5 scipy==1.11.3
```

## Usage

Run the script to perform the customer segmentation:

```bash
# Basic usage
python som_mall_customers.py

# If your Python command is python3
python3 som_mall_customers.py
```

## Customizing Parameters

You can modify the following parameters in the script:

```python
# Adjust the grid size (rows, columns)
grid_shape = (3, 3)  # Change to (4, 4) for more segments, (2, 2) for fewer

# Adjust the training parameters
num_epochs = 8  # Increase for more thorough training
initial_learning_rate = 0.5  # Higher for more aggressive updates
learning_rate_decay = 0.9  # Lower for faster decay
```

## Dataset

![Dataset](/images/dataset.png)

The project uses the "Mall Customer Segmentation Data" which contains the following features:

- Customer ID
- Age
- Annual Income (k$)
- Spending Score (1-100)
- Gender

For the SOM analysis, we focus primarily on Age and Annual Income.

## Troubleshooting

### Common Issues

#### Interactive Visualizations Not Working
- Ensure you're running the script in an environment that supports interactive plotting.
- Try using `%matplotlib notebook` if running in Jupyter Notebook.
- For remote servers, consider using `matplotlib.use('Agg')` and saving plots as files.

#### Memory Issues
- Reduce `num_epochs` or dataset size if you encounter memory problems.
- Decrease the frequency of snapshots with `if i % max(1, len(data) // 5) == 0`.

## How It Works

### Self-Organizing Maps

SOMs are a type of artificial neural network that use unsupervised learning to produce a low-dimensional representation of high-dimensional data. The process works as follows:

1. **Initialization:** Random weights are assigned to each neuron in the grid.
2. **Competition:** For each input, the neuron with the closest weights (Best Matching Unit) is found.
3. **Cooperation:** The BMU and its neighbors' weights are updated to better match the input.
4. **Adaptation:** This process repeats, with decreasing learning rate and neighborhood radius.

### Visualizations

- **U-Matrix:** Shows distances between adjacent neurons to identify cluster boundaries.
- **Weight Trajectories:** Tracks how neuron weights change during training.
- **Segment Explorer:** Interactive tool for exploring each customer segment in detail.
- **Customer Distribution:** Shows how customers are distributed across segments.

## Results

After running the script, you'll get:

- An interactive visualization showing the training process.
- A U-Matrix showing the cluster boundaries.
- Customer segment analysis with demographic profiles.
- An interactive segment explorer for detailed inspection.
- A summary report (PDF/PNG) with key findings.

![initial_data](/results/initial_data.png)
![weight_trajectories](/results/neuron_trajectories.png)
![final](/results/final.png)
![analysis](/results/analysis.png)
![seg1](/results/seg1.png)
![seg2](/results/seg2.png)
![seg3](/results/seg3.png)
![seg4](/results/seg4.png)
![seg5](/results/seg5.png)
![seg6](/results/seg6.png)
![seg7](/results/seg7.png)
![seg8](/results/seg8.png)
![seg9](/results/seg9.png)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mall Customer dataset from Kaggle.
- Inspired by the concept of Self-Organizing Maps by Teuvo Kohonen.

## Created

2023-03-19 13:21:21 by Rishikesh0523