# timeseries_w

Timeseries Analysis

## Overview

This repository contains code and resources for performing time series analysis, focusing on energy consumption data. It includes exploratory data analysis (EDA), modeling, and testing components.

## Repository Structure

- `EDA+model.ipynb`: A Jupyter Notebook that combines exploratory data analysis and modeling of the time series data.
- `energy_model.py`: A Python script defining functions and classes for modeling energy consumption time series data.
- `test_energy_model.py`: A Python script containing unit tests for the functions and classes defined in `energy_model.py`.
- `test_data/`: A directory containing sample data used for testing and analysis.
- `requirements.txt`: A file listing the Python dependencies required to run the project.

## Getting Started

To get started with this project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Kanaliha/timeseries_w.git

2. **Navigate to the project directory**:

    ```bash
    cd timeseries_w

3. **Install the required dependencies**:
   
    ```bash
    pip install -r requirements.txt

4. **Run the tests**:

    ```bash
    python -m unittest discover

5. **Use energy model**:

    ```bash
    # usage example
    python energy_model.py --input .\test_data\SG.csv --quantity Consumption

**Model output**
The model opens a plot displaying the data for the chosen quantity together with a fitted model. It also returns information about the accuracy of the created model according to the Coefficient of Determination (R²) metric.