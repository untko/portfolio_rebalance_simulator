# Portfolio Rebalancing Simulator

This application simulates and visualizes the performance of different portfolio rebalancing strategies based on historical market data.

## Features

- **Define Your Portfolio:**
  - Set the initial investment amount.
  - Add multiple assets to your portfolio (e.g., BTC, Gold).
  - Specify the desired asset allocation ratio using a slider or manual input.

- **Select Simulation Range:**
  - Use predefined time ranges (e.g., 5 years, 1 year, 1 month).
  - Define a custom date range for the simulation.

- **Choose Rebalancing Strategies:**
  - **Periodic Rebalancing:** Set a fixed time interval for rebalancing (daily, weekly, monthly, quarterly, yearly, or custom days).
  - **Threshold Rebalancing:** Rebalance the portfolio whenever an asset's weight deviates by a certain percentage (e.g., 0.1%, 1%, 5%, 10%, 50%, or custom %).

- **Visualize Results:**
  - Run the simulation to see how your portfolio would have performed.
  - Compare the performance of different rebalancing strategies on an interactive chart.
  - An "automatic" option can select and display the best-performing strategies.

## Tech Stack

- **Backend:** Python
  - **Web Framework:** Streamlit - for creating the interactive web interface.
  - **Data Handling:** Pandas - for data manipulation and analysis.
  - **Numerical Operations:** NumPy - for calculations.
  - **Financial Data:** yfinance - to fetch historical market data for assets.
- **Charting:** Plotly - for creating interactive visualizations of the simulation results.

## Getting Started

### Prerequisites

- Python 3.8+
- Pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rebalance_simulator_v2
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
