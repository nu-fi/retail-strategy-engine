[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-strategy-engine.streamlit.app/)
# 🛒 Retail Strategy Engine (Streamlit App)

An interactive analytics tool that combines **Customer Segmentation (RFM)** and **Market Basket Analysis (Apriori)** to generate data-driven cross-selling strategies.

## 🎯 Business Problem
Retailers often struggle to identify which products to bundle together. Standard analysis treats all customers the same, leading to generic recommendations. This tool solves that by segmenting customers first.

## 🚀 Key Features
* **Automated Data Cleaning:** Handles missing values and cancelled transactions.
* **RFM Segmentation:** Classifies users into 'Champions', 'Loyal', 'At Risk', etc.
* **Dynamic MBA:** Runs Apriori algorithm on specific segments (e.g., "What do Champions buy together?").
* **Memory Optimization:** Handles large datasets by limiting analysis to Top-N popular items.
* **Visualizations:** Interactive Network Graphs for product associations.

## 🛠 Tech Stack
* **Core:** Python 3.9+
* **UI:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** MLxtend (Apriori, Association Rules)
* **Visualization:** NetworkX, Matplotlib

## 💻 How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`