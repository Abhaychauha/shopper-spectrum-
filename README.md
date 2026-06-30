# 🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce

Shopper Spectrum analyzes e-commerce transaction data to uncover customer purchasing patterns. It segments customers using **RFM (Recency, Frequency, Monetary) analysis** with **K-Means clustering**, and recommends products using **item-based collaborative filtering** with **cosine similarity**. Both capabilities are exposed through an interactive **Streamlit** web app.

## Features

### 🎯 Product Recommendation
Enter a product name and get the **top 5 similar products**, based on cosine similarity computed over the customer–product purchase matrix.

### 🧩 Customer Segmentation
Enter a customer's Recency, Frequency, and Monetary values and instantly get their predicted segment:
- **High-Value** — recent, frequent, big spenders
- **Regular** — steady purchasers, moderate spend
- **Occasional** — infrequent, low-spend buyers
- **At-Risk** — haven't purchased in a long time

## Project Structure

```
.
├── app.py                      # Streamlit web application
├── train_model.py              # Data cleaning, RFM, clustering & similarity pipeline
├── requirements.txt            # Python dependencies
├── rfm_kmeans_model.pkl        # Trained KMeans model
├── rfm_scaler.pkl              # Fitted StandardScaler
├── cluster_label_map.pkl       # Cluster ID -> segment label mapping
└── product_similarity.pkl      # Product-product cosine similarity matrix
```

## Dataset

The model is trained on an online retail transaction dataset (`online_retail.csv`) with the following columns:

| Column | Description |
|---|---|
| InvoiceNo | Transaction number |
| StockCode | Unique product/item code |
| Description | Name of the product |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per product |
| CustomerID | Unique identifier for each customer |
| Country | Country where the customer is based |

> The raw dataset is not included in this repo due to size. Place `online_retail.csv` in the project root before running the training script.

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/shopper-spectrum.git
cd shopper-spectrum
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the models
Pretrained `.pkl` artifacts are included, so this step is optional unless you want to retrain on new data.
```bash
python train_model.py --input online_retail.csv --outdir .
```

### 4. Run the app
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

## How It Works

**Preprocessing:** rows with missing CustomerID, cancelled invoices, and non-positive quantity/price are removed.

**Customer Segmentation:** Recency, Frequency, and Monetary values are computed per customer, standardized with `StandardScaler`, and clustered with `KMeans (k=4)`. Each cluster is automatically labeled by comparing its average RFM profile against the others.

**Product Recommendation:** a customer × product purchase matrix is built, and cosine similarity is computed between products to find the closest matches to any given item.

## Tech Stack

Python, Pandas, NumPy, scikit-learn, Streamlit

## License

This project is open source and available under the [MIT License](LICENSE).
