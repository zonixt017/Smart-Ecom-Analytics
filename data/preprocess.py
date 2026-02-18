"""
Preprocess all 3 real Kaggle datasets into clean unified CSVs
that the Streamlit app and notebooks will use.

Run once from project root:
    python data/preprocess.py
"""
import pandas as pd
import numpy as np
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLIST   = os.path.join(ROOT, 'data', 'olist')
CHURN_F = os.path.join(ROOT, 'data', 'churn', 'data_ecommerce_customer_churn.csv')
FLIP_F  = os.path.join(ROOT, 'data', 'flipkart', 'flipkart_product.csv')
OUT     = os.path.join(ROOT, 'data')

# ─────────────────────────────────────────────────────────────────────────────
# 1. OLIST  →  orders_clean.csv
# ─────────────────────────────────────────────────────────────────────────────
print("Processing Olist...")

orders   = pd.read_csv(os.path.join(OLIST, 'olist_orders_dataset.csv'),
                       parse_dates=['order_purchase_timestamp',
                                    'order_delivered_customer_date',
                                    'order_estimated_delivery_date'])
items    = pd.read_csv(os.path.join(OLIST, 'olist_order_items_dataset.csv'))
payments = pd.read_csv(os.path.join(OLIST, 'olist_order_payments_dataset.csv'))
customers= pd.read_csv(os.path.join(OLIST, 'olist_customers_dataset.csv'))
products = pd.read_csv(os.path.join(OLIST, 'olist_products_dataset.csv'))
trans    = pd.read_csv(os.path.join(OLIST, 'product_category_name_translation.csv'))
reviews  = pd.read_csv(os.path.join(OLIST, 'olist_order_reviews_dataset.csv'))

# Translate product categories
products = products.merge(trans, on='product_category_name', how='left')
products['category'] = products['product_category_name_english'].fillna(
    products['product_category_name'].fillna('unknown'))

# Payment totals per order
pay_total = payments.groupby('order_id')['payment_value'].sum().reset_index()
pay_total.columns = ['order_id', 'total_amount']

# Items per order
items_agg = items.groupby('order_id').agg(
    quantity=('order_item_id', 'count'),
    product_id=('product_id', 'first')
).reset_index()

# Review score per order
rev_score = reviews.groupby('order_id')['review_score'].mean().reset_index()

# Merge everything
df = (orders
      .merge(customers[['customer_id','customer_unique_id','customer_city','customer_state']],
             on='customer_id', how='left')
      .merge(pay_total, on='order_id', how='left')
      .merge(items_agg, on='order_id', how='left')
      .merge(products[['product_id','category']], on='product_id', how='left')
      .merge(rev_score, on='order_id', how='left'))

# Keep only delivered orders with payment
df = df[df['order_status'] == 'delivered'].copy()
df = df.dropna(subset=['total_amount', 'order_purchase_timestamp'])

df['order_date']   = df['order_purchase_timestamp'].dt.date
df['order_year']   = df['order_purchase_timestamp'].dt.year
df['order_month']  = df['order_purchase_timestamp'].dt.month
df['category']     = df['category'].fillna('unknown').str.replace('_', ' ').str.title()
df['quantity']     = df['quantity'].fillna(1).astype(int)
df['review_score'] = df['review_score'].fillna(3.0).round(1)

# Rename for consistency with app
df = df.rename(columns={
    'customer_unique_id': 'customer_id',
    'order_purchase_timestamp': 'transaction_date',
    'customer_city': 'city',
    'customer_state': 'state',
})

cols = ['order_id','customer_id','transaction_date','order_date',
        'order_year','order_month','total_amount','quantity',
        'product_id','category','city','state','review_score']
df = df[cols].reset_index(drop=True)

out_path = os.path.join(OUT, 'orders_clean.csv')
df.to_csv(out_path, index=False)
print(f"  Saved orders_clean.csv  — {len(df):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CHURN  →  churn_clean.csv
# ─────────────────────────────────────────────────────────────────────────────
print("Processing Churn...")

churn = pd.read_csv(CHURN_F)
churn.columns = [c.strip() for c in churn.columns]

# Drop rows where Churn is missing
churn = churn.dropna(subset=['Churn']).copy()

# Fill numeric NAs with median
num_cols = churn.select_dtypes(include='number').columns.tolist()
for c in num_cols:
    churn[c] = churn[c].fillna(churn[c].median())

# Fill categorical NAs with mode
cat_cols = churn.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    churn[c] = churn[c].fillna(churn[c].mode()[0])

churn['Churn'] = churn['Churn'].astype(int)

out_path = os.path.join(OUT, 'churn_clean.csv')
churn.to_csv(out_path, index=False)
print(f"  Saved churn_clean.csv   — {len(churn):,} rows, cols: {list(churn.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FLIPKART  →  flipkart_clean.csv
# ─────────────────────────────────────────────────────────────────────────────
print("Processing Flipkart...")

flip = pd.read_csv(FLIP_F, encoding='latin-1')
flip.columns = [c.strip() for c in flip.columns]

# Clean price: remove currency symbols, commas
def clean_price(val):
    if pd.isna(val):
        return np.nan
    s = re.sub(r'[^\d.]', '', str(val).replace(',', ''))
    try:
        return float(s)
    except:
        return np.nan

flip['price_clean'] = flip['Price'].apply(clean_price)

# Clean rating
flip['rating'] = pd.to_numeric(flip['Rate'], errors='coerce')
flip = flip.dropna(subset=['rating'])
flip['rating'] = flip['rating'].clip(1, 5)

# Clean product name — remove garbled chars
flip['product_name'] = (flip['ProductName']
                        .str.encode('ascii', errors='ignore')
                        .str.decode('ascii')
                        .str.strip()
                        .str[:80])

# Extract category from product name (first word group before first digit/special)
def extract_category(name):
    if pd.isna(name) or name.strip() == '':
        return 'Other'
    parts = name.split()
    # Take first 1-2 meaningful words
    words = [w for w in parts[:3] if w.isalpha() and len(w) > 2]
    return ' '.join(words[:2]) if words else 'Other'

flip['category'] = flip['product_name'].apply(extract_category)

# Review text
flip['review_text'] = flip['Review'].fillna('').astype(str).str[:200]
flip['summary']     = flip['Summary'].fillna('').astype(str).str[:100]

# Keep useful columns
flip_out = flip[['product_name','category','price_clean','rating','review_text','summary']].copy()
flip_out = flip_out.rename(columns={'price_clean': 'price'})
flip_out = flip_out.dropna(subset=['product_name'])
flip_out = flip_out[flip_out['product_name'].str.len() > 3]
flip_out = flip_out.reset_index(drop=True)

out_path = os.path.join(OUT, 'flipkart_clean.csv')
flip_out.to_csv(out_path, index=False)
print(f"  Saved flipkart_clean.csv — {len(flip_out):,} rows")

print("\nAll preprocessing complete!")
print(f"Output files in: {OUT}")
