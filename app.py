import streamlit as st
import pandas as pd
import datetime as dt
import networkx as nx
import matplotlib.pyplot as plt
import os
import multiprocessing # Library untuk menjalankan proses terpisah
import queue # Untuk mengambil hasil dari proses
from mba_utils import calculate_mba_process # Import fungsi dari file sebelah

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Retail Strategy Tool", layout="wide")

st.title("💎 Customer Segmentation & Market Basket Analysis")
st.markdown("""
**Workflow Linear:** Pilih Data -> Cleaning -> Segmentasi (RFM) -> Analisis Pola Belanja (MBA).
""")

# --- 2. PENGATURAN PARAMETER ---
with st.expander("⚙️ Konfigurasi Parameter Analisis (Klik untuk Buka)", expanded=False):
    st.info("Atur parameter ini untuk mengontrol sensitivitas analisis.")
    p1, p2, p3 = st.columns(3)
    
    with p1:
        st.markdown("**1. Optimasi Memori**")
        max_products = st.slider("Batasi Produk (Top N)", 100, 2000, 500, 50)
        # Timeout slider
        timeout_limit = st.number_input("Batas Waktu (Detik)", value=30, min_value=5, max_value=300)
        algo_choice = st.selectbox("Algoritma", ["fpgrowth (Cepat)", "apriori (Klasik)"])
    
    with p2:
        st.markdown("**2. Filter Populer**")
        min_support = st.slider("Min Support", 0.001, 0.2, 0.02, 0.001)
    
    with p3:
        st.markdown("**3. Filter Asosiasi**")
        min_metric = st.selectbox("Metric", ["lift", "confidence"])
        min_threshold = st.slider("Threshold", 0.1, 10.0, 1.0, 0.1)

# --- 3. FUNGSI UTAMA ---
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='cp1252')
    return data

def plot_network_graph(rules, n_rules=20):
    top_rules = rules.sort_values("lift", ascending=False).head(n_rules)
    G = nx.DiGraph()
    for i, row in top_rules.iterrows():
        ant = str(row['antecedents'])[:20] 
        con = str(row['consequents'])[:20]
        G.add_edge(ant, con, weight=row['lift'], confidence=row['confidence'])

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, k=2, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='#FF6B6B', alpha=0.9, ax=ax)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    width = [w * 0.5 for w in weights] if weights else 1.0
    nx.draw_networkx_edges(G, pos, width=width, edge_color='#4ECDC4', arrowstyle='->', arrowsize=15, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', font_weight='bold', ax=ax)
    edge_labels = {(u, v): f"Lift: {d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    ax.axis('off')
    return fig

# Wrapper agar multiprocessing bisa kirim hasil lewat Queue
def process_wrapper(q, basket, min_sup, metric, thresh, algo):
    res, msg = calculate_mba_process(basket, min_sup, metric, thresh, algo)
    q.put((res, msg))

# --- 4. PEMILIHAN DATA ---
st.divider()
st.header("📂 1. Pilih Sumber Data")

data_source = st.radio("Sumber Data:", ["Upload File CSV Sendiri", "Pakai Dataset Default (Online Retail II)"])
df = None
default_cols = {} 

if data_source == "Upload File CSV Sendiri":
    uploaded_file = st.file_uploader("Upload File Transaksi (Format CSV)", type=['csv'])
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.success("✅ File Terupload dan Terload!")
            with st.expander("Lihat Sampel Data"):
                st.dataframe(df.head())
        except Exception as e: st.error(f"Gagal: {e}")
else: 
    LOCAL_PATH = "dataset/online_retail_II.csv"
    if os.path.exists(LOCAL_PATH):
        try:
            df = load_data(LOCAL_PATH)
            default_cols = {'tx': 'Invoice', 'item': 'Description', 'cust': 'Customer ID', 'qty': 'Quantity', 'price': 'Price', 'date': 'InvoiceDate'}
            st.success(f"✅ Data Lokal Terload: {LOCAL_PATH}")
            with st.expander("Lihat Sampel Data"):
                st.dataframe(df.head())
        except Exception as e: st.error(f"Gagal: {e}")
    else: st.warning(f"File tidak ada di: {LOCAL_PATH}")

# --- 5. LOGIKA APLIKASI ---
if df is not None:
    st.divider()
    st.header("🧹 2. Data Preparation")
    
    def get_index(columns, search_name):
        try: return list(columns).index(search_name)
        except: return 0

    c1, c2, c3 = st.columns(3)
    with c1:
        tx_col = st.selectbox("Kolom Invoice", df.columns, index=get_index(df.columns, default_cols.get('tx', '')))
    with c2:
        item_col = st.selectbox("Kolom Produk", df.columns, index=get_index(df.columns, default_cols.get('item', '')))
    with c3:
        cust_col = st.selectbox("Kolom Customer ID", df.columns, index=get_index(df.columns, default_cols.get('cust', '')))
        
    with st.expander("Opsi Cleaning"):
        rm_cancel = st.checkbox("Hapus Cancel ('C')", value=True)
        rm_null = st.checkbox("Hapus Null Customer", value=True)

    if st.button("Proses Data ⬇️"):
        df_clean = df.copy()
        if rm_null:
            df_clean = df_clean.dropna(subset=[cust_col])
            df_clean[cust_col] = df_clean[cust_col].astype(str).str.split('.').str[0]
        if rm_cancel:
            df_clean[tx_col] = df_clean[tx_col].astype(str)
            df_clean = df_clean[~df_clean[tx_col].str.contains("C", na=False)]
        
        # Fallback Cols
        if not default_cols:
             qty_col = next((c for c in df.columns if 'quantity' in c.lower()), df.columns[3])
             price_col = next((c for c in df.columns if 'price' in c.lower()), df.columns[5])
             date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[4])
        else:
             qty_col = default_cols.get('qty'); price_col = default_cols.get('price'); date_col = default_cols.get('date')

        try:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col])
            df_clean[qty_col] = pd.to_numeric(df_clean[qty_col])
            df_clean[price_col] = pd.to_numeric(df_clean[price_col])
            df_clean['TotalAmount'] = df_clean[qty_col] * df_clean[price_col]
            df_clean = df_clean[(df_clean[qty_col] > 0) & (df_clean[price_col] > 0)]
            
            st.session_state['df_clean'] = df_clean
            st.session_state['cols'] = {'tx': tx_col, 'item': item_col, 'cust': cust_col, 'date': date_col}
            st.success(f"Siap! {df_clean.shape[0]} Baris.")
        except Exception as e: st.error(f"Error Konversi: {e}")

    # --- 6. RFM ---
    if 'df_clean' in st.session_state:
        st.divider()
        st.header("👥 3. Segmentasi (RFM)")
        df_proc = st.session_state['df_clean']; cols = st.session_state['cols']
        
        if st.button("Jalankan RFM ⬇️"):
            with st.spinner("Menghitung RFM..."):
                analysis_date = df_proc[cols['date']].max() + dt.timedelta(days=1)
                rfm = df_proc.groupby(cols['cust']).agg({
                    cols['date']: lambda x: (analysis_date - x.max()).days,
                    cols['tx']: 'nunique', 'TotalAmount': 'sum'
                })
                rfm.rename(columns={cols['date']: 'Recency', cols['tx']: 'Frequency', 'TotalAmount': 'Monetary'}, inplace=True)
                
                rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
                rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
                
                def segment(r):
                    if r['R_Score'] >= 4 and r['F_Score'] >= 4: return 'Champions'
                    elif r['R_Score'] >= 2 and r['F_Score'] >= 3: return 'Loyal'
                    elif r['R_Score'] <= 2 and r['F_Score'] >= 4: return 'At Risk'
                    elif r['R_Score'] >= 3 and r['F_Score'] <= 2: return 'New Customers'
                    elif r['R_Score'] <= 2 and r['F_Score'] <= 2: return 'Lost'
                    else: return 'Potential'
                rfm['Segment'] = rfm.apply(segment, axis=1)
                st.session_state['rfm_result'] = rfm
        
        if 'rfm_result' in st.session_state:
            rfm = st.session_state['rfm_result']
            st.bar_chart(rfm['Segment'].value_counts())

            # --- 7. MBA (MULTIPROCESSING) ---
            st.divider()
            st.header("🛒 4. Analisis MBA (Safe Mode)")
            f1, f2 = st.columns(2)
            country_col = next((c for c in df_proc.columns if 'country' in c.lower()), None)
            if country_col:
                with f1: sel_country = st.selectbox("Negara", sorted(df_proc[country_col].astype(str).unique()))
            with f2: sel_seg = st.selectbox("Segmen", rfm['Segment'].unique())
            
            if st.button("Mulai Analisis (Dengan Timeout) 🚀"):
                target_cust = rfm[rfm['Segment'] == sel_seg].index
                df_target = df_proc[(df_proc[cols['cust']].isin(target_cust)) & (df_proc[country_col] == sel_country)]
                
                if df_target.empty:
                    st.warning("Data Kosong.")
                else:
                    st.write(f"Menganalisis **{df_target.shape[0]}** baris...")
                    
                    # 1. Optimasi Basket
                    top = df_target[cols['item']].value_counts().head(max_products).index
                    df_target = df_target[df_target[cols['item']].isin(top)]
                    basket = pd.crosstab(df_target[cols['tx']], df_target[cols['item']]).astype('bool')

                    # 2. RUN PROCESS (KILLABLE)
                    result_queue = multiprocessing.Queue()
                    algo_code = 'fpgrowth' if 'fpgrowth' in algo_choice else 'apriori'
                    
                    p = multiprocessing.Process(
                        target=process_wrapper, 
                        args=(result_queue, basket, min_support, min_metric, min_threshold, algo_code)
                    )
                    
                    p.start()
                    
                    with st.spinner(f"Sedang menghitung... (Max {timeout_limit} detik)"):
                        p.join(timeout=timeout_limit)
                    
                    if p.is_alive():
                        p.terminate() # KILL PROCESS SEKARANG!
                        p.join()
                        st.error(f"⛔ **TIMEOUT!** Proses dimatikan paksa karena > {timeout_limit} detik.")
                        st.info("Tips: Kurangi 'Batasi Produk' atau naikkan 'Min Support'.")
                    else:
                        if not result_queue.empty():
                            rules_res, msg = result_queue.get()
                            if rules_res is None:
                                st.warning(f"Gagal: {msg}")
                            else:
                                st.success(f"✅ Selesai! {len(rules_res)} Rules.")
                                # Fix PyArrow
                                rules_res['antecedents'] = rules_res['antecedents'].apply(lambda x: ', '.join(list(x)))
                                rules_res['consequents'] = rules_res['consequents'].apply(lambda x: ', '.join(list(x)))
                                
                                t1, t2 = st.tabs(["Tabel", "Grafik"])
                                with t1: st.dataframe(rules_res.sort_values('lift', ascending=False).head(10))
                                with t2: st.pyplot(plot_network_graph(rules_res))
                        else:
                            st.error("Proses mati tanpa hasil (Out of Memory?).")