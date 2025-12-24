import streamlit as st
import pandas as pd
import datetime as dt
import networkx as nx
import matplotlib.pyplot as plt
import os
import multiprocessing 
import queue 
from mba_utils import calculate_mba_process 

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Retail Strategy Tool", layout="wide")

st.title("Customer Segmentation & Market Basket Analysis")
st.markdown("""
**Workflow Linear:** Choose Data -> Cleaning -> Segmentation (RFM) -> Market Basket Analysis (MBA).
""")

# --- 2. PARAMETER SETTING ---
with st.expander("⚙️ Parameter configuration for MBA analysis.", expanded=False):
    st.info("Adjust these parameters to control the analysis sensitivity.")
    p1, p2, p3 = st.columns(3)
    
    with p1:
        st.markdown("**1. Memory Optimization**")
        max_products = st.slider("Limit Products (Top N)", 100, 2000, 500, 50)
        # Timeout slider
        timeout_limit = st.number_input("Timeout (Seconds)", value=30, min_value=5, max_value=300)
        algo_choice = st.selectbox("Algorithm", ["fpgrowth (Fast)", "apriori (Classic)"])
    
    with p2:
        st.markdown("**2. Filter Popular**")
        min_support = st.slider("Min Support", 0.001, 0.2, 0.02, 0.001)
    
    with p3:
        st.markdown("**3. Association Rules Filter**")
        min_metric = st.selectbox("Metric", ["lift", "confidence"])
        min_threshold = st.slider("Threshold", 0.1, 10.0, 1.0, 0.1)

# --- 3. UTILITY FUNCTIONS ---
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

# --- Multiprocessing Wrapper ---
def process_wrapper(q, basket, min_sup, metric, thresh, algo):
    res, msg = calculate_mba_process(basket, min_sup, metric, thresh, algo)
    q.put((res, msg))

# --- 4. DATA SELECTION ---
st.divider()
st.header("📂 1. Choose Data Source")

data_source = st.radio("Data Source:", ["Upload File CSV", "Dataset Default (Online Retail II)"])
df = None
default_cols = {} 

if data_source == "Upload File CSV":
    uploaded_file = st.file_uploader("Upload File (Format CSV)", type=['csv'])
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.success("✅ Data loaded from uploaded file.")
            with st.expander("Show Sample Data"):
                st.dataframe(df.head())
        except Exception as e: st.error(f"Error: {e}")
else: 
    LOCAL_PATH = "dataset/online_retail_II.csv"
    if os.path.exists(LOCAL_PATH):
        try:
            df = load_data(LOCAL_PATH)
            default_cols = {'tx': 'Invoice', 'item': 'Description', 'cust': 'Customer ID', 'qty': 'Quantity', 'price': 'Price', 'date': 'InvoiceDate'}
            st.success(f"✅ Data Default Loaded from: {LOCAL_PATH}")
            with st.expander("Show Sample Data"):
                st.dataframe(df.head())
        except Exception as e: st.error(f"Error: {e}")
    else: st.warning(f"File not found in: {LOCAL_PATH}")

# --- 5. DATA PREPARATION ---
if df is not None:
    st.divider()
    st.header("🧹 2. Data Preparation")
    
    def get_index(columns, search_name):
        try: return list(columns).index(search_name)
        except: return 0

    c1, c2, c3 = st.columns(3)
    with c1:
        tx_col = st.selectbox("Transaction ID Column", df.columns, index=get_index(df.columns, default_cols.get('tx', '')))
    with c2:
        item_col = st.selectbox("Item Column", df.columns, index=get_index(df.columns, default_cols.get('item', '')))
    with c3:
        cust_col = st.selectbox("Customer ID Column", df.columns, index=get_index(df.columns, default_cols.get('cust', '')))
        
    with st.expander("Cleaning Options", expanded=True):
        rm_cancel = st.checkbox("Delete Cancelled ('C') Transaction", value=True)
        rm_null = st.checkbox("Delete Null Customer", value=True)
        rm_duplicates = st.checkbox("Delete Duplicate Rows", value=True)
        rm_items = st.checkbox("Delete Null Items", value=True)
        
    if st.button("Data Processing ⬇️"):
        df_clean = df.copy()
        if rm_null:
            df_clean = df_clean.dropna(subset=[cust_col])
            df_clean[cust_col] = df_clean[cust_col].astype(str).str.split('.').str[0]
        if rm_cancel:
            df_clean[tx_col] = df_clean[tx_col].astype(str)
            df_clean = df_clean[~df_clean[tx_col].str.contains("C", na=False)]
        if rm_duplicates:
            before_rows = df.shape[0]
            df = df.drop_duplicates()
            after_rows = df.shape[0]
        if rm_items:
            df_clean = df_clean.dropna(subset=[item_col])
        
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
            st.success(f"Data Ready! {df_clean.shape[0]} Rows.")
        except Exception as e: st.error(f"Convertion Error: {e}")

    # --- 6. RFM ---
    if 'df_clean' in st.session_state:
        st.divider()
        st.header("👥 3. Segmentation (RFM)")
        df_proc = st.session_state['df_clean']; cols = st.session_state['cols']
        
        if st.button("Run RFM ⬇️"):
            with st.spinner("Calculating RFM..."):
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
            st.header("🛒 4. MBA Analysis (Safe Mode)")
            f1, f2 = st.columns(2)
            country_col = next((c for c in df_proc.columns if 'country' in c.lower()), None)
            if country_col:
                with f1: sel_country = st.selectbox("Country", sorted(df_proc[country_col].astype(str).unique()))
            with f2: sel_seg = st.selectbox("Segment", rfm['Segment'].unique())
            
            if st.button("Start Analysis (With Timeout) 🚀"):
                target_cust = rfm[rfm['Segment'] == sel_seg].index
                df_target = df_proc[(df_proc[cols['cust']].isin(target_cust)) & (df_proc[country_col] == sel_country)]
                
                if df_target.empty:
                    st.warning("Data Empty.")
                else:
                    st.write(f"Analyzing **{df_target.shape[0]}** rows...")
                    
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
                    
                    with st.spinner(f"Calculating... (Max {timeout_limit} seconds)"):
                        p.join(timeout=timeout_limit)
                    
                    if p.is_alive():
                        p.terminate() # KILL PROCESS SEKARANG!
                        p.join()
                        st.error(f"⛔ **TIMEOUT!** Process killed due to > {timeout_limit} seconds.")
                        st.info("Tips: Reduce 'Limit Products' or increase 'Min Support'.")
                        st.info("or try running on a local computer with higher specifications.")
                    else:
                        if not result_queue.empty():
                            rules_res, msg = result_queue.get()
                            if rules_res is None:
                                st.warning(f"Error: {msg}")
                            else:
                                st.success(f"✅ Done! {len(rules_res)} Rules.")
                                # Fix PyArrow
                                rules_res['antecedents'] = rules_res['antecedents'].apply(lambda x: ', '.join(list(x)))
                                rules_res['consequents'] = rules_res['consequents'].apply(lambda x: ', '.join(list(x)))

                                t1, t2 = st.tabs(["Table", "Graph"])
                                with t1: st.dataframe(rules_res.sort_values('lift', ascending=False).head(10))
                                with t2: st.pyplot(plot_network_graph(rules_res))
                        else:
                            st.error("Process died without result (Out of Memory?).")