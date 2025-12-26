import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import networkx as nx
import matplotlib.pyplot as plt
import os
import multiprocessing 
from mba_utils import calculate_mba_process 

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Retail Strategy Tool", layout="wide")

st.title("Customer Segmentation & Market Basket Analysis")
st.markdown("""
A comprehensive tool to segment your customers using RFM analysis and uncover hidden purchasing patterns with Market Basket Analysis (MBA). Empower your retail strategy with data-driven insights! The perfect companion for online and offline retailers aiming to boost sales and customer loyalty. This app is built with Streamlit and leverages powerful libraries like Pandas, NetworkX, and mlxtend.
""")

st.markdown("""
**Workflow Linear:** Choose Data -> Cleaning -> Segmentation (RFM) -> Market Basket Analysis (MBA).
""")

# --- 3. UTILITY FUNCTIONS ---
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='cp1252')
    return data

def plot_network_graph(rules, n_rules=20):
    """
    Fungsi untuk memvisualisasikan Network Graph dari Association Rules
    dengan fitur: 
    1. Pemendekan nama produk (Depan + Akronim Tengah + Belakang).
    2. Label pintar (di atas/bawah node agar tidak tertumpuk).
    3. Garis melengkung (curved edges).
    """
    
    # --- 1. PREPARASI DATA ---
    
    # Ambil n_rules teratas berdasarkan Lift
    top_rules = rules.sort_values("lift", ascending=False).head(n_rules).copy()

    # Fungsi internal untuk memendekkan nama
    def shorten_middle_acronym(name):
        # Handle jika input berupa frozenset atau string
        s = list(name)[0] if isinstance(name, (frozenset, set)) else str(name)
        words = s.split()
        
        # Kalau cuma 1 atau 2 kata, biarkan utuh
        if len(words) <= 2:
            return s
        
        # LOGIKA: DEPAN + TENGAH (AKRONIM) + BELAKANG
        first_word = words[0]
        last_word = words[-1]
        
        # Ambil kata-kata di tengah, ambil huruf depannya saja, tambah titik
        middle_acronym = "".join([w[0] + "." for w in words[1:-1]])
        
        return f"{first_word} {middle_acronym} {last_word}"

    # Terapkan fungsi pemendek nama
    top_rules['ant_short'] = top_rules['antecedents'].apply(shorten_middle_acronym)
    top_rules['con_short'] = top_rules['consequents'].apply(shorten_middle_acronym)

    # Filter Self-Loop (Hapus jika Asal == Tujuan)
    top_rules = top_rules[top_rules['ant_short'] != top_rules['con_short']]

    # --- 2. BANGUN GRAPH ---
    G = nx.DiGraph()
    for i, row in top_rules.iterrows():
        G.add_edge(row['ant_short'], row['con_short'], 
                   weight=row['lift'], 
                   confidence=row['confidence'])

    # --- 3. VISUALISASI ---
    # Buat figure dan axes agar bisa dikembalikan (return fig)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Layout: k=3.5 agar node menyebar
    pos = nx.spring_layout(G, k=3.5, seed=123) 

    # A. Gambar Node
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#FF6B6B', alpha=0.9, ax=ax)

    # B. Gambar Edge (Garis Melengkung)
    edges = G.edges()
    if edges:
        weights = [G[u][v]['weight'] for u, v in edges]
        # Normalisasi ketebalan garis (opsional, dikali 0.5 biar gak ketebalan)
        width_lines = [w * 0.5 for w in weights]
    else:
        width_lines = 1

    nx.draw_networkx_edges(G, pos, 
                           width=width_lines, 
                           edge_color='#4ECDC4', 
                           arrowstyle='->', 
                           arrowsize=20,
                           connectionstyle="arc3,rad=0.1", # Garis melengkung
                           ax=ax)

    # C. Gambar Label Confidence
    edge_labels = {(u, v): f"Conf: {d['confidence']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    # D. Gambar Label Produk (Logika Cerdas: Atas/Bawah)
    # Hitung titik tengah vertikal grafik
    y_values = [y for x, y in pos.values()]
    if y_values:
        y_center = sum(y_values) / len(y_values)
    else:
        y_center = 0

    label_positions = {}
    for node, (x, y) in pos.items():
        # Jika node di atas rata-rata, label geser ke atas (+0.08)
        # Jika node di bawah rata-rata, label geser ke bawah (-0.08)
        if y > y_center:
            label_positions[node] = (x, y + 0.08)
        else:
            label_positions[node] = (x, y - 0.08)

    nx.draw_networkx_labels(G, label_positions, 
                            font_size=9, 
                            font_family='sans-serif', 
                            font_weight='bold',
                            font_color='black',
                            ax=ax)

    # Final settings pada axes
    ax.set_title("Product Association Network (Top Rules)", fontsize=16)
    ax.axis('off')
    ax.margins(0.2) # Margin agar label tidak terpotong

    return fig

# --- Multiprocessing Wrapper ---
def process_wrapper(q, basket, min_sup, metric, thresh, algo):
    res, msg = calculate_mba_process(basket, min_sup, metric, thresh, algo)
    q.put((res, msg))

# --- 4. DATA SELECTION ---
st.divider()
st.header("1. Choose Data Source")

data_source = st.radio("Data Source:", ["Upload File CSV", "Dataset Default (Online Retail II)"])
df = None
default_cols = {} 

if data_source == "Upload File CSV":
    uploaded_file = st.file_uploader("Upload File (Format CSV)", type=['csv'])
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.success("Data loaded from uploaded file.")
            with st.expander("Show Sample Data"):
                st.dataframe(df.head())
        except Exception as e: st.error(f"Error: {e}")
else: 
    LOCAL_PATH = "dataset/online_retail_II.csv"
    if os.path.exists(LOCAL_PATH):
        try:
            df = load_data(LOCAL_PATH)
            default_cols = {'tx': 'Invoice', 'item': 'Description', 'cust': 'Customer ID', 'qty': 'Quantity', 'price': 'Price', 'date': 'InvoiceDate'}
            st.success(f"Data Default Loaded from: {LOCAL_PATH}")
            with st.expander("Show Sample Data"):
                st.dataframe(df.head())
        except Exception as e: st.error(f"Error: {e}")
    else: st.warning(f"File not found in: {LOCAL_PATH}")

# --- 5. DATA PREPARATION ---
if df is not None:
    st.divider()
    st.header("2. Data Preparation")
    
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
        rm_cancel = st.checkbox("Delete Cancelled Transaction ('C') and Adjusting Transactions ('A')", value=True)
        rm_null = st.checkbox("Delete Null Values", value=True)
        rm_duplicates = st.checkbox("Delete Duplicate Rows", value=True)
        
    if st.button("Data Processing"):
        df_clean = df.copy()
        
        # Fallback Cols
        if not default_cols:
             qty_col = next((c for c in df.columns if 'quantity' in c.lower()), df.columns[3])
             price_col = next((c for c in df.columns if 'price' in c.lower()), df.columns[5])
             date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[4])
        else:
             qty_col = default_cols.get('qty'); price_col = default_cols.get('price'); date_col = default_cols.get('date')

        if rm_null:
            critical_cols = [tx_col, item_col, cust_col, qty_col, price_col, date_col]
            df_clean = df_clean.dropna(subset=critical_cols)
            df_clean[cust_col] = df_clean[cust_col].astype(str).str.split('.').str[0]
        if rm_cancel:
            df_clean[tx_col] = df_clean[tx_col].astype(str)
            df_clean = df_clean[~df_clean[tx_col].str.contains("C|A", na=False)]
        if rm_duplicates:
            before_rows = df.shape[0]
            df = df.drop_duplicates()
            after_rows = df.shape[0]

        try:
            # 1. Type Conversion
            df_clean[date_col] = pd.to_datetime(df_clean[date_col])
            df_clean[qty_col] = pd.to_numeric(df_clean[qty_col])
            df_clean[price_col] = pd.to_numeric(df_clean[price_col])
            
            # 2. Calculation & Filtering
            df_clean['TotalAmount'] = df_clean[qty_col] * df_clean[price_col]
            df_clean = df_clean[(df_clean[qty_col] > 0) & (df_clean[price_col] > 0)]
            
            # 3. Save to Session State
            st.session_state['df_clean'] = df_clean
            st.session_state['cols'] = {'tx': tx_col, 'item': item_col, 'cust': cust_col, 'date': date_col}
            
            # Clear previous RFM results if new data is processed
            if 'rfm_result' in st.session_state:
                del st.session_state['rfm_result']
            
            st.success(f"Data Ready! {df_clean.shape[0]} Rows processed.")

            # ==========================================
            # QUICK EDA (NEW FEATURE)
            # ==========================================
            with st.expander("Quick Data Overview (Click to Close)", expanded=True):
                
                # A. Summary Metrics
                m1, m2, m3 = st.columns(3)
                total_sales = df_clean['TotalAmount'].sum()
                total_tx = df_clean[tx_col].nunique()
                total_items = df_clean[item_col].nunique()
                
                m1.metric("Total Sales", f"${total_sales:,.0f}")
                m2.metric("Total Transactions", f"{total_tx:,}")
                m3.metric("Unique Products", f"{total_items:,}")
                
                st.divider()

                # B. Top 10 Best-Selling Products
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Top 10 Best-Selling Products")
                    # Group by Item, Sum Quantity, Sort Descending, Take Top 10
                    top_products = df_clean.groupby(item_col)[qty_col].sum().sort_values(ascending=False).head(10)
                    st.bar_chart(top_products)
                    
                # C. Sales Trend Over Time
                with c2:
                    st.subheader("Daily Sales Trend")
                    # Group by Date, Sum TotalAmount
                    daily_sales = df_clean.groupby(pd.Grouper(key=date_col, freq='D'))['TotalAmount'].sum()
                    st.line_chart(daily_sales)

                # D. Data Preview
                st.caption("First 5 rows of cleaned data:")
                st.dataframe(df_clean.head())

        except Exception as e: 
            st.error(f"Conversion Error: {e}")

    # --- 6. RFM ---
    if 'df_clean' in st.session_state:
        
        st.divider()
        st.header("3. Segmentation (RFM)")
        df_proc = st.session_state['df_clean']; cols = st.session_state['cols']
        
        if st.button("Start RFM Analysis") or 'rfm_result' not in st.session_state:
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
            # with st.expander("Show RFM Sample Data"):
            #     st.dataframe(rfm.head())
            
            with st.expander("RFM Segmentation Explanation", expanded=False):
                
                rfm = st.session_state['rfm_result']
                # --- RFM DASHBOARD ---
                st.divider()
                st.markdown("### The Customer Landscape")
                st.info("The data speaks: Who are your VIPs, and who is drifting away?")
                # 1. Calculate & Identify The "Real" VIPs (Dynamic)
                # Kita cari segmen mana yang total belanja (sum Monetary)-nya paling besar
                segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
                
                # Ambil nama segmen Juara & Nilai duitnya
                top_segment_name = segment_revenue.index[0]
                top_segment_rev = segment_revenue.iloc[0]
                top_segment_count = rfm[rfm['Segment'] == top_segment_name].shape[0]
                
                # Global Stats
                total_cust = rfm.shape[0]
                total_revenue = rfm['Monetary'].sum()
                n_atrisk = rfm[rfm['Segment'] == 'At Risk'].shape[0]
                
                # Display Metrics
                m1, m2, m3 = st.columns(3)
                
                # Metric 1: Total Omset Toko (Dollar Formatted)
                m1.metric(
                    label="Total Revenue",
                    value=f"${total_revenue:,.0f}", 
                    delta=f"{total_cust:,} Customers"
                )
                
                # Metric 2: Dynamic VIP
                m2.metric(
                    label=f"Top Spenders: {top_segment_name}", 
                    value=f"${top_segment_rev:,.0f}",          
                    delta=f"{top_segment_count:,} People",      
                    delta_color="normal"
                )
                
                # Metric 3: Warning Signal
                m3.metric(
                    label="At Risk Customers", 
                    value=f"{n_atrisk:,}", 
                    delta="Need Retention Strategy", 
                    delta_color="inverse"
                )
                

                # 2. Visual Story
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.markdown("#### Segmentation Distribution")
                    # Calculate Revenue per Segment for deeper insight
                    # Sort descending agar yang paling kaya ada di paling atas
                    rev_per_seg = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
                    st.bar_chart(rev_per_seg)
                    with st.expander("ℹ️ Guide: Segment Definitions (Click to Open)"):
                        st.markdown("""
                        | Segment | Description | Strategy |
                        | :--- | :--- | :--- |
                        | 🏆 **Champions** | Recent, frequent, high spenders. | Reward them. They are your VIPs. |
                        | 💎 **Loyal** | Regular buyers with good spend. | Upsell higher value products. |
                        | 🌱 **New Customers** | Bought recently, but not often yet. | Build relationship, offer welcome deals. |
                        | 🧐 **Potential** | Moderate recency and frequency. | Offer limited-time discounts to convert. |
                        | 🚨 **At Risk** | Big spenders who haven't bought lately. | Send 'We Miss You' emails/Win-back promos. |
                        | 💤 **Lost** | Long time no see, low frequency. | Ignore or use low-cost automated emails. |
                        """)
                        st.caption("*The chart above shows which segment contributes the most Revenue (Monetary).*")
                    
                with c2:
                    st.markdown("#### Quick Insights")
                    
                    # --- LOGIKA DINAMIS ---
                    # Ambil nama segmen nomor 1 (paling banyak duitnya)
                    top_seg_name = rev_per_seg.index[0]
                    
                    # Hitung rata-rata belanja segmen tersebut vs Rata-rata toko
                    top_seg_avg = rfm[rfm['Segment'] == top_seg_name]['Monetary'].mean()
                    global_avg = rfm['Monetary'].mean()
                    
                    # Hitung seberapa 'sultan' mereka (Multiplier)
                    multiplier = top_seg_avg / global_avg if global_avg > 0 else 1
                    
                    st.success(f"**Dominant Segment: {top_seg_name}**")
                    
                    st.write(f"""
                    - **{top_seg_name}** have an average spend of **${top_seg_avg:,.0f}**.
                    - This is **{multiplier:.1f}x higher** than the average customer!
                    - **Action:** Prioritize budget to retain **{top_seg_name}** and reactivate 'At Risk' users.
                    - **Note:** Don't overspend on 'Lost' customers.
                    """)
                

            # --- 7. MBA (MULTIPROCESSING) ---
            st.divider()
            st.header("4. MBA Analysis per Segment")
            f1, f2 = st.columns(2)
            country_col = next((c for c in df_proc.columns if 'country' in c.lower()), None)
            if country_col:
                with f1: sel_country = st.selectbox("Country", sorted(df_proc[country_col].astype(str).unique()))
            with f2: sel_seg = st.selectbox("Segment", rfm['Segment'].unique())

            # --- 2. PARAMETER SETTING ---
            with st.expander("Parameter configuration for MBA analysis.", expanded=False):
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
            
            if st.button("Start MBA Analysis"):
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
                        st.error(f"**TIMEOUT!** Process killed due to > {timeout_limit} seconds.")
                        st.warning("Tips: Reduce 'Limit Products' or increase 'Min Support' or reduce the threshold or try running on a local computer with higher specifications.")
                    else:
                        if not result_queue.empty():
                            rules_res, msg = result_queue.get()
                            if rules_res is None:
                                st.warning(f"Error: {msg}")
                            else:
                                st.divider()
                                st.markdown("### The Secret Life of Shopping Baskets")
                                
                                # Get Top 1 Rule for the "Hero Section"
                                top_rule = rules_res.sort_values('lift', ascending=False).iloc[0]
                                
                                # Safe conversion from frozenset to string
                                ant_name = list(top_rule['antecedents'])[0] if isinstance(top_rule['antecedents'], (set, frozenset)) else str(top_rule['antecedents'])
                                con_name = list(top_rule['consequents'])[0] if isinstance(top_rule['consequents'], (set, frozenset)) else str(top_rule['consequents'])
                                
                                lift_val = top_rule['lift']
                                conf_val = top_rule['confidence'] * 100
                                
                                # The "Hero Section"
                                st.success(f"""
                                #### Top Pattern Discovery (The Golden Rule)
                                
                                Customers in this segment show a strong behavioral pattern:
                                
                                > **"If they buy `{ant_name}`..."**
                                >
                                > **"...There is a {conf_val:.1f}% probability they will ALSO buy `{con_name}`."**
                                
                                **Relationship Strength (Lift): {lift_val:.2f}x** *(Meaning: These two items are {lift_val:.2f} times more likely to be bought together than by random chance.)*
                                """)

                                # Automated Business Recommendations
                                st.markdown("#### Strategic Recommendations:")
                                c_strat1, c_strat2 = st.columns(2)
                                with c_strat1:
                                    st.info(f"**Create an Exclusive Bundle**\n\nCombine `{ant_name}` + `{con_name}` into a single SKU/Package. Offer a small incentive (e.g., 5% off) for this bundle to increase Average Order Value (AOV).")
                                with c_strat2:
                                    st.warning(f"**Optimize Layout (UX/Shelf)**\n\nDon't separate them! If online, ensure `{con_name}` appears in the 'You May Also Like' section when a user views `{ant_name}`.")

                                # Detailed Data (Tabs)
                                st.write("---")
                                st.write("#### Explore More Patterns")
                                
                                # Update the tabs list to include the new one
                                tab1, tab2, tab3 = st.tabs(["📋 Scenario Table", "🕸️ Network Graph", "📊 Metric Comparison"])
                                
                                # --- TAB 1: TABLE (Existing) ---
                                with tab1:
                                    # (Your existing table code here)
                                    display_df = rules_res.sort_values('lift', ascending=False).head(10).copy()
                                    display_df['antecedents'] = display_df['antecedents'].apply(lambda x: list(x)[0] if isinstance(x, (set, frozenset)) else str(x))
                                    display_df['consequents'] = display_df['consequents'].apply(lambda x: list(x)[0] if isinstance(x, (set, frozenset)) else str(x))
                                    display_df = display_df[['antecedents', 'consequents', 'confidence', 'lift']]
                                    display_df.columns = ['If Buying...', '...Then Buy', 'Probability (%)', 'Lift Strength']
                                    display_df['Probability (%)'] = (display_df['Probability (%)'] * 100).round(1)
                                    display_df['Lift Strength'] = display_df['Lift Strength'].round(2)
                                    st.dataframe(display_df)

                                # --- TAB 2: GRAPH (Existing) ---
                                with tab2:
                                    st.caption("Thicker arrows indicate a stronger relationship.")
                                    fig = plot_network_graph(rules_res)
                                    st.pyplot(fig)

                                # --- TAB 3: BAR CHART ---
                                with tab3:
                                    st.markdown("### Lift vs. Confidence Comparison")
                                    st.caption("This chart compares the normalized strength (Lift) vs. probability (Confidence) of random rules.")

                                    # 1. Safety: Ensure we don't sample more than available rules
                                    sample_size = min(10, len(rules_res))
                                    
                                    # 2. Random Sampling
                                    rules_random = rules_res.sample(sample_size, random_state=42)
                                    
                                    # 3. Data Prep (User's Logic)
                                    # We use .values instead of .to_numpy() for safer compatibility, but logic remains same
                                    rules_lift = rules_random['lift'].values
                                    # Avoid division by zero if max is 0
                                    if rules_lift.max() > 0:
                                        rules_lift = rules_lift / rules_lift.max()
                                        
                                    rules_conf = rules_random['confidence'].values
                                    if rules_conf.max() > 0:
                                        rules_conf = rules_conf / rules_conf.max()
                                    
                                    # 4. Plotting
                                    fig_bar, ax = plt.subplots(figsize=(12, 6), dpi=100)
                                    width = 0.40
                                    indices = np.arange(len(rules_random))
                                    
                                    ax.bar(indices - 0.2, rules_lift, width, color='black', label='Lift (Normalized)')
                                    ax.bar(indices + 0.2, rules_conf, width, hatch='//', edgecolor='black', facecolor='white', label='Confidence (Normalized)')
                                    
                                    ax.set_xlabel('Rule Index (Random Sample)')
                                    ax.set_ylabel('Normalized Metric Value')
                                    ax.set_title('Comparison of Metric Strength')
                                    ax.set_xticks(indices)
                                    ax.legend()
                                    
                                    st.pyplot(fig_bar)
                                    
                                    with st.expander("See Rule Details for these Bars"):
                                        # 1. Buat copy agar tidak merusak data asli
                                        clean_df = rules_random.copy()
                                        
                                        # 2. Hapus kata 'frozenset' dengan mengambil isi set-nya
                                        # Logika: Ubah set jadi list, lalu gabungkan dengan koma
                                        clean_df['antecedents'] = clean_df['antecedents'].apply(lambda x: ', '.join(list(x)))
                                        clean_df['consequents'] = clean_df['consequents'].apply(lambda x: ', '.join(list(x)))
                                        
                                        # 3. Rapikan angka desimal
                                        clean_df['lift'] = clean_df['lift'].round(2)
                                        clean_df['confidence'] = clean_df['confidence'].round(2)
                                        
                                        # 4. Tampilkan kolom yang sudah bersih
                                        st.dataframe(clean_df[['antecedents', 'consequents', 'lift', 'confidence']])
                        else:
                            st.error("Process died without result (Out of Memory?).")