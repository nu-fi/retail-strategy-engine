# Simpan sebagai: mba_utils.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

def calculate_mba_process(basket, min_sup, metric, thresh, algo='fpgrowth'):
    """
    Fungsi ini berjalan di Process terpisah (bisa dibunuh).
    Menggunakan fpgrowth karena lebih cepat dari apriori.
    """
    try:
        # 1. Frequent Itemsets
        # Kita pakai fpgrowth karena jauh lebih hemat memori dibanding apriori
        if algo == 'apriori':
            freq_items = apriori(basket, min_support=min_sup, use_colnames=True)
        else:
            freq_items = fpgrowth(basket, min_support=min_sup, use_colnames=True)
        
        if freq_items.empty:
            return None, "Kosong (Support ketinggian)"
            
        # 2. Association Rules
        rules = association_rules(freq_items, metric=metric, min_threshold=thresh)
        
        if rules.empty:
            return None, "Kosong (Threshold ketinggian)"
            
        return rules, "Sukses"
        
    except Exception as e:
        return None, str(e)