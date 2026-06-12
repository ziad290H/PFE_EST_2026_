import pandas as pd
import numpy as np

DATA_PATH = '/mnt/c/Users/hp/Desktop/PFE_2026/PFE_EST_FBS/pipeline_corrige_final_originale/data1/'
SAVE_PATH = '/mnt/c/Users/hp/Desktop/PFE_2026/PFE_EST_FBS/pipeline_corrige_final_originale/data1/'
def features_enginering():
    print("chargement de data")
    df = pd.read_csv(DATA_PATH + "df_cleaned.csv")
    print("Dtaset Disponible: ", df.shape)
    
    # ═══════════════════════════════════════════════════════════════════
    # 4.1 — Features TEMPORELLES  [CORRIGÉ — Remarque 14]
    # Dates brutes → Flags (0/1) + Délais (jours entiers)
    # ═══════════════════════════════════════════════════════════════════
    print('⏱️  Remarque 14 — Transformation dates → flags + délais...')

    # ── Conversion sécurisée ─────────────────────────────────────────
    DATE_COLS = [c for c in df.columns
                if '_at' in c and c not in ('returned_at_item', 'returned_at_order')]
    for col in DATE_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    ref_col = 'created_at_item' if 'created_at_item' in df.columns else 'created_at_order'
    date_coupure = df[ref_col].max()
    print(f'  📅 date_coupure = {date_coupure}')

    # ── FLAGS (0 = absent · 1 = présent) ─────────────────────────────
    ship_col  = next((c for c in ['shipped_at_item',   'shipped_at_order']   if c in df.columns), None)
    deliv_col = next((c for c in ['delivered_at_item', 'delivered_at_order'] if c in df.columns), None)

    if ship_col:
        df['a_ete_expedie'] = df[ship_col].notna().astype(int)
        print('  ✅ a_ete_expedie  (0/1)')

    if deliv_col:
        df['a_ete_livre'] = df[deliv_col].notna().astype(int)
        print('  ✅ a_ete_livre    (0/1)')

    # ── DÉLAIS (jours entiers) ────────────────────────────────────────
    if ship_col and ref_col in df.columns:
        df['delai_traitement_jours'] = (
            (df[ship_col] - df[ref_col]).dt.total_seconds().div(86400).round().astype('Int64')
        )
        print('  ✅ delai_traitement_jours  (NaN si non expédié)')

    if deliv_col and ship_col:
        df['delai_livraison_jours'] = (
            (df[deliv_col] - df[ship_col]).dt.total_seconds().div(86400).round().astype('Int64')
        )
        print('  ✅ delai_livraison_jours   (NaN si non livré)')

    # ── ANCIENNETÉ COMPTE CLIENT ──────────────────────────────────────
    user_date_col = 'created_at_user' if 'created_at_user' in df.columns else None
    if user_date_col:
        df['anciennete_compte_jours'] = (
            (date_coupure - df[user_date_col]).dt.total_seconds().div(86400).round().astype('Int64')
        )
        print('  ✅ anciennete_compte_jours')

    # ── FEATURES SAISONNIÈRES ────────────────────────────────────────
    if ref_col in df.columns:
        df['order_month']     = df[ref_col].dt.month.astype('Int64')
        df['order_dayofweek'] = df[ref_col].dt.dayofweek.astype('Int64')
        df['order_hour']      = df[ref_col].dt.hour.astype('Int64')
        df['is_weekend']      = (df['order_dayofweek'] >= 5).astype(int)
        df['order_quarter']   = df[ref_col].dt.quarter.astype('Int64')
        print('  ✅ order_month, order_dayofweek, order_hour, is_weekend, order_quarter')

    # ── SUPPRESSION DES DATES BRUTES ─────────────────────────────────
    dates_brutes = [c for c in df.columns
                    if '_at' in c and c not in ('returned_at_item', 'returned_at_order')]
    df.drop(columns=dates_brutes, inplace=True, errors='ignore')
    print(f'  🗑️  Dates brutes supprimées : {dates_brutes}')

    print(f'\n✅ Étape 4.1 corrigée (Remarque 14) — shape: {df.shape}')


    # ═══════════════════════════════════════════════
    # 4.2 — Features PRODUIT
    # Ratio prix, marge, segment prix
    # ═══════════════════════════════════════════════
    print('📦 Features produit...')

    # Ratio prix de vente / prix catalogue
    if 'sale_price' in df.columns and 'retail_price' in df.columns:
        df['price_ratio'] = (df['sale_price'] / df['retail_price'].replace(0, np.nan)).fillna(1.0)
        df['discount_pct'] = ((df['retail_price'] - df['sale_price']) / df['retail_price'].replace(0, np.nan)).fillna(0.0).clip(0, 1)
        print(f'  ✅ price_ratio, discount_pct')

    # Marge brute estimée
    if 'sale_price' in df.columns and 'cost' in df.columns:
        df['gross_margin'] = df['sale_price'] - df['cost']
        df['margin_rate']  = (df['gross_margin'] / df['sale_price'].replace(0, np.nan)).fillna(0.0)
        print(f'  ✅ gross_margin, margin_rate')

    # Segment de prix (quantiles en 4 groupes)
    if 'sale_price' in df.columns:
        df['price_segment'] = pd.qcut(df['sale_price'], q=4, labels=['bas','moyen_bas','moyen_haut','haut'])
        df['price_segment'] = df['price_segment'].astype(str)
        print(f'  ✅ price_segment (4 segments)')

    print(f'\n✅ Features produit OK — shape: {df.shape}')

    
    # ═══════════════════════════════════════════════════════════════════
    # 4.3 — Features COMPORTEMENT CLIENT  [CORRIGÉ — Remarque 15]
    # Leave-One-Out encoding pour éliminer le data leakage
    #
    # PROBLÈME ORIGINAL :
    #   user_return_rate_hist = groupby(user_id)[returned].mean()
    #   → inclut la ligne elle-même → le modèle voit sa propre cible
    #   → AUC = 1.0 garanti, résultat sans valeur
    #
    # SOLUTION — Leave-One-Out avec lissage :
    #   taux_i = (sum(Y_groupe) − Y_i) / (count_groupe − 1)
    # ═══════════════════════════════════════════════════════════════════
    print('👤 Remarque 15 — Features comportement (leave-one-out)...')

    def loo_target_encode(df, group_col, target_col, new_col, smoothing=10):
        """
        Leave-One-Out target encoding avec lissage.
        taux_lisse_i = (n * taux_LOO + smoothing * moy_globale) / (n + smoothing)
        Groupes de taille 1 → moyenne globale.
        """
        global_mean = df[target_col].mean()
        grp = df.groupby(group_col)[target_col].agg(['sum', 'count']).reset_index()
        grp.columns = [group_col, '_sum', '_count']
        tmp = df[[group_col, target_col]].merge(grp, on=group_col, how='left')
        loo_sum   = tmp['_sum']   - tmp[target_col]
        loo_count = tmp['_count'] - 1
        loo_rate  = loo_sum / loo_count.replace(0, np.nan)
        smoothed  = (loo_count * loo_rate + smoothing * global_mean) / (loo_count + smoothing)
        df[new_col] = smoothed.fillna(global_mean).values
        return df

    # ── user : taux retour LOO + total commandes / retours ───────────
    if 'user_id' in df.columns and 'returned' in df.columns:
        grp = df.groupby('user_id')['returned'].agg(['sum', 'count']).reset_index()
        grp.columns = ['user_id', '_u_sum', '_u_count']
        tmp = df[['user_id', 'returned']].merge(grp, on='user_id', how='left')
        # Fix: Explicitly use .values for element-wise operation to avoid index misalignment issues
        df['user_total_orders']  = tmp['_u_count'].values - 1
        df['user_total_returns'] = tmp['_u_sum'].values   - df['returned'].values
        df = loo_target_encode(df, 'user_id',   'returned', 'user_return_rate_hist', smoothing=10)
        print('  ✅ user_return_rate_hist (LOO) + user_total_orders + user_total_returns')

    # ── catégorie : taux retour LOO ──────────────────────────────────
    if 'category' in df.columns and 'returned' in df.columns:
        df = loo_target_encode(df, 'category', 'returned', 'cat_return_rate',   smoothing=5)
        print('  ✅ cat_return_rate   (LOO lissé)')

    # ── marque : taux retour LOO ─────────────────────────────────────
    if 'brand' in df.columns and 'returned' in df.columns:
        df = loo_target_encode(df, 'brand',    'returned', 'brand_return_rate', smoothing=5)
        print('  ✅ brand_return_rate (LOO lissé)')

    print(f'\n✅ Features comportement corrigées — shape: {df.shape}')

    # ── Diagnostic corrélation (vérification leakage) ────────────────
    print('\n📊 Diagnostic corrélation features vs Y (returned)...')

    import seaborn as sns
    import matplotlib.pyplot as plt

    num_cols    = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'returned']
    corr_with_y = (df[num_cols + ['returned']].corr()['returned']
                .drop('returned').abs().sort_values(ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, len(corr_with_y) * 0.3 + 2)))
    fig.suptitle('Diagnostic Leakage — Corrélation features vs Y (returned)',
                fontsize=13, fontweight='bold')

    colors = ['#e74c3c' if v > 0.9 else '#f39c12' if v > 0.7 else '#27ae60'
            for v in corr_with_y.values]
    axes[0].barh(corr_with_y.index[::-1], corr_with_y.values[::-1], color=colors[::-1])
    axes[0].axvline(0.9, color='red',    linestyle='--', lw=1.5, label='r > 0.9 → leakage certain')
    axes[0].axvline(0.7, color='orange', linestyle='--', lw=1.5, label='r > 0.7 → à inspecter')
    axes[0].set_xlabel('|corrélation| avec returned')
    axes[0].set_title('Corrélation de chaque feature avec Y')
    axes[0].legend(fontsize=8)

    top15 = corr_with_y.head(15).index.tolist()
    mask  = np.triu(np.ones((len(top15), len(top15)), dtype=bool)) # Corrected mask creation
    sns.heatmap(df[top15].corr(), ax=axes[1], mask=mask, cmap='RdYlGn_r',
                center=0, vmin=-1, vmax=1, annot=True, fmt='.2f',
                annot_kws={'size': 7}, linewidths=0.3, square=True)
    axes[1].set_title('Corrélation inter-features (Top 15)')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    #plt.show()

    print('\n' + '─'*55)
    print('  |r| avec Y (returned) — top 20')
    print('─'*55)
    for feat, val in corr_with_y.head(20).items():
        flag = '🔴 LEAKAGE' if val > 0.9 else '🟠 À inspecter' if val > 0.7 else '🟢 OK'
        print(f'  {feat:<35} {val:.4f}  {flag}')

    leakage_feats = corr_with_y[corr_with_y > 0.9].index.tolist()
    if leakage_feats:
        print(f'\n⚠️  Features supprimées (r > 0.9) : {leakage_feats}')
        df.drop(columns=leakage_feats, inplace=True, errors='ignore')
    else:
        print('\n✅ Aucune feature avec r > 0.9 — leakage corrigé.')
        from sklearn.preprocessing import LabelEncoder
    print('🔠 Encodage catégoriel...')

    # ⚠️  CORRECTION LEAKAGE :
    #   item_status  contient "Returned"  → encodé, il prédit parfaitement Y  → AUC = 1.00
    #   order_status contient "Returned"  → même problème
    #   Ces deux colonnes sont donc EXCLUES de l'encodage (et de la modélisation)
    LEAKAGE_COLS = ['item_status', 'order_status']

    CAT_COLS = [c for c in ['category','brand','department','user_gender',
                            'country','state','city','user_traffic_source',
                            'price_segment']
                if c in df.columns]

    # Supprimer les colonnes de leakage du dataframe AVANT l'encodage
    for col in LEAKAGE_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f'  🗑️  {col} supprimé (leakage — contient "Returned")')

    le_dict = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        print(f'  ✅ {col:<25} → {col}_enc  ({df[col].nunique()} classes)')

    print(f'\n✅ Encodage terminé (leakage supprimé) — shape: {df.shape}')

    # ═══════════════════════════════════════════════
    # 4.5 — Sélection des features FINALES  [CORRIGÉ — Remarques 14, 15 + Leakage status]
    # ═══════════════════════════════════════════════
    print('🎯 Sélection des features (post-correction)...')

    NUM_FEATURES = [c for c in [
        # Prix & marges
        'sale_price', 'retail_price', 'cost', 'num_of_item', 'age',
        'price_ratio', 'discount_pct', 'gross_margin', 'margin_rate',
        # Délais (Remarque 14 — remplacent delivery_days / days_to_ship)
        'delai_traitement_jours',
        'delai_livraison_jours',
        # Flags expédition / livraison (Remarque 14)
        'a_ete_expedie',
        'a_ete_livre',
        # Ancienneté compte (Remarque 14)
        'anciennete_compte_jours',
        # Comportement user (Remarque 15 — LOO corrigé)
        'user_return_rate_hist',
        'user_total_returns',
        'user_total_orders',
        # Taux catégorie / marque (Remarque 15 — LOO corrigé)
        'cat_return_rate',
        'brand_return_rate',
        # Saisonnalité
        'order_month', 'order_dayofweek', 'order_hour', 'is_weekend', 'order_quarter',
    ] if c in df.columns]

    # ── Colonnes encodées (sans leakage) ────────────────────────────
    # item_status_enc et order_status_enc sont INTERDITES : elles contiennent
    # "Returned" et causent AUC = 1.00.  Elles ont été supprimées en étape 4.4.
    LEAKAGE_ENC = ['item_status_enc', 'order_status_enc']
    ENC_FEATURES = [c for c in df.columns
                    if c.endswith('_enc') and c not in LEAKAGE_ENC]

    FEATURES = list(dict.fromkeys(NUM_FEATURES + ENC_FEATURES))
    TARGET   = 'returned'

    # ── Garde-fou final : vérifier qu'aucune colonne de leakage ne survit ────────
    LEAKAGE_SUSPECTS = [
        'item_status', 'order_status',          # colonnes brutes
        'item_status_enc', 'order_status_enc',  # colonnes encodées
        'returned_at_item', 'returned_at_order' # dates de retour
    ]
    leaked = [c for c in FEATURES if c in LEAKAGE_SUSPECTS]
    if leaked:
        print(f'⚠️  ATTENTION — colonnes de leakage détectées et retirées : {leaked}')
        FEATURES = [c for c in FEATURES if c not in leaked]
    else:
        print('✅ Aucune colonne de leakage dans FEATURES')

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X = X.fillna(X.median(numeric_only=True))

    print(f'\n  Features totales : {len(FEATURES)}')
    print(f'    Numériques     : {len(NUM_FEATURES)}')
    print(f'    Encodées       : {len(ENC_FEATURES)}')
    print(f'  Exemples         : {len(X):,}')
    print(f'  Taux retour      : {y.mean():.3%}')
    print(f'\n  Liste des features :')
    for i, f in enumerate(FEATURES, 1):
        print(f'    {i:>2}. {f}')
    print(f'\n✅ Étape 4 corrigée — X:{X.shape}, y:{y.shape}')
    print('💡 AUC attendu après correction : 0.70 – 0.85')
    print('   Un AUC > 0.95 indique encore un leakage résiduel.')


    print("----------here is how is the data is now-----\n ", df.shape)
    return (X, y)