# ═══════════════════════════════════════════════
# 1.3 Imports & Configuration globale
# ═══════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, json, joblib
from pathlib import Path
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi'       : 120,
    'axes.spines.top'  : False,
    'axes.spines.right': False
})
sns.set_theme(style='whitegrid', palette='husl')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.4f}'.format)


#DATA_PATH = '/mnt/c/Users/hp/Desktop/PFE_2026/PFE_EST_FBS/pipeline_corrige_final_originale/data1/'
#SAVE_PATH = '/mnt/c/Users/hp/Desktop/PFE_2026/PFE_EST_FBS/pipeline_corrige_final_originale/output/'



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data1"
SAVE_PATH = BASE_DIR / "output"
os.makedirs(SAVE_PATH, exist_ok=True)

print(f'📁 DATA_PATH : {DATA_PATH}')
print(f'📁 SAVE_PATH : {SAVE_PATH}')
print('✅ Imports OK')

# ═══════════════════════════════════════════════
# 1.4 Chargement des 7 fichiers CSV
# ═══════════════════════════════════════════════
print('📥 Chargement...')

orders       = pd.read_csv(DATA_PATH + 'orders.csv')
order_items  = pd.read_csv(DATA_PATH + 'order_items.csv')
products     = pd.read_csv(DATA_PATH + 'products.csv')
users        = pd.read_csv(DATA_PATH + 'users.csv')
events       = pd.read_csv(DATA_PATH + 'events.csv')
inventory    = pd.read_csv(DATA_PATH + 'inventory_items.csv')
dist_centers = pd.read_csv(DATA_PATH + 'distribution_centers.csv')

tables = {
    'orders'              : orders,
    'order_items'         : order_items,
    'products'            : products,
    'users'               : users,
    'events'              : events,
    'inventory_items'     : inventory,
    'distribution_centers': dist_centers
}

print(f'\n{"Table":<25} {"Lignes":>10} {"Colonnes":>10}')
print('─' * 48)
for name, df_ in tables.items():
    print(f'{name:<25} {df_.shape[0]:>10,} {df_.shape[1]:>10}')
print('\n✅ Étape 1 terminée')

#Etape 2 Exploration des Donnees(EDA)
# ═══════════════════════════════════════════════
# 2.1 Création de la variable cible : returned
# returned = 1 si returned_at non-null
# returned = 0 si returned_at null
# ═══════════════════════════════════════════════
if 'returned_at' in order_items.columns:
    order_items['returned'] = order_items['returned_at'].notnull().astype(int)
    print('✅ Cible créée depuis : returned_at')
elif 'status' in order_items.columns:
    order_items['returned'] = (order_items['status'] == 'Returned').astype(int)
    print('✅ Cible créée depuis : status == Returned')

return_rate = order_items['returned'].mean()
counts      = order_items['returned'].value_counts()

print(f'\n📊 Variable cible :')
print(f'   Non retournés (0) : {counts.get(0,0):>8,}  ({counts.get(0,0)/len(order_items):.1%})')
print(f'   Retournés     (1) : {counts.get(1,0):>8,}  ({counts.get(1,0)/len(order_items):.1%})')
print(f'   Taux de retour    : {return_rate:.2%}')
print(f'   Ratio déséquilibre: {counts.get(0,1)/counts.get(1,1):.1f}:1')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Variable Cible — Déséquilibre de classes', fontsize=13, fontweight='bold')

axes[0].bar(['Non retourné (0)','Retourné (1)'], counts.values,
            color=['#27ae60','#e74c3c'], edgecolor='white', linewidth=1.5, width=0.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v*1.01, f'{v:,}\n({v/len(order_items):.1%})',
                 ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('Comptage')
axes[0].set_ylabel('Nombre de lignes')

axes[1].pie(counts.values,
            labels=[f'Non retourné\n{counts.get(0,0):,}', f'Retourné\n{counts.get(1,0):,}'],
            colors=['#27ae60','#e74c3c'], autopct='%1.1f%%',
            startangle=90, explode=(0,0.05), textprops={'fontsize':11})
axes[1].set_title('Proportion')
plt.tight_layout()
##plt.show()

# ═══════════════════════════════════════════════
# 2.2 Fusion des tables pour l'EDA
# Corrections appliquées :
#  - user_id retiré de orders (conflit avec order_items)
#  - products.id renommé product_id
#  - users.id renommé user_id
#  - gender renommé user_gender
#  - dates converties avec utc=True
# ═══════════════════════════════════════════════

# orders sans user_id
orders_eda = orders[[
    c for c in ['order_id','status','created_at','returned_at',
                'shipped_at','delivered_at','num_of_item'] if c in orders.columns
]].copy().rename(columns={
    'created_at':'created_at_order', 'returned_at':'returned_at_order',
    'shipped_at':'shipped_at_order', 'delivered_at':'delivered_at_order',
    'status':'order_status'
})

# products : id → product_id
pk = 'id' if ('id' in products.columns and 'product_id' not in products.columns) else 'product_id'
products_eda = products[[
    c for c in [pk,'cost','category','name','brand','retail_price','department'] if c in products.columns
]].copy().rename(columns={pk:'product_id'})

# users : id → user_id
uk = 'id' if ('id' in users.columns and 'user_id' not in users.columns) else 'user_id'
users_eda = users[[
    c for c in [uk,'age','gender','country','city','state','traffic_source','created_at'] if c in users.columns
]].copy().rename(columns={
    uk:'user_id', 'created_at':'created_at_user',
    'gender':'user_gender', 'traffic_source':'user_traffic_source'
})

# order_items : renommer colonnes ambiguës
oi_eda = order_items[[
    c for c in ['id','order_id','user_id','product_id','inventory_item_id',
                'status','created_at','shipped_at','delivered_at',
                'returned_at','sale_price','returned'] if c in order_items.columns
]].copy().rename(columns={
    'id':'item_id', 'created_at':'created_at_item', 'shipped_at':'shipped_at_item',
    'delivered_at':'delivered_at_item', 'returned_at':'returned_at_item', 'status':'item_status'
})

# Fusion
df_eda = oi_eda.merge(orders_eda,   on='order_id',   how='left')
df_eda = df_eda.merge(products_eda, on='product_id', how='left')
df_eda = df_eda.merge(users_eda,    on='user_id',    how='left')

# Dates
for col in df_eda.columns:
    if '_at' in col and df_eda[col].dtype == 'object':
        df_eda[col] = pd.to_datetime(df_eda[col], errors='coerce', utc=True)

df_eda['order_month']     = df_eda['created_at_order'].dt.month
df_eda['order_dayofweek'] = df_eda['created_at_order'].dt.dayofweek
df_eda['order_hour']      = df_eda['created_at_order'].dt.hour
df_eda['is_weekend']      = (df_eda['order_dayofweek'] >= 5).astype(int)

print(f'✅ Fusion EDA : {df_eda.shape}')



# ═══════════════════════════════════════════════
# 2.3 Taux de retour par Catégorie, Pays, Trafic
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Taux de retour par segment', fontsize=14, fontweight='bold')

if 'category' in df_eda.columns:
    cat_r = df_eda.groupby('category')['returned'].mean().sort_values(ascending=False)
    bars  = axes[0].bar(cat_r.index, cat_r.values, color=sns.color_palette('husl', len(cat_r)))
    axes[0].axhline(return_rate, color='black', linestyle='--', lw=1.5, label=f'Moy: {return_rate:.1%}')
    axes[0].set_title('Par Catégorie')
    axes[0].set_ylabel('Taux de retour')
    axes[0].tick_params(axis='x', rotation=40)
    axes[0].legend(fontsize=9)
    for bar, v in zip(bars, cat_r.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f'{v:.1%}', ha='center', fontsize=8)

if 'country' in df_eda.columns:
    ctr_r = df_eda.groupby('country')['returned'].mean().sort_values(ascending=False).head(8)
    axes[1].bar(ctr_r.index, ctr_r.values, color=sns.color_palette('rocket', len(ctr_r)))
    axes[1].axhline(return_rate, color='black', linestyle='--', lw=1.5)
    axes[1].set_title('Par Pays (Top 8)')
    axes[1].tick_params(axis='x', rotation=40)

traf_col = 'user_traffic_source' if 'user_traffic_source' in df_eda.columns else None
if traf_col:
    trf_r = df_eda.groupby(traf_col)['returned'].mean().sort_values(ascending=False)
    axes[2].bar(trf_r.index, trf_r.values, color=sns.color_palette('mako', len(trf_r)))
    axes[2].axhline(return_rate, color='black', linestyle='--', lw=1.5)
    axes[2].set_title('Par Source de Trafic')
    axes[2].tick_params(axis='x', rotation=40)

plt.tight_layout()
#plt.show()



# ═══════════════════════════════════════════════
# 2.4 Prix vs Retour + Saisonnalité
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle('Prix & Temporalité', fontsize=14, fontweight='bold')

if 'sale_price' in df_eda.columns:
    for val, color, label in [(0,'#27ae60','Non retourné'),(1,'#e74c3c','Retourné')]:
        df_eda[df_eda.returned==val]['sale_price'].hist(
            bins=40, alpha=0.6, ax=axes[0], color=color, label=label, density=True)
    axes[0].set_title('Distribution prix de vente')
    axes[0].set_xlabel('Prix ($)')
    axes[0].legend()
    m0 = df_eda[df_eda.returned==0]['sale_price'].mean()
    m1 = df_eda[df_eda.returned==1]['sale_price'].mean()
    print(f'Prix moyen NON retourné : {m0:.2f} | RETOURNÉ : {m1:.2f}')

monthly = df_eda.groupby('order_month')['returned'].mean()
axes[1].plot(monthly.index, monthly.values, marker='o', color='#e74c3c', lw=2)
axes[1].fill_between(monthly.index, monthly.values, alpha=0.15, color='#e74c3c')
axes[1].set_title('Taux retour par Mois')
axes[1].set_xticks(range(1,13))
axes[1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

day_r = df_eda.groupby('order_dayofweek')['returned'].mean()
axes[2].bar(range(7), day_r.values,
            color=['#e74c3c' if d>=5 else '#3498db' for d in range(7)])
axes[2].set_title('Taux retour par Jour')
axes[2].set_xticks(range(7))
axes[2].set_xticklabels(['Lun','Mar','Mer','Jeu','Ven','Sam','Dim'])

plt.tight_layout()
#plt.show()


# ═══════════════════════════════════════════════
# 3.1 Fusion DÉFINITIVE pour la modélisation
# ═══════════════════════════════════════════════
print('🔗 Fusion définitive...')

# orders : sans user_id
orders_slim = orders[[
    c for c in ['order_id','status','created_at','returned_at',
                'shipped_at','delivered_at','num_of_item'] if c in orders.columns
]].copy().rename(columns={
    'created_at':'created_at_order',   'returned_at':'returned_at_order',
    'shipped_at':'shipped_at_order',   'delivered_at':'delivered_at_order',
    'status':'order_status'
})

# products : id → product_id
pk = 'id' if ('id' in products.columns and 'product_id' not in products.columns) else 'product_id'
products_slim = products[[
    c for c in [pk,'cost','category','name','brand','retail_price','department'] if c in products.columns
]].copy().rename(columns={pk:'product_id'})

# users : id → user_id
uk = 'id' if ('id' in users.columns and 'user_id' not in users.columns) else 'user_id'
users_slim = users[[
    c for c in [uk,'age','gender','country','city','state','traffic_source','created_at'] if c in users.columns
]].copy().rename(columns={
    uk:'user_id', 'created_at':'created_at_user',
    'gender':'user_gender', 'traffic_source':'user_traffic_source'
})

# order_items : table de base avec renommages
oi = order_items[[
    c for c in ['id','order_id','user_id','product_id','inventory_item_id',
                'status','created_at','shipped_at','delivered_at',
                'returned_at','sale_price','returned'] if c in order_items.columns
]].copy().rename(columns={
    'id':'item_id', 'created_at':'created_at_item', 'shipped_at':'shipped_at_item',
    'delivered_at':'delivered_at_item', 'returned_at':'returned_at_item', 'status':'item_status'
})

# inventory : ajouter le coût
inv_k = 'id' if ('id' in inventory.columns and 'inventory_item_id' not in inventory.columns) else 'inventory_item_id'
if 'inventory_item_id' in oi.columns and 'cost' in inventory.columns:
    inv_slim = inventory[[inv_k,'cost']].copy().rename(columns={inv_k:'inventory_item_id'})
    oi = oi.merge(inv_slim, on='inventory_item_id', how='left')
    print('   cost ajouté depuis inventory_items')

# Merges
df = oi.merge(orders_slim,   on='order_id',   how='left')
print(f'  Après + orders   : {df.shape}')
df = df.merge(products_slim, on='product_id', how='left')
print(f'  Après + products : {df.shape}')
df = df.merge(users_slim,    on='user_id',    how='left')
print(f'  Après + users    : {df.shape}')
print(f'\n✅ Fusion terminée : {df.shape[0]:,} lignes × {df.shape[1]} colonnes')



# ═══════════════════════════════════════════════
# 3.2 Conversion des types de données
# Dates stockées en string → datetime64[ns, UTC]
# ═══════════════════════════════════════════════
print('📅 Conversion des dates...')
for col in df.columns:
    if '_at' in col and df[col].dtype == 'object':
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        print(f'  ✅ {col} → datetime64[ns, UTC]')

n_dates = len(df.select_dtypes(include=['datetimetz']).columns)
print(f'\n✅ {n_dates} colonnes en datetime')


# ═══════════════════════════════════════════════
# 3.3 Rapport valeurs manquantes + Nettoyage initial
# ═══════════════════════════════════════════════
before_nan = len(df)
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_show = missing_pct[missing_pct > 0].sort_values(ascending=False)

print(f'Colonnes avec NaN : {len(missing_show)}')

if len(missing_show) > 0:
    fig, ax = plt.subplots(figsize=(11, max(3, len(missing_show)*0.32)))
    bar_colors = ['#e74c3c' if v>30 else '#f39c12' if v>10 else '#27ae60'
                  for v in missing_show.values]
    ax.barh(missing_show.index[::-1], missing_show.values[::-1], color=bar_colors[::-1])
    ax.set_xlabel('% valeurs manquantes')
    ax.set_title('Valeurs manquantes par colonne  (Rouge >30% | Orange >10% | Vert <10%)',
                 fontweight='bold')
    ax.axvline(30, color='red',    linestyle='--', alpha=0.6, label='>30%')
    ax.axvline(10, color='orange', linestyle='--', alpha=0.6, label='>10%')
    ax.legend()
    plt.tight_layout()
    #plt.show()

# Supprimer colonnes >50% NaN (sauf les colonnes de retour qui sont normales)
KEEP_EVEN_IF_NAN = ['returned_at_item', 'returned_at_order']
cols_drop = [c for c in missing_pct[missing_pct>50].index if c not in KEEP_EVEN_IF_NAN]
if cols_drop:
    df.drop(columns=cols_drop, inplace=True)
    print(f'🗑️  Colonnes supprimées (>50% NaN) : {cols_drop}')

# Supprimer lignes avec NaN sur colonnes critiques
critical = [c for c in ['sale_price','category','order_id','user_id','item_id'] if c in df.columns]
df.dropna(subset=critical, inplace=True)
df.drop_duplicates(subset=['item_id'], inplace=True)

print(f'Lignes supprimées : {before_nan - len(df):,}')
print(f'✅ Après nettoyage NaN : {df.shape}')




# ═══════════════════════════════════════════════
# 3.4 Suppression des outliers de prix (1%-99%)
# ═══════════════════════════════════════════════
before_out = len(df)

if 'sale_price' in df.columns:
    q01 = df['sale_price'].quantile(0.01)
    q99 = df['sale_price'].quantile(0.99)

    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    fig.suptitle('Nettoyage outliers prix', fontweight='bold')
    df['sale_price'].hist(bins=60, ax=axes[0], color='#e74c3c', alpha=0.8)
    axes[0].set_title(f'AVANT  (min={df.sale_price.min():.0f}, max={df.sale_price.max():.0f})')

    df = df[(df['sale_price'] >= q01) & (df['sale_price'] <= q99)].copy()

    df['sale_price'].hist(bins=60, ax=axes[1], color='#27ae60', alpha=0.8)
    axes[1].set_title(f'APRÈS  [{q01:.2f} – {q99:.2f}]')
    plt.tight_layout()
    #plt.show()

    print(f'Lignes supprimées (outliers) : {before_out - len(df):,}')

print(f'\n✅ Étape 3 terminée — Dataset propre : {df.shape[0]:,} lignes × {df.shape[1]} colonnes')




# ═══════════════════════════════════════════════
# 3.5.0 — Diagnostic AVANT imputation
# ═══════════════════════════════════════════════
nan_avant = df.isnull().sum()
nan_avant = nan_avant[nan_avant > 0].sort_values(ascending=False)

print('━'*65)
print('  DIAGNOSTIC VALEURS MANQUANTES — AVANT IMPUTATION')
print('━'*65)
print(f'  {"Colonne":<35} {"NaN":>8}  {"NaN%":>7}  Type')
print('  ' + '─'*60)
for col, n in nan_avant.items():
    pct  = n / len(df) * 100
    typ  = str(df[col].dtype)
    icon = '🔴' if pct > 30 else '🟠' if pct > 10 else '🟡'
    print(f'  {icon} {col:<33} {n:>8,}  {pct:>6.1f}%  {typ}')

print(f'\n  Total NaN : {df.isnull().sum().sum():,}')
print(f'  Total lignes : {len(df):,}')



# ═══════════════════════════════════════════════
# 3.5.1 — Imputation colonnes NUMÉRIQUES
# Stratégie :
#   - age           → médiane par country (ou médiane globale)
#   - cost          → médiane par category (ou médiane globale)
#   - retail_price  → médiane par category (ou médiane globale)
#   - num_of_item   → médiane globale (entier)
#   - autres num    → médiane globale
# ═══════════════════════════════════════════════
print('🔢 Imputation numériques...')

# --- age : médiane par country ---
if 'age' in df.columns and df['age'].isnull().any():
    n_before = df['age'].isnull().sum()
    if 'country' in df.columns:
        df['age'] = df.groupby('country')['age'].transform(
            lambda x: x.fillna(x.median())
        )
    df['age'].fillna(df['age'].median(), inplace=True)  # résidu si group vide
    df['age'] = df['age'].round().astype('Int64')        # entier nullable
    print(f'  ✅ age          : {n_before:,} NaN → médiane par pays')

# --- cost : médiane par category ---
if 'cost' in df.columns and df['cost'].isnull().any():
    n_before = df['cost'].isnull().sum()
    if 'category' in df.columns:
        df['cost'] = df.groupby('category')['cost'].transform(
            lambda x: x.fillna(x.median())
        )
    df['cost'].fillna(df['cost'].median(), inplace=True)
    print(f'  ✅ cost         : {n_before:,} NaN → médiane par catégorie')

# --- retail_price : médiane par category ---
if 'retail_price' in df.columns and df['retail_price'].isnull().any():
    n_before = df['retail_price'].isnull().sum()
    if 'category' in df.columns:
        df['retail_price'] = df.groupby('category')['retail_price'].transform(
            lambda x: x.fillna(x.median())
        )
    df['retail_price'].fillna(df['retail_price'].median(), inplace=True)
    print(f'  ✅ retail_price : {n_before:,} NaN → médiane par catégorie')

# --- num_of_item : médiane globale (entier) ---
if 'num_of_item' in df.columns and df['num_of_item'].isnull().any():
    n_before = df['num_of_item'].isnull().sum()
    med_val  = int(df['num_of_item'].median())
    df['num_of_item'].fillna(med_val, inplace=True)
    print(f'  ✅ num_of_item  : {n_before:,} NaN → médiane globale ({med_val})')

# --- toutes autres colonnes numériques restantes ---
other_num = df.select_dtypes(include=['float64','int64']).columns
for col in other_num:
    if df[col].isnull().any() and col not in ['returned']:
        n_b = df[col].isnull().sum()
        df[col].fillna(df[col].median(), inplace=True)
        print(f'  ✅ {col:<20} : {n_b:,} NaN → médiane globale')

print('\n✅ Imputation numériques terminée')




# ═══════════════════════════════════════════════
# 3.5.2 — Imputation colonnes CATÉGORIELLES
# Stratégie :
#   - brand      → 'Unknown' (marque non connue)
#   - department → mode par category
#   - user_gender → mode global
#   - city, state → 'Unknown'
#   - item_status / order_status → mode
#   - user_traffic_source → 'Unknown'
#   - autres object → mode ou 'Unknown'
# ═══════════════════════════════════════════════
print('🔤 Imputation catégorielles...')

# --- brand ---
if 'brand' in df.columns and df['brand'].isnull().any():
    n_b = df['brand'].isnull().sum()
    df['brand'].fillna('Unknown', inplace=True)
    print(f'  ✅ brand               : {n_b:,} NaN → "Unknown"')

# --- department : mode par category ---
if 'department' in df.columns and df['department'].isnull().any():
    n_b = df['department'].isnull().sum()
    if 'category' in df.columns:
        def fill_dept(grp):
            mode_val = grp.mode()
            return grp.fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')
        df['department'] = df.groupby('category')['department'].transform(fill_dept)
    df['department'].fillna(df['department'].mode().iloc[0] if df['department'].notna().any() else 'Unknown', inplace=True)
    print(f'  ✅ department          : {n_b:,} NaN → mode par catégorie')

# --- user_gender ---
if 'user_gender' in df.columns and df['user_gender'].isnull().any():
    n_b = df['user_gender'].isnull().sum()
    mode_g = df['user_gender'].mode().iloc[0] if df['user_gender'].notna().any() else 'Unknown'
    df['user_gender'].fillna(mode_g, inplace=True)
    print(f'  ✅ user_gender         : {n_b:,} NaN → mode global ("{mode_g}")')

# --- city ---
if 'city' in df.columns and df['city'].isnull().any():
    n_b = df['city'].isnull().sum()
    df['city'].fillna('Unknown', inplace=True)
    print(f'  ✅ city                : {n_b:,} NaN → "Unknown"')

# --- state ---
if 'state' in df.columns and df['state'].isnull().any():
    n_b = df['state'].isnull().sum()
    df['state'].fillna('Unknown', inplace=True)
    print(f'  ✅ state               : {n_b:,} NaN → "Unknown"')

# --- user_traffic_source ---
if 'user_traffic_source' in df.columns and df['user_traffic_source'].isnull().any():
    n_b = df['user_traffic_source'].isnull().sum()
    df['user_traffic_source'].fillna('Unknown', inplace=True)
    print(f'  ✅ user_traffic_source : {n_b:,} NaN → "Unknown"')

# --- item_status ---
if 'item_status' in df.columns and df['item_status'].isnull().any():
    n_b = df['item_status'].isnull().sum()
    mode_s = df['item_status'].mode().iloc[0] if df['item_status'].notna().any() else 'Unknown'
    df['item_status'].fillna(mode_s, inplace=True)
    print(f'  ✅ item_status         : {n_b:,} NaN → mode ("{mode_s}")')

# --- order_status ---
if 'order_status' in df.columns and df['order_status'].isnull().any():
    n_b = df['order_status'].isnull().sum()
    mode_s = df['order_status'].mode().iloc[0] if df['order_status'].notna().any() else 'Unknown'
    df['order_status'].fillna(mode_s, inplace=True)
    print(f'  ✅ order_status        : {n_b:,} NaN → mode ("{mode_s}")')

# --- toutes autres colonnes object restantes ---
other_obj = [c for c in df.select_dtypes(include='object').columns if df[c].isnull().any()]
for col in other_obj:
    n_b = df[col].isnull().sum()
    mode_v = df[col].mode()
    fill_v = mode_v.iloc[0] if len(mode_v) > 0 else 'Unknown'
    df[col].fillna(fill_v, inplace=True)
    print(f'  ✅ {col:<25} : {n_b:,} NaN → mode/Unknown')

print('\n✅ Imputation catégorielles terminée')


# ═══════════════════════════════════════════════
# 3.5.2 — Imputation colonnes CATÉGORIELLES
# Stratégie :
#   - brand      → 'Unknown' (marque non connue)
#   - department → mode par category
#   - user_gender → mode global
#   - city, state → 'Unknown'
#   - item_status / order_status → mode
#   - user_traffic_source → 'Unknown'
#   - autres object → mode ou 'Unknown'
# ═══════════════════════════════════════════════
print('🔤 Imputation catégorielles...')

# --- brand ---
if 'brand' in df.columns and df['brand'].isnull().any():
    n_b = df['brand'].isnull().sum()
    df['brand'].fillna('Unknown', inplace=True)
    print(f'  ✅ brand               : {n_b:,} NaN → "Unknown"')

# --- department : mode par category ---
if 'department' in df.columns and df['department'].isnull().any():
    n_b = df['department'].isnull().sum()
    if 'category' in df.columns:
        def fill_dept(grp):
            mode_val = grp.mode()
            return grp.fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')
        df['department'] = df.groupby('category')['department'].transform(fill_dept)
    df['department'].fillna(df['department'].mode().iloc[0] if df['department'].notna().any() else 'Unknown', inplace=True)
    print(f'  ✅ department          : {n_b:,} NaN → mode par catégorie')

# --- user_gender ---
if 'user_gender' in df.columns and df['user_gender'].isnull().any():
    n_b = df['user_gender'].isnull().sum()
    mode_g = df['user_gender'].mode().iloc[0] if df['user_gender'].notna().any() else 'Unknown'
    df['user_gender'].fillna(mode_g, inplace=True)
    print(f'  ✅ user_gender         : {n_b:,} NaN → mode global ("{mode_g}")')

# --- city ---
if 'city' in df.columns and df['city'].isnull().any():
    n_b = df['city'].isnull().sum()
    df['city'].fillna('Unknown', inplace=True)
    print(f'  ✅ city                : {n_b:,} NaN → "Unknown"')

# --- state ---
if 'state' in df.columns and df['state'].isnull().any():
    n_b = df['state'].isnull().sum()
    df['state'].fillna('Unknown', inplace=True)
    print(f'  ✅ state               : {n_b:,} NaN → "Unknown"')

# --- user_traffic_source ---
if 'user_traffic_source' in df.columns and df['user_traffic_source'].isnull().any():
    n_b = df['user_traffic_source'].isnull().sum()
    df['user_traffic_source'].fillna('Unknown', inplace=True)
    print(f'  ✅ user_traffic_source : {n_b:,} NaN → "Unknown"')

# --- item_status ---
if 'item_status' in df.columns and df['item_status'].isnull().any():
    n_b = df['item_status'].isnull().sum()
    mode_s = df['item_status'].mode().iloc[0] if df['item_status'].notna().any() else 'Unknown'
    df['item_status'].fillna(mode_s, inplace=True)
    print(f'  ✅ item_status         : {n_b:,} NaN → mode ("{mode_s}")')

# --- order_status ---
if 'order_status' in df.columns and df['order_status'].isnull().any():
    n_b = df['order_status'].isnull().sum()
    mode_s = df['order_status'].mode().iloc[0] if df['order_status'].notna().any() else 'Unknown'
    df['order_status'].fillna(mode_s, inplace=True)
    print(f'  ✅ order_status        : {n_b:,} NaN → mode ("{mode_s}")')

# --- toutes autres colonnes object restantes ---
other_obj = [c for c in df.select_dtypes(include='object').columns if df[c].isnull().any()]
for col in other_obj:
    n_b = df[col].isnull().sum()
    mode_v = df[col].mode()
    fill_v = mode_v.iloc[0] if len(mode_v) > 0 else 'Unknown'
    df[col].fillna(fill_v, inplace=True)
    print(f'  ✅ {col:<25} : {n_b:,} NaN → mode/Unknown')

print('\n✅ Imputation catégorielles terminée')



# ═══════════════════════════════════════════════
# 3.5.3 — Imputation colonnes DATES
# Stratégie :
#   - returned_at_item / returned_at_order → NaN normal (article non retourné)
#     → On NE les impute PAS (NaN = information métier valide)
#   - shipped_at_item / shipped_at_order   → médiane de délai depuis created_at
#   - delivered_at_item / delivered_at_order → médiane de délai depuis shipped_at
#   - created_at_user → médiane globale
# ═══════════════════════════════════════════════
print('📅 Imputation dates...')

DATE_SKIP = ['returned_at_item', 'returned_at_order']

date_cols_all = df.select_dtypes(include=['datetimetz','datetime64']).columns.tolist()

for col in date_cols_all:
    if col in DATE_SKIP:
        print(f'  ⏭️  {col:<30} conservé tel quel (NaN = non retourné)')
        continue
    if df[col].isnull().any():
        n_b     = df[col].isnull().sum()
        median_t = df[col].dropna().sort_values().iloc[len(df[col].dropna())//2]
        df[col]  = df[col].fillna(median_t)
        print(f'  ✅ {col:<30} : {n_b:,} NaN → médiane temporelle')

print('\n✅ Imputation dates terminée')



# ═══════════════════════════════════════════════
# 3.5.4 — Rapport FINAL des valeurs manquantes
# ═══════════════════════════════════════════════
nan_apres = df.isnull().sum()
nan_apres_pos = nan_apres[nan_apres > 0]

print('━'*65)
print('  RAPPORT FINAL — VALEURS MANQUANTES APRÈS IMPUTATION')
print('━'*65)

if len(nan_apres_pos) == 0:
    print('\n  ✅ AUCUNE valeur manquante sur les colonnes non-date !')
else:
    print(f'  {"Colonne":<35} {"NaN":>8}  {"NaN%":>7}  Remarque')
    print('  ' + '─'*65)
    for col, n in nan_apres_pos.items():
        pct  = n / len(df) * 100
        rem  = '(NaN = non retourné ✅)' if col in DATE_SKIP else '⚠️  à vérifier'
        print(f'  {col:<35} {n:>8,}  {pct:>6.1f}%  {rem}')

# Bilan visuel avant/après
print(f'\n  Total NaN AVANT : {nan_avant.sum():,}')
print(f'  Total NaN APRÈS : {nan_apres.sum():,}')
pct_resolu = (1 - nan_apres.sum() / max(nan_avant.sum(), 1)) * 100
print(f'  NaN résolus     : {pct_resolu:.1f}%')

# Graphique comparatif
cols_compare = [c for c in nan_avant.index if c not in DATE_SKIP]
if cols_compare:
    avant_vals = [nan_avant.get(c, 0) / len(df) * 100 for c in cols_compare]
    apres_vals = [nan_apres.get(c, 0) / len(df) * 100 for c in cols_compare]

    x = np.arange(len(cols_compare))
    fig, ax = plt.subplots(figsize=(max(8, len(cols_compare)*1.1), 4))
    ax.bar(x - 0.2, avant_vals, width=0.38, label='Avant', color='#e74c3c', alpha=0.8)
    ax.bar(x + 0.2, apres_vals, width=0.38, label='Après', color='#27ae60', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cols_compare, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('% NaN')
    ax.set_title('Valeurs manquantes — Avant vs Après imputation', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    #plt.show()
    
    
    
    
    
    # ═══════════════════════════════════════════════
# VÉRIF 1 — Shape & Variable cible
# ═══════════════════════════════════════════════
print('━'*55)
print('  VÉRIF 1 — Shape & Variable cible')
print('━'*55)
print(f'  Lignes   : {df.shape[0]:,}')
print(f'  Colonnes : {df.shape[1]}')

if 'returned' in df.columns:
    vc  = df['returned'].value_counts()
    nan = df['returned'].isnull().sum()
    print(f'\n  returned = 0 (non retourné) : {vc.get(0,0):>8,}  ({vc.get(0,0)/len(df):.2%})')
    print(f'  returned = 1 (retourné)     : {vc.get(1,0):>8,}  ({vc.get(1,0)/len(df):.2%})')
    print(f'  Taux de retour              : {df.returned.mean():.4f}')
    print(f'  NaN dans returned           : {nan}  {"✅" if nan==0 else "❌"}')
else:
    print('  ❌ ERREUR : colonne returned absente !')


# ═══════════════════════════════════════════════
# VÉRIF 2 — Inventaire complet des colonnes
# ═══════════════════════════════════════════════
print('━'*75)
print('  VÉRIF 2 — Inventaire des colonnes')
print('━'*75)
print(f'  {"Colonne":<35} {"Type":<25} {"NaN":>8}  {"NaN%":>6}  Statut')
print('  ' + '─'*70)

DATE_SKIP = ['returned_at_item', 'returned_at_order']

for col in df.columns:
    dtype   = str(df[col].dtype)
    n_nan   = df[col].isnull().sum()
    pct_nan = n_nan / len(df) * 100
    if n_nan == 0:
        st = '✅ propre'
    elif col in DATE_SKIP:
        st = '✅ NaN métier (non retourné)'
    elif pct_nan > 50:
        st = '⚠️  > 50%'
    elif pct_nan > 10:
        st = '🟡 > 10%'
    else:
        st = '🟢 < 10%'
    print(f'  {col:<35} {dtype:<25} {n_nan:>8,}  {pct_nan:>5.1f}%  {st}')



# ═══════════════════════════════════════════════
# VÉRIF 3 — Types de données
# ═══════════════════════════════════════════════
print('━'*55)
print('  VÉRIF 3 — Types de données')
print('━'*55)

num_cols  = df.select_dtypes(include=['int64','float64','Int64']).columns.tolist()
cat_cols  = df.select_dtypes(include=['object','category']).columns.tolist()
date_cols = df.select_dtypes(include=['datetime64','datetimetz']).columns.tolist()

print(f'  Numériques  ({len(num_cols):>2}) : {num_cols}')
print(f'\n  Catégoriels ({len(cat_cols):>2}) : {cat_cols}')
print(f'\n  Dates       ({len(date_cols):>2}) : {date_cols}')

print('\n  Colonnes clés attendues :')
checks = [
    ('item_id',          'int64',    'Clé primaire order_items'),
    ('sale_price',       'float64',  'Prix de vente'),
    ('returned',         'int64',    'Variable cible 0/1'),
    ('category',         'object',   'Catégorie produit (jointure products)'),
    ('user_gender',      'object',   'Genre client (jointure users)'),
    ('order_status',     'object',   'Statut commande (jointure orders)'),
    ('created_at_order', 'datetime', 'Date commande convertie'),
]
for col, exp_type, desc in checks:
    if col in df.columns:
        actual = str(df[col].dtype)
        ok     = exp_type in actual
        print(f'    {"✅" if ok else "⚠️ "} {col:<25} {actual:<25} ({desc})')
    else:
        print(f'    ❌ {col:<25} ABSENT                    ({desc})')
    


# ═══════════════════════════════════════════════
# VÉRIF 4 — Statistiques descriptives
# ═══════════════════════════════════════════════
print('━'*55)
print('  VÉRIF 4 — Statistiques descriptives')
print('━'*55)

stat_cols = [c for c in ['sale_price','retail_price','cost','num_of_item','age'] if c in df.columns]
print(df[stat_cols].describe().round(2))

if 'category' in df.columns:
    print('\n  Valeurs uniques par colonne catégorielle :')
    for col in [c for c in ['category','brand','department','user_gender','country',
                             'user_traffic_source','order_status','item_status'] if c in df.columns]:
        n   = df[col].nunique()
        top = df[col].value_counts().index[0]
        pct_nan = df[col].isnull().mean() * 100
        icon = '✅' if pct_nan == 0 else '⚠️'
        print(f'    {icon} {col:<25} : {n:>4} valeurs uniques  (top: {top})  NaN: {pct_nan:.1f}%')


# ═══════════════════════════════════════════════
# VÉRIF 5 — Doublons & Intégrité des jointures
# ═══════════════════════════════════════════════
print('━'*55)
print('  VÉRIF 5 — Doublons & Intégrité')
print('━'*55)

dupes = df.duplicated(subset=['item_id']).sum()
print(f'  Doublons sur item_id  : {dupes}  {"✅ OK" if dupes==0 else "❌"}')

print('\n  Qualité des jointures :')
join_checks = [
    ('category',         'products'),
    ('user_gender',      'users'),
    ('order_status',     'orders'),
    ('created_at_order', 'orders (dates)'),
]
for col, source in join_checks:
    if col in df.columns:
        pct = df[col].isnull().mean()*100
        icon = '✅' if pct < 5 else '⚠️ '
        print(f'    {icon} {col:<25} ← {source:<20} NaN: {pct:.1f}%')
    else:
        print(f'    ❌ {col:<25} ← {source:<20} ABSENT')

print('\n  Plage de prix (après nettoyage outliers) :')
if 'sale_price' in df.columns:
    print(f'    Min  : {df.sale_price.min():.2f}')
    print(f'    Max  : {df.sale_price.max():.2f}')
    print(f'    Mean : {df.sale_price.mean():.2f}')
    print(f'    Std  : {df.sale_price.std():.2f}')


# ═══════════════════════════════════════════════
# VÉRIF 6 — Dashboard visuel du dataset final
# ═══════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    f'Dashboard — Dataset final  ({df.shape[0]:,} lignes × {df.shape[1]} colonnes)',
    fontsize=14, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Variable cible
ax1 = fig.add_subplot(gs[0,0])
vc = df['returned'].value_counts()
ax1.pie(vc.values,
        labels=[f'Non retourné\n{vc.get(0,0):,}', f'Retourné\n{vc.get(1,0):,}'],
        colors=['#27ae60','#e74c3c'], autopct='%1.1f%%',
        startangle=90, explode=(0,0.05))
ax1.set_title('Variable cible : returned', fontweight='bold')

# 2. Distribution prix
ax2 = fig.add_subplot(gs[0,1])
if 'sale_price' in df.columns:
    df['sale_price'].hist(bins=50, ax=ax2, color='#3498db', alpha=0.8)
    ax2.axvline(df.sale_price.mean(), color='red', linestyle='--', label=f'Moy: {df.sale_price.mean():.1f}')
    ax2.set_title('Distribution sale_price', fontweight='bold')
    ax2.set_xlabel('Prix ($)')
    ax2.legend()

# 3. NaN restants
ax3 = fig.add_subplot(gs[0,2])
nan_left = (df.isnull().sum()/len(df)*100)
nan_left_non_date = nan_left.drop(labels=[c for c in DATE_SKIP if c in nan_left.index], errors='ignore')
nan_left_show = nan_left_non_date[nan_left_non_date > 0].sort_values(ascending=False).head(10)
if len(nan_left_show) > 0:
    c_bar = ['#e74c3c' if v>30 else '#f39c12' if v>10 else '#27ae60' for v in nan_left_show.values]
    ax3.barh(nan_left_show.index[::-1], nan_left_show.values[::-1], color=c_bar[::-1])
    ax3.set_title('NaN restants (hors dates retour)', fontweight='bold')
    ax3.set_xlabel('% NaN')
else:
    ax3.text(0.5, 0.5, '✅ 0 NaN\nsur colonnes\ncritiques',
             ha='center', va='center', fontsize=16, color='#27ae60',
             fontweight='bold', transform=ax3.transAxes)
    ax3.axis('off')
    ax3.set_title('NaN restants', fontweight='bold')

# 4. Taux retour par catégorie
ax4 = fig.add_subplot(gs[1,0])
if 'category' in df.columns:
    cat_r = df.groupby('category')['returned'].mean().sort_values()
    ax4.barh(cat_r.index, cat_r.values, color=sns.color_palette('husl', len(cat_r)))
    ax4.axvline(df.returned.mean(), color='black', linestyle='--', lw=1.5)
    ax4.set_title('Taux retour par Catégorie', fontweight='bold')
    ax4.set_xlabel('Taux retour')

# 5. Taux retour par pays
ax5 = fig.add_subplot(gs[1,1])
if 'country' in df.columns:
    ctr = df.groupby('country')['returned'].mean().sort_values().tail(10)
    ax5.barh(ctr.index, ctr.values, color=sns.color_palette('rocket', len(ctr)))
    ax5.axvline(df.returned.mean(), color='black', linestyle='--', lw=1.5)
    ax5.set_title('Taux retour par Pays (Top 10)', fontweight='bold')
    ax5.set_xlabel('Taux retour')

# 6. Types colonnes
ax6 = fig.add_subplot(gs[1,2])
tc = df.dtypes.astype(str).value_counts()
ax6.pie(tc.values, labels=[f'{t}\n({v} col.)' for t,v in tc.items()],
        autopct='%1.0f%%', startangle=90,
        colors=sns.color_palette('Set2', len(tc)))
ax6.set_title('Types de colonnes', fontweight='bold')

plt.tight_layout()
#plt.show()


# ═══════════════════════════════════════════════
# VÉRIF 7 — Aperçu des premières lignes
# ═══════════════════════════════════════════════
print('━'*55)
print('  VÉRIF 7 — Aperçu du dataset final (5 lignes)')
print('━'*55)
print(df.head(5))

# ═══════════════════════════════════════════════
# ✅ BILAN FINAL
# ═══════════════════════════════════════════════
DATE_SKIP = ['returned_at_item', 'returned_at_order']

# NaN hors colonnes dates-retour (normales)
nan_critique = df.drop(columns=[c for c in DATE_SKIP if c in df.columns]).isnull().sum().sum()
total_dupes  = df.duplicated(subset=['item_id']).sum()
n_num        = len(df.select_dtypes(include=['int64','float64','Int64']).columns)
n_cat        = len(df.select_dtypes(include=['object','category']).columns)
n_date       = len(df.select_dtypes(include=['datetime64','datetimetz']).columns)
has_cat      = 'category'         in df.columns
has_gender   = 'user_gender'      in df.columns
has_date     = 'created_at_order' in df.columns
has_returned = 'returned'         in df.columns and df['returned'].isnull().sum() == 0

print()
print('╔' + '═'*60 + '╗')
print('║         ✅  BILAN — ÉTAPES 1, 2, 3 + IMPUTATION NaN         ║')
print('╠' + '═'*60 + '╣')
print(f'║  Lignes dans df                : {df.shape[0]:>10,}              ║')
print(f'║  Colonnes dans df              : {df.shape[1]:>10}              ║')
print(f'║    ├── Numériques              : {n_num:>10}              ║')
print(f'║    ├── Catégorielles           : {n_cat:>10}              ║')
print(f'║    └── Dates (datetime64 UTC)  : {n_date:>10}              ║')
print('╠' + '═'*60 + '╣')
print(f'║  Taux de retour global         : {df.returned.mean():>10.2%}              ║')
print(f'║  Articles non retournés (0)    : {(df.returned==0).sum():>10,}              ║')
print(f'║  Articles retournés     (1)    : {(df.returned==1).sum():>10,}              ║')
print('╠' + '═'*60 + '╣')
print(f'║  NaN critiques restants        : {nan_critique:>10,}              ║')
print(f'║  Doublons (item_id)            : {total_dupes:>10}              ║')
print('╠' + '═'*60 + '╣')
checks_final = [
    (has_returned,        'Variable cible returned propre (0 NaN)'),
    (total_dupes == 0,    'Aucun doublon sur item_id'),
    (nan_critique == 0,   'NaN critiques = 0 (hors dates retour)'),
    (has_cat,             'Jointure products OK (category présente)'),
    (has_gender,          'Jointure users OK (user_gender présente)'),
    (has_date,            'Dates converties en datetime64 UTC'),
    (n_date >= 2,         'Au moins 2 colonnes dates converties'),
]
all_ok = all(ok for ok, _ in checks_final)
for ok, msg in checks_final:
    print(f'║    {"✅" if ok else "❌"}  {msg:<52}║')
print('╠' + '═'*60 + '╣')
if all_ok:
    print('║   🚀  PRÊT POUR L\'ÉTAPE 4 — FEATURE ENGINEERING            ║')
else:
    print('║   ⚠️   REVOIR LES POINTS MARQUÉS ❌ CI-DESSUS                ║')
print('╚' + '═'*60 + '╝')


# ═══════════════════════════════════════════════
# 💾 SAUVEGARDE sur Google Drive
# ═══════════════════════════════════════════════
CLEANED_PATH = DATA_PATH + 'df_cleaned.csv'
df.to_csv(CLEANED_PATH, index=False)
size_mb = os.path.getsize(CLEANED_PATH) / 1024 / 1024

print(f'✅ Sauvegardé : {CLEANED_PATH}')
print(f'   Taille     : {size_mb:.2f} MB')
print(f'   Shape      : {df.shape}')
print()
print('💡 Pour recharger sans tout réexécuter :')
print('   df = pd.read_csv(DATA_PATH + \'df_cleaned.csv\', parse_dates=True)')