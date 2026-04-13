"""
===================================================================================
视网膜裂孔表型聚类 + 敏感性分析 + 生存分析  
===================================================================================

研究目标:
  1. 基于裂孔特征进行无监督聚类，发现临床表型
  2. Bootstrap 稳定性评估 + 内/外部队列验证
  3. 敏感性分析：用聚类标签替代原始裂孔变量，对比模型性能
  4. NRI / IDI 定量评估模型改进幅度
  5. Kaplan-Meier 生存曲线 + Log-rank 检验
===================================================================================
"""

import os
os.environ['SCIPY_ARRAY_API'] = '1'
import warnings
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             calinski_harabasz_score, roc_auc_score,
                             brier_score_loss, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency, mannwhitneyu, norm

# ---- K-Prototypes (混合型聚类) ----
try:
    from kmodes.kprototypes import KPrototypes
    KPROTOTYPES_AVAILABLE = True
except ImportError:
    KPROTOTYPES_AVAILABLE = False
    print("⚠️ 未安装 kmodes 库，请安装: pip install kmodes")

warnings.filterwarnings('ignore')

# ---- 生存分析 ----
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("⚠️ 未安装 lifelines 库，将跳过生存曲线分析")
    print("   安装方法: pip install lifelines")

# ---- 复用模型训练模块----
import importlib.util
spec = importlib.util.spec_from_file_location("model_training", "2_模型训练.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

MetricsCalculator = mt.MetricsCalculator
MultiModelTrainer = mt.MultiModelTrainer
make_column_transformer = mt.make_column_transformer


# ============================================================
# 0. 全局配置
# ============================================================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

FIG_DIR = "Figures_Clustering"
TABLE_DIR = "Tables_Clustering"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class Config:
    """集中管理全部参数"""

    # ---- 数据路径 ----
    TRAIN_DATA = 'Internal_setA_NaN.csv'
    EXTERNAL_DATA = 'External_setA_NaN.csv'
    CLINICAL_DATA = None            
    TARGET_COL = 'Recurrence'
    INDEX_COL = 'Patient_Name'

    # ---- 聚类参数 ----
    K_RANGE = (2, 8)                # 搜索范围
    K_FINAL = 3                     # 最终聚类数 
    AUTO_SELECT_K = False
    N_BOOTSTRAP = 100               # Bootstrap 重采样次数

    # ---- 裂孔特征关键词  ----
    TEAR_KEYWORDS = [
        'Break_Loc_Inferior',
        'Macular_Hole',
        'number_of_breaks',
        'Largest_Break_Diameter',
        'Atrophic_Holes',
        'Lattice_Degeneration',
    ]

    # ---- 裂孔特征: 按变量类型分组 ----
    TEAR_CATEGORICAL = [
        'Break_Loc_Inferior',
        'Macular_Hole',
        'Atrophic_Holes',
        'Lattice_Degeneration',
    ]
    TEAR_CONTINUOUS = [
        'number_of_breaks',
        'Largest_Break_Diameter',
    ]

    # ---- K-Prototypes 参数 ----
    GAMMA = None                    # 分类权重

    # ---- 敏感性分析: 被替换的裂孔原始特征 ----
    BREAK_NUMERIC_FEATURES = ['number_of_breaks', 'Largest_Break_Diameter']
    BREAK_CATEGORICAL_FEATURES = ['Break_Loc_Inferior']
    BREAK_CLUSTER_COL = 'Break_Cluster'

    # ---- 敏感性分析: 非裂孔术前特征 ----
    LOG_FEATURES = ['Symptom_Duration']
    NUM_FEATURES = ['AL', 'RD_Extent']
    CAT_FEATURES = ['PVR_Grade_Pre', 'Choroidal_Detachment',
                    'Lens_Status_Pre', 'Macular_status']

    # ---- 原术前模型 pkl (用于 NRI 比较) ----
    PREOP_MODEL_PKL = 'model_results/Model_Package_20260216_1724.pkl'

    # ---- 交叉验证 ----
    CV_FOLDS = 5
    CV_REPEATS = 20

    # ---- 输出 ----
    OUTPUT_DIR = 'model_results'
    RANDOM_STATE = 42


# ============================================================
# SCI 规范变量标签映射
# ============================================================
FEATURE_DISPLAY_LABELS = {
    # 裂孔特征
    'Break_Loc_Inferior':    'Inferior break',
    'Macular_Hole':          'Macular hole',
    'number_of_breaks':      'Number of retinal breaks',
    'Largest_Break_Diameter': 'Largest break diameter (DD)',
    'Atrophic_Holes':        'Atrophic holes',
    'Lattice_Degeneration':  'Lattice degeneration',
    # 其他术前特征
    'Symptom_Duration':      'Symptom duration (days)',
    'AL':                    'Axial length (mm)',
    'RD_Extent':             'RD extent (quadrants)',
    'PVR_Grade_Pre':         'Preoperative PVR grade',
    'Choroidal_Detachment':  'Choroidal detachment',
    'Lens_Status_Pre':       'Preoperative lens status',
    'Macular_status':        'Macular status (on/off)',
    'Break_Cluster':         'Retinal break phenotype cluster',
}


def get_display_label(varname):
    """返回 SCI 规范展示标签；无映射则返回原名"""
    return FEATURE_DISPLAY_LABELS.get(varname, varname)


# ============================================================
# 工具函数
# ============================================================
def _ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


def save_publication_fig(fig_name, dpi=600):
    """保存高清发表级图片"""
    save_path = os.path.join(FIG_DIR, fig_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"📷 图片已保存: {save_path}")


# ============================================================
# 1. 裂孔特征筛选
# ============================================================
def select_tear_features(df, keywords=None, verbose=True):
    """
    基于关键词从数据集列名中筛选裂孔相关特征。

    Returns: list[str]
    """
    keywords = keywords or Config.TEAR_KEYWORDS
    tear_features = [
        col for col in df.columns
        if any(kw in col for kw in keywords)
    ]
    if verbose:
        print("=" * 70)
        print("🔍 裂孔特征筛选结果")
        print("=" * 70)
        print(f"  数据集总特征数: {len(df.columns)}")
        print(f"  裂孔聚类特征数: {len(tear_features)}")
        for i, feat in enumerate(tear_features, 1):
            print(f"    {i:2d}. {feat}")
        print("=" * 70 + "\n")
    return tear_features


# ============================================================
# 2. 数据预处理 (分类/连续分别处理)
# ============================================================
def preprocess_features(df_raw, tear_features, cat_features=None, cont_features=None):

    cat_features = cat_features or Config.TEAR_CATEGORICAL
    cont_features = cont_features or Config.TEAR_CONTINUOUS

    # 过滤: 只保留数据集中实际存在的列
    cat_features = [f for f in cat_features if f in tear_features and f in df_raw.columns]
    cont_features = [f for f in cont_features if f in tear_features and f in df_raw.columns]

    # 合并顺序: 连续在前, 分类在后
    ordered_features = cont_features + cat_features
    X_raw = df_raw[ordered_features].copy()
    original_index = X_raw.index          # 保存原始索引
    X_raw = X_raw.reset_index(drop=True)  # 用整数索引避免重复 Patient_Name 问题

    # 日志: 缺失值
    nan_counts = X_raw.isnull().sum()
    filled = nan_counts[nan_counts > 0]
    if len(filled) > 0:
        print("🔧 缺失值填补:")
        for feat, cnt in filled.items():
            strategy = "众数" if feat in cat_features else "中位数"
            print(f"   {feat}: {strategy}填补 {cnt} 个缺失值")
    else:
        print("   ✅ 裂孔特征无缺失值")

    # ---------- 连续变量: 中位数填补 + Z-score ----------
    imputer_cont = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    if cont_features:
        X_cont_imp = pd.DataFrame(
            imputer_cont.fit_transform(X_raw[cont_features]),
            index=X_raw.index, columns=cont_features
        )
        X_cont_scaled = pd.DataFrame(
            scaler.fit_transform(X_cont_imp),
            index=X_raw.index, columns=cont_features
        )
    else:
        X_cont_imp = pd.DataFrame(index=X_raw.index)
        X_cont_scaled = pd.DataFrame(index=X_raw.index)

    # ---------- 分类变量: 众数填补, 保持 0/1 ----------
    imputer_cat = SimpleImputer(strategy='most_frequent')

    if cat_features:
        X_cat_imp = pd.DataFrame(
            imputer_cat.fit_transform(X_raw[cat_features]),
            index=X_raw.index, columns=cat_features
        ).astype(int)
    else:
        X_cat_imp = pd.DataFrame(index=X_raw.index)

    # ---------- 拼接 ----------
    X_imputed = pd.concat([X_cont_imp, X_cat_imp], axis=1)           # 原始单位
    X_cluster = pd.concat([X_cont_scaled, X_cat_imp], axis=1)        # 聚类用

    # 恢复原始索引
    X_imputed.index = original_index
    X_cluster.index = original_index

    # 分类变量的列索引
    cat_indices = list(range(len(cont_features), len(cont_features) + len(cat_features)))

    print(f"✅ 预处理完成: 连续 {len(cont_features)} 个 (Z-score), "
          f"分类 {len(cat_features)} 个 (保持原值)")
    print(f"   分类变量列索引 (categorical): {cat_indices}")
    print(f"   聚类矩阵形状: {X_cluster.shape}")

    return X_cluster, X_imputed, cat_indices, imputer_cat, imputer_cont, scaler


# ============================================================
# 3. Gower 距离 (混合型数据的 Silhouette)
# ============================================================
def gower_distance_matrix(X, cat_indices):

    n, p = X.shape
    cont_indices = [i for i in range(p) if i not in cat_indices]
    D = np.zeros((n, n))

    X_arr = np.array(X, dtype=float)

    # 连续部分: 归一化范围
    if cont_indices:
        X_cont = X_arr[:, cont_indices]
        ranges = X_cont.max(axis=0) - X_cont.min(axis=0)
        ranges[ranges == 0] = 1.0
        for i in range(n):
            diff = np.abs(X_cont[i] - X_cont[i + 1:]) / ranges
            D[i, i + 1:] += diff.mean(axis=1)

    # 分类部分: 不匹配比例
    if cat_indices:
        X_cat = X_arr[:, cat_indices]
        for i in range(n):
            mismatch = (X_cat[i] != X_cat[i + 1:]).mean(axis=1)
            D[i, i + 1:] += mismatch

    # 平均
    n_parts = (1 if cont_indices else 0) + (1 if cat_indices else 0)
    D /= max(n_parts, 1)
    D = D + D.T
    return D


def silhouette_score_gower(X, labels, cat_indices):
    """基于 Gower 距离的 Silhouette Score (适用于混合型数据)"""
    D = gower_distance_matrix(X, cat_indices)
    return silhouette_score(D, labels, metric='precomputed')


# ============================================================
# 4. K 值评估 (Cost + Silhouette + CH Index)
# ============================================================
def evaluate_k(X, cat_indices, k_range=(2, 8), gamma=None):

    K_range = range(*k_range)
    costs, sil_scores, ch_scores = [], [], []

    X_arr = np.array(X)

    print("正在评估不同 K 值 (K-Prototypes) ...")
    for k in K_range:
        kp = KPrototypes(n_clusters=k, init='Cao', random_state=RANDOM_STATE,
                         n_init=10, gamma=gamma)
        labels = kp.fit_predict(X_arr, categorical=cat_indices)
        costs.append(kp.cost_)

        # Silhouette: 使用 Gower 距离
        sil = silhouette_score_gower(X, labels, cat_indices)
        sil_scores.append(sil)

        # CH Index: 在连续列上计算 (近似)
        cont_indices = [i for i in range(X_arr.shape[1]) if i not in cat_indices]
        if cont_indices:
            ch = calinski_harabasz_score(X_arr[:, cont_indices], labels)
        else:
            ch = 0
        ch_scores.append(ch)

        print(f"  K={k}: Cost={costs[-1]:.2f}, Silhouette(Gower)={sil:.3f}, CH={ch:.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cost (Elbow)
    axes[0].plot(K_range, costs, 'bo-', markersize=8, linewidth=2)
    axes[0].set_title('Elbow Method (K-Prototypes Cost)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[0].set_ylabel('Cost (Lower is Better)', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Silhouette Analysis
    axes[1].plot(K_range, sil_scores, 'ro-', markersize=8, linewidth=2)
    axes[1].set_title('Silhouette Analysis', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Calinski-Harabasz Index
    axes[2].plot(K_range, ch_scores, 'go-', markersize=8, linewidth=2)
    axes[2].set_title('Calinski-Harabasz Index', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[2].set_ylabel('CH Index (Higher is Better)', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_publication_fig("Figure_K_Selection.png")
    plt.show()

    best_sil_k = list(K_range)[np.argmax(sil_scores)]
    best_ch_k = list(K_range)[np.argmax(ch_scores)]
    print(f"\n📊 算法推荐:")
    print(f"   • 基于 Silhouette Score: K = {best_sil_k} (Score = {max(sil_scores):.3f})")
    print(f"   • 基于 Calinski-Harabasz: K = {best_ch_k} (Index = {max(ch_scores):.1f})")
    return best_sil_k, best_ch_k


# ============================================================
# 5. Bootstrap 稳定性评估
# ============================================================
def bootstrap_stability(X, cat_indices, k, n_bootstrap=100, gamma=None):

    print(f"\n🔁 Bootstrap 聚类稳定性测试 (K-Prototypes, 重采样次数={n_bootstrap})...")
    n = X.shape[0]
    X_arr = np.array(X)

    kp_base = KPrototypes(n_clusters=k, init='Cao', random_state=RANDOM_STATE,
                          n_init=20, gamma=gamma)
    labels_base = kp_base.fit_predict(X_arr, categorical=cat_indices)
    aris = []

    for i in range(n_bootstrap):
        indices = resample(range(n), n_samples=n, random_state=i)
        X_boot = X_arr[indices]
        kp_boot = KPrototypes(n_clusters=k, init='Cao', random_state=RANDOM_STATE,
                              n_init=10, gamma=gamma)
        labels_boot = kp_boot.fit_predict(X_boot, categorical=cat_indices)

        mapped = np.full(n, -1)
        for j, idx in enumerate(indices):
            mapped[idx] = labels_boot[j]
        valid = mapped != -1
        if valid.sum() > 0:
            aris.append(adjusted_rand_score(labels_base[valid], mapped[valid]))

    mean_ari = np.mean(aris)
    std_ari = np.std(aris)
    ci_lower = np.percentile(aris, 2.5)
    ci_upper = np.percentile(aris, 97.5)
    print(f"   📊 ARI统计:")
    print(f"      均值: {mean_ari:.3f} ± {std_ari:.3f}")
    print(f"      95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    if mean_ari > 0.8:
        print(f"      ✅ 结果评价: 极佳 (Highly Stable)")
    elif mean_ari > 0.6:
        print(f"      ✅ 结果评价: 可靠 (Stable)")
    elif mean_ari > 0.4:
        print(f"      ⚠️  结果评价: 中等 (Moderate)")
    else:
        print(f"      ❌ 结果评价: 不稳定 (Unstable) - 建议减少K值或检查数据")

    return mean_ari, aris


# ============================================================
# 6. 聚类执行 (K-Prototypes, 含复发风险排序)
# ============================================================
def run_clustering(X, cat_indices, k, y_event=None, gamma=None):

    X_arr = np.array(X)
    kp = KPrototypes(n_clusters=k, init='Cao', random_state=RANDOM_STATE,
                     n_init=20, gamma=gamma)
    labels = kp.fit_predict(X_arr, categorical=cat_indices)

    # ---- 按复发风险排序 ----
    if y_event is not None:
        y_arr = np.asarray(y_event)
        event_rates = {}
        for c in range(k):
            mask = labels == c
            event_rates[c] = y_arr[mask].mean() if mask.sum() > 0 else 0
        sorted_clusters = sorted(event_rates, key=event_rates.get)
        remap = {old: new for new, old in enumerate(sorted_clusters)}
        labels = np.array([remap[c] for c in labels])

        print(f"  ✅ 聚类标签已按复发风险升序重映射:")
        for new_id in range(k):
            old_id = sorted_clusters[new_id]
            print(f"     Cluster {new_id} ← 原 {old_id} (事件率 {event_rates[old_id]:.1%})")
    else:
        sorted_clusters = list(range(k))
        print("  ⚠️ 未提供结局变量，聚类标签未排序")

    # 分布
    dist = pd.Series(labels).value_counts().sort_index()
    print(f"\n📊 样本分布:")
    for cid, cnt in dist.items():
        print(f"   Cluster {cid}: {cnt:4d} ({cnt / len(labels) * 100:5.1f}%)")

    return kp, labels, sorted_clusters


# ============================================================
# 7. 外部验证集聚类分配 (K-Prototypes)
# ============================================================
def assign_external_clusters(kprot_model, X_internal, external_csv, tear_features,
                             cat_indices, imputer_cat, imputer_cont, scaler,
                             sorted_clusters=None,
                             cat_features=None, cont_features=None):

    cat_features = cat_features or Config.TEAR_CATEGORICAL
    cont_features = cont_features or Config.TEAR_CONTINUOUS

    print(f"\n📦 正在处理外部验证集...")
    try:
        df_ext = pd.read_csv(external_csv, index_col=Config.INDEX_COL)
        print(f"   原始样本数: {len(df_ext)}")
        print(f"   原始缺失值总数: {df_ext.isnull().sum().sum()}")

        # 过滤实际存在的特征
        avail_cont = [f for f in cont_features if f in tear_features and f in df_ext.columns]
        avail_cat = [f for f in cat_features if f in tear_features and f in df_ext.columns]

        missing_feats = set(tear_features) - set(avail_cont + avail_cat)
        if missing_feats:
            print(f"   ⚠️ 警告: 外部数据集缺失特征: {missing_feats}")

        # 连续变量: 用内部集中位数填补 + 标准化
        if avail_cont:
            X_ext_cont = pd.DataFrame(
                imputer_cont.transform(df_ext[avail_cont]),
                index=df_ext.index, columns=avail_cont
            )
            X_ext_cont_sc = pd.DataFrame(
                scaler.transform(X_ext_cont),
                index=df_ext.index, columns=avail_cont
            )
        else:
            X_ext_cont_sc = pd.DataFrame(index=df_ext.index)

        # 分类变量: 用内部集众数填补
        if avail_cat:
            X_ext_cat = pd.DataFrame(
                imputer_cat.transform(df_ext[avail_cat]),
                index=df_ext.index, columns=avail_cat
            ).astype(int)
        else:
            X_ext_cat = pd.DataFrame(index=df_ext.index)

        X_ext_merged = pd.concat([X_ext_cont_sc, X_ext_cat], axis=1)
        print(f"   ✅ 缺失值已填补 + 连续变量标准化（使用内部集参数）")

        ext_labels = kprot_model.predict(np.array(X_ext_merged), categorical=cat_indices)

        # 应用与内部集相同的风险排序重映射
        if sorted_clusters is not None:
            remap = {old: new for new, old in enumerate(sorted_clusters)}
            ext_labels = np.array([remap[c] for c in ext_labels])

        df_ext['Cluster_ID'] = ext_labels

        out = external_csv.replace('.csv', '_with_Cluster.csv')
        df_ext.to_csv(out, encoding='utf-8-sig')
        dist = pd.Series(ext_labels).value_counts().sort_index()
        print(f"   ✅ 外部验证集聚类完成: {out}")
        print(f"   📊 各簇样本分布:")
        for cid, cnt in dist.items():
            print(f"      Cluster {cid}: {cnt:3d} ({cnt / len(ext_labels) * 100:.1f}%)")
        return df_ext
    except FileNotFoundError:
        print(f"   ❌ 错误: 未找到文件 {external_csv}")
        return None
    except Exception as e:
        print(f"   ❌ 处理外部验证集时出错: {e}")
        return None


# ============================================================
# 8. 可视化: 树状图 (Gower 距离)
# ============================================================
def plot_dendrogram(X, cat_indices=None):

    plt.figure(figsize=(14, 7))
    if cat_indices:
        D = gower_distance_matrix(X, cat_indices)
        condensed = squareform(D)
        Z = linkage(condensed, method='average')  # Gower 距离用 average linkage
        method_label = "Average Linkage + Gower Distance"
    else:
        Z = linkage(X, method='ward')
        method_label = "Ward's Method"

    dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True, leaf_font_size=10)
    plt.title(f"Hierarchical Clustering Dendrogram ({method_label})",
              fontsize=14, fontweight='bold')
    plt.xlabel("Sample Index or Cluster Size", fontsize=12)
    plt.ylabel("Linkage Distance", fontsize=12)
    plt.legend()
    save_publication_fig("Figure_Dendrogram.png")
    plt.show()
    print("✅ 树状图已保存")


# ============================================================
# 9. 可视化: t-SNE (Gower 距离)
# ============================================================
def plot_tsne(X, labels, k, cat_indices=None):
    print("正在运行 t-SNE (Gower 距离, 可能需要1-2分钟)...")
    if cat_indices:
        D = gower_distance_matrix(X, cat_indices)
        tsne = TSNE(n_components=2,
                    perplexity=min(30, len(X) // 4),
                    metric='precomputed',
                    init='random',       # precomputed 不支持 pca init
                    learning_rate='auto',
                    random_state=RANDOM_STATE, n_jobs=-1)
        emb = tsne.fit_transform(D)
    else:
        tsne = TSNE(n_components=2,
                    perplexity=min(30, len(X) // 4),
                    init='pca', learning_rate='auto',
                    random_state=RANDOM_STATE, n_jobs=-1)
        emb = tsne.fit_transform(X)

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='viridis',
                          s=120, alpha=0.7, edgecolors='white', linewidths=0.5)
    plt.colorbar(scatter, label='Cluster ID', ticks=range(k))
    plt.title(f"t-SNE Visualization of Retinal Tear Phenotypes (K={k})",
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.2)
    save_publication_fig("Figure_tSNE_Clusters.png")
    plt.show()
    print("✅ t-SNE可视化完成")


# ============================================================
# 10. 可视化: 聚类特征热图 (K-Prototypes)
# ============================================================
def plot_cluster_heatmap(kprot_model, X, tear_features, k, cat_indices=None,
                        sorted_clusters=None, cluster_labels=None):

    cat_indices = cat_indices or []
    col_names = list(X.columns) if hasattr(X, 'columns') else [f'F{i}' for i in range(X.shape[1])]
    X_arr = np.array(X, dtype=float)

    # 直接从数据计算每簇中心
    centers_data = np.zeros((k, X_arr.shape[1]))
    for cid in range(k):
        mask = cluster_labels == cid
        if mask.sum() > 0:
            # 连续列: 均值; 分类列: 众数 (用均值近似, 0/1变量均值=比例)
            centers_data[cid] = X_arr[mask].mean(axis=0)

    centers = pd.DataFrame(
        centers_data, columns=col_names,
        index=[f'Cluster {i}' for i in range(k)]
    )
    n_top = min(25, len(col_names))
    top_feats = centers.var(axis=0).nlargest(n_top).index.tolist()

    # SCI 规范标签映射
    display_labels = [get_display_label(f) for f in top_feats]

    plt.figure(figsize=(16, 10))
    heatmap_data = centers[top_feats].T.copy()
    heatmap_data.index = display_labels
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True,
                fmt=".2f", linewidths=0.5,
                cbar_kws={'label': 'Feature Value (Z-Score for continuous, 0/1 for categorical)'},
                annot_kws={'size': 9})
    plt.title(f"Differentiating Features by Cluster (K-Prototypes)",
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.yticks(rotation=0)
    save_publication_fig("Figure_Cluster_Heatmap.png")
    plt.show()
    print(f"✅ 特征热图完成 (展示前{n_top}个差异特征)")


# ============================================================
# 10. 临床画像表 (原始单位)
# ============================================================
def generate_clinical_profile(labels, df_original, tear_features,
                              filename='Cluster_Clinical_Profile.csv'):
    print(f"\n📋 生成临床画像表...")
    df = df_original[tear_features].copy()
    df['Cluster_ID'] = labels

    profile = {}
    for cid in sorted(df['Cluster_ID'].unique()):
        sub = df[df['Cluster_ID'] == cid]
        col_name = f'Cluster {cid}\n(n={len(sub)})'
        profile[col_name] = {}
        for feat in tear_features:
            if feat not in sub.columns:
                continue
            vals = sub[feat].dropna()
            # 判断是连续变量还是分类变量
            if vals.nunique() <= 5 or set(vals.unique()).issubset({0, 1}):
                if set(vals.unique()).issubset({0, 1}):
                    # 二分类变量
                    pct = (vals == 1).sum() / len(vals) * 100
                    profile[col_name][feat] = f"{pct:.1f}%"
                else:
                    # 多分类变量 - 显示众数
                    mode_val = vals.mode().iloc[0] if len(vals.mode()) > 0 else 'N/A'
                    profile[col_name][feat] = str(mode_val)
            else:
                # 连续变量 - 显示均值±标准差
                if vals.std() > 0:
                    profile[col_name][feat] = f"{vals.mean():.2f} ± {vals.std():.2f}"
                else:
                    profile[col_name][feat] = f"{vals.median():.2f}"

    profile_df = pd.DataFrame(profile)
    # SCI 规范标签映射
    profile_df.index = [get_display_label(f) for f in profile_df.index]
    out_path = os.path.join(TABLE_DIR, filename)
    profile_df.to_csv(out_path, encoding='utf-8-sig')
    print(f"   ✅ 临床画像表已保存: {out_path}")
    print(f"\n[临床画像预览 - 前5个特征]")
    print(profile_df.head())
    return profile_df


# ============================================================
# 11. Kaplan-Meier 生存曲线
# ============================================================
def plot_km_survival(df, cluster_col='Cluster_ID',
                     time_col='Follow_up_Time', event_col='Recurrence',
                     fig_name='Figure_KM_Survival.png'):

    if not LIFELINES_AVAILABLE:
        print("⚠️ lifelines 未安装, 跳过 KM 分析")
        return None

    time_candidates = [time_col, 'Follow_up_Time', 'Followup_Time',
                       'Time_to_Event', 'Tamponade_Duration', 'Time_to_SOR']
    actual_time = None
    for tc in time_candidates:
        if tc in df.columns:
            actual_time = tc
            break
    if actual_time is None:
        print("⚠️ 未找到随访时间列, 跳过 KM 分析")
        return None
    if event_col not in df.columns:
        print(f"⚠️ 未找到事件列 '{event_col}', 跳过 KM 分析")
        return None

    df_clean = df[[cluster_col, actual_time, event_col]].dropna()
    print(f"\n📈 绘制 Kaplan-Meier 生存曲线...")
    print(f"   有效样本数: {len(df_clean)} / {len(df)}")

    plt.figure(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    clusters = sorted(df_clean[cluster_col].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(clusters), 3)))

    for i, cid in enumerate(clusters):
        mask = df_clean[cluster_col] == cid
        n_c = mask.sum()
        n_ev = int(df_clean.loc[mask, event_col].sum())
        kmf.fit(df_clean.loc[mask, actual_time],
                df_clean.loc[mask, event_col],
                label=f'Cluster {cid} (n={n_c}, events={n_ev})')
        kmf.plot_survival_function(ci_show=True, color=colors[i], linewidth=2.5)

    # Log-rank
    p_value = None
    try:
        lr = multivariate_logrank_test(
            df_clean[actual_time], df_clean[cluster_col], df_clean[event_col])
        p_value = lr.p_value

        plt.title(f"Recurrence-Free Survival by Retinal Tear Phenotype\n"
                  f"Log-rank test: p = {p_value:.4f}" +
                  (" *" if p_value < 0.05 else ""),
                  fontsize=14, fontweight='bold')

        print(f"   📊 Log-rank Test: p = {p_value:.4f}")
        if p_value < 0.001:
            print(f"      ✅ 各簇复发风险差异极显著 (p < 0.001)")
        elif p_value < 0.05:
            print(f"      ✅ 各簇复发风险差异显著 (p < 0.05)")
        else:
            print(f"      ⚠️  聚类未能显著区分预后 (p ≥ 0.05)")

    except Exception as e:
        print(f"   ⚠️ Log-rank检验失败: {e}")
        plt.title("Recurrence-Free Survival by Retinal Tear Phenotype",
                  fontsize=14, fontweight='bold')

    plt.xlabel('Time to Recurrence (Months)', fontsize=12)
    plt.ylabel('Recurrence-Free Probability', fontsize=12)
    plt.xlim(left=0); plt.ylim([0, 1.05])
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    save_publication_fig(fig_name)
    plt.show()

    return {'p_value': p_value}



# ============================================================
# 13. NRI / IDI 计算
# ============================================================
def compute_nri(y_true, prob_old, prob_new, threshold=None):

    y = np.array(y_true, dtype=int)
    po = np.array(prob_old, dtype=float)
    pn = np.array(prob_new, dtype=float)

    ev = y == 1
    nev = y == 0
    n_ev = ev.sum()
    n_nev = nev.sum()
    results = {}

    # ---- 1. Category-free NRI (cfNRI) ----
    eu = np.sum(pn[ev] > po[ev])
    ed = np.sum(pn[ev] < po[ev])
    nu = np.sum(pn[nev] > po[nev])
    nd = np.sum(pn[nev] < po[nev])

    nri_e = (eu - ed) / n_ev if n_ev > 0 else 0
    nri_n = (nd - nu) / n_nev if n_nev > 0 else 0
    cfnri = nri_e + nri_n

    se_e = np.sqrt((eu + ed) / n_ev ** 2) if n_ev > 0 else 0
    se_n = np.sqrt((nu + nd) / n_nev ** 2) if n_nev > 0 else 0
    se = np.sqrt(se_e ** 2 + se_n ** 2)
    z = cfnri / se if se > 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))

    results['cfNRI'] = {
        'value': cfnri, 'events_component': nri_e, 'non_events_component': nri_n,
        'se': se, 'z': z, 'p_value': p,
        '95CI': [cfnri - 1.96 * se, cfnri + 1.96 * se],
        'detail': {'event_up': int(eu), 'event_down': int(ed),
                   'nonevent_up': int(nu), 'nonevent_down': int(nd)}
    }

    # ---- 2. Category-based NRI----
    if threshold is not None:
        if isinstance(threshold, (int, float)):
            thresholds = [threshold]
        else:
            thresholds = sorted(threshold)

        def classify(probs, cuts):
            return np.digitize(probs, cuts)

        co = classify(po, thresholds)
        cn = classify(pn, thresholds)

        euc = np.sum(cn[ev] > co[ev])
        edc = np.sum(cn[ev] < co[ev])
        nuc = np.sum(cn[nev] > co[nev])
        ndc = np.sum(cn[nev] < co[nev])

        nri_ec = (euc - edc) / n_ev if n_ev > 0 else 0
        nri_nc = (ndc - nuc) / n_nev if n_nev > 0 else 0
        cat_nri = nri_ec + nri_nc

        se_c = np.sqrt(
            (euc + edc) / n_ev ** 2 + (nuc + ndc) / n_nev ** 2
        ) if (n_ev > 0 and n_nev > 0) else 0
        z_c = cat_nri / se_c if se_c > 0 else 0
        p_c = 2 * (1 - norm.cdf(abs(z_c)))

        n_cats = len(thresholds) + 1
        reclass_table_events = np.zeros((n_cats, n_cats), dtype=int)
        reclass_table_non_events = np.zeros((n_cats, n_cats), dtype=int)
        for i in range(n_cats):
            for j in range(n_cats):
                reclass_table_events[i, j] = np.sum((co[ev] == i) & (cn[ev] == j))
                reclass_table_non_events[i, j] = np.sum((co[nev] == i) & (cn[nev] == j))

        results['categorical_NRI'] = {
            'value': cat_nri, 'events_component': nri_ec, 'non_events_component': nri_nc,
            'se': se_c, 'z': z_c, 'p_value': p_c,
            '95CI': [cat_nri - 1.96 * se_c, cat_nri + 1.96 * se_c],
            'thresholds_used': thresholds,
            'reclassification_table_events': reclass_table_events,
            'reclassification_table_non_events': reclass_table_non_events,
            'detail': {
                'event_up': int(euc), 'event_down': int(edc),
                'nonevent_up': int(nuc), 'nonevent_down': int(ndc)
            }
        }

    # ---- 3. IDI (Integrated Discrimination Improvement) ----
    idi_e = np.mean(pn[ev]) - np.mean(po[ev])
    idi_n = np.mean(pn[nev]) - np.mean(po[nev])
    idi = idi_e - idi_n

    n_boot = 1000
    boot_idis = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        yt, p_o, p_n = y[idx], po[idx], pn[idx]
        e_, ne_ = yt == 1, yt == 0
        if e_.sum() > 0 and ne_.sum() > 0:
            boot_idis.append(
                (np.mean(p_n[e_]) - np.mean(p_o[e_])) -
                (np.mean(p_n[ne_]) - np.mean(p_o[ne_]))
            )
    se_idi = np.std(boot_idis) if boot_idis else 0
    z_idi = idi / se_idi if se_idi > 0 else 0
    p_idi = 2 * (1 - norm.cdf(abs(z_idi)))

    results['IDI'] = {
        'value': idi, 'events_component': idi_e, 'non_events_component': idi_n,
        'se': se_idi, 'z': z_idi, 'p_value': p_idi,
        '95CI': [idi - 1.96 * se_idi, idi + 1.96 * se_idi]
    }
    return results


# ============================================================
# 14. 风险分层计算
# ============================================================
def compute_risk_strata(y_true, y_prob, threshold):

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    tertile_cuts = np.percentile(y_prob, [33.33, 66.67])
    risk_group_tertile = np.digitize(y_prob, tertile_cuts)
    risk_group_binary = (y_prob >= threshold).astype(int)
    fixed_cuts = [0.10, 0.30]
    risk_group_fixed = np.digitize(y_prob, fixed_cuts)

    def group_stats(groups, n_grp):
        stats = {}
        for g in range(n_grp):
            mask = groups == g
            n_total = mask.sum()
            if n_total == 0:
                stats[g] = {'n': 0, 'events': 0, 'event_rate': 0, 'mean_prob': 0}
                continue
            stats[g] = {
                'n': int(n_total),
                'events': int(y_true[mask].sum()),
                'non_events': int(n_total - y_true[mask].sum()),
                'event_rate': float(y_true[mask].mean()),
                'mean_prob': float(y_prob[mask].mean()),
                'prob_range': [float(y_prob[mask].min()), float(y_prob[mask].max())]
            }
        return stats

    return {
        'y_true': y_true, 'y_prob': y_prob,
        'tertile': {
            'cuts': tertile_cuts.tolist(), 'groups': risk_group_tertile,
            'labels': ['Low', 'Medium', 'High'],
            'stats': group_stats(risk_group_tertile, 3)
        },
        'binary': {
            'threshold': float(threshold), 'groups': risk_group_binary,
            'labels': ['Low Risk', 'High Risk'],
            'stats': group_stats(risk_group_binary, 2)
        },
        'fixed': {
            'cuts': fixed_cuts, 'groups': risk_group_fixed,
            'labels': ['Low (<10%)', 'Medium (10-30%)', 'High (>30%)'],
            'stats': group_stats(risk_group_fixed, 3)
        }
    }


# ============================================================
# 15. 裂孔聚类特征生成 (K-Prototypes)
# ============================================================
def create_break_cluster(train_df, external_df, cfg):

    print(f"\n{'='*60}")
    print("🔧 裂孔特征聚类 (K-Prototypes)")
    print(f"{'='*60}")

    break_num_feats = cfg.BREAK_NUMERIC_FEATURES
    break_cat_feats = cfg.BREAK_CATEGORICAL_FEATURES
    n_clusters = cfg.K_FINAL or 3
    print(f"  连续裂孔特征: {break_num_feats}")
    print(f"  分类裂孔特征: {break_cat_feats}")
    print(f"  聚类数: {n_clusters}")

    # 提取裂孔相关列 (连续在前, 分类在后)
    ordered_feats = break_num_feats + break_cat_feats
    X_tr = train_df[ordered_feats].copy()
    X_ex = external_df[ordered_feats].copy()

    # 数值型转换
    for c in break_num_feats:
        X_tr[c] = pd.to_numeric(X_tr[c], errors='coerce')
        X_ex[c] = pd.to_numeric(X_ex[c], errors='coerce')

    # 分类变量编码 (如果非 0/1 需要 LabelEncoder)
    les = {}
    for c in break_cat_feats:
        unique_vals = pd.concat([X_tr[c], X_ex[c]]).dropna().unique()
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            le = LabelEncoder()
            combined = pd.concat([X_tr[c], X_ex[c]]).dropna().unique()
            le.fit(combined)
            X_tr[c] = X_tr[c].map(lambda x, _l=le: _l.transform([x])[0] if pd.notna(x) else np.nan)
            X_ex[c] = X_ex[c].map(lambda x, _l=le: _l.transform([x])[0] if pd.notna(x) else np.nan)
            les[c] = le

    # 连续变量: 中位数填补 + 标准化
    imp_cont = SimpleImputer(strategy='median')
    sc = StandardScaler()

    if break_num_feats:
        X_tr_cont = pd.DataFrame(
            imp_cont.fit_transform(X_tr[break_num_feats]),
            columns=break_num_feats, index=X_tr.index
        )
        X_ex_cont = pd.DataFrame(
            imp_cont.transform(X_ex[break_num_feats]),
            columns=break_num_feats, index=X_ex.index
        )
        X_tr_cont_s = pd.DataFrame(
            sc.fit_transform(X_tr_cont), columns=break_num_feats, index=X_tr.index
        )
        X_ex_cont_s = pd.DataFrame(
            sc.transform(X_ex_cont), columns=break_num_feats, index=X_ex.index
        )
    else:
        X_tr_cont_s = pd.DataFrame(index=X_tr.index)
        X_ex_cont_s = pd.DataFrame(index=X_ex.index)

    # 分类变量: 众数填补
    imp_cat = SimpleImputer(strategy='most_frequent')

    if break_cat_feats:
        X_tr_cat = pd.DataFrame(
            imp_cat.fit_transform(X_tr[break_cat_feats]),
            columns=break_cat_feats, index=X_tr.index
        ).astype(int)
        X_ex_cat = pd.DataFrame(
            imp_cat.transform(X_ex[break_cat_feats]),
            columns=break_cat_feats, index=X_ex.index
        ).astype(int)
    else:
        X_tr_cat = pd.DataFrame(index=X_tr.index)
        X_ex_cat = pd.DataFrame(index=X_ex.index)

    # 合并: 连续在前, 分类在后
    X_tr_merged = pd.concat([X_tr_cont_s, X_tr_cat], axis=1)
    X_ex_merged = pd.concat([X_ex_cont_s, X_ex_cat], axis=1)
    cat_indices = list(range(len(break_num_feats), len(break_num_feats) + len(break_cat_feats)))

    # K-Prototypes 聚类
    kp = KPrototypes(n_clusters=n_clusters, init='Cao', random_state=RANDOM_STATE,
                     n_init=10, gamma=cfg.GAMMA)
    tr_labels = kp.fit_predict(np.array(X_tr_merged), categorical=cat_indices)
    ex_labels = kp.predict(np.array(X_ex_merged), categorical=cat_indices)

    # 按事件率升序 remap（0=低风险, 1=中风险, 2=高风险）
    y_tr = train_df[cfg.TARGET_COL].values
    cluster_event_rates = {}
    for c in range(n_clusters):
        mask = tr_labels == c
        cluster_event_rates[c] = y_tr[mask].mean() if mask.sum() > 0 else 0

    sorted_clusters = sorted(cluster_event_rates, key=cluster_event_rates.get)
    remap = {old: new for new, old in enumerate(sorted_clusters)}
    tr_labels = np.array([remap[c] for c in tr_labels])
    ex_labels = np.array([remap[c] for c in ex_labels])

    # 添加到 DataFrame
    train_new = train_df.copy()
    external_new = external_df.copy()
    train_new[cfg.BREAK_CLUSTER_COL] = tr_labels
    external_new[cfg.BREAK_CLUSTER_COL] = ex_labels

    # 打印聚类结果
    cluster_labels_name = {0: '低风险裂孔', 1: '中风险裂孔', 2: '高风险裂孔'} if n_clusters == 3 else \
                          {i: f'裂孔类型_{i}' for i in range(n_clusters)}

    print(f"\n  聚类结果（按复发风险排序）:")
    for nid in range(n_clusters):
        oid = sorted_clusters[nid]
        mask_train = tr_labels == nid
        mask_ext = ex_labels == nid
        label_name = cluster_labels_name.get(nid, f'Cluster_{nid}')
        print(f"    Cluster {nid} ({label_name}): "
              f"训练集 n={mask_train.sum()}, 外部集 n={mask_ext.sum()}, "
              f"训练集事件率={cluster_event_rates[oid]:.1%}")

    # 聚类模型打包
    cluster_model = {
        'kprototypes': kp, 'scaler': sc,
        'imputer_cont': imp_cont, 'imputer_cat': imp_cat,
        'label_encoders': les, 'cluster_remap': remap,
        'break_num_features': break_num_feats,
        'break_cat_features': break_cat_feats,
        'cat_indices': cat_indices,
        'n_clusters': n_clusters,
        'cluster_event_rates': cluster_event_rates,
    }

    return train_new, external_new, cluster_model


# ============================================================
# 16. 从 pkl 中安全提取特征列表
# ============================================================
def extract_feature_lists_from_pkg(pkg):

    feat_info = pkg.get('feature_names', pkg.get('feature_info', {}))

    if isinstance(feat_info, dict):
        log_feats = feat_info.get('log', feat_info.get('log_features', []))
        num_feats = feat_info.get('numeric', feat_info.get('num_features', []))
        cat_feats = feat_info.get('categorical', feat_info.get('cat_features', []))
    else:
        log_feats = ['Symptom_Duration']
        num_feats = ['AL', 'RD_Extent', 'number_of_breaks', 'Largest_Break_Diameter']
        cat_feats = ['PVR_Grade_Pre', 'Break_Loc_Inferior', 'Choroidal_Detachment',
                     'Lens_Status_Pre', 'Macular_status']

    return log_feats, num_feats, cat_feats


# ============================================================
# 17. 敏感性分析完整流程
# ============================================================
def run_sensitivity_analysis(cfg, timestamp):

    print(f"\n{'='*80}")
    print("  敏感性分析 — 裂孔聚类特征替换")
    print(f"  将 {cfg.BREAK_NUMERIC_FEATURES + cfg.BREAK_CATEGORICAL_FEATURES}")
    print(f"  替换为 1 个裂孔聚类特征 (K={cfg.K_FINAL or 3})")
    print(f"{'='*80}")

    # --------------------------------------------------------
    # 1. 加载数据（裂孔聚类替换）
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("敏感性分析 - 加载数据（裂孔聚类替换）")
    print(f"{'='*60}")

    train_df = pd.read_csv(cfg.TRAIN_DATA, index_col=0)
    external_df = pd.read_csv(cfg.EXTERNAL_DATA, index_col=0)
    train_df = train_df.dropna(subset=[cfg.TARGET_COL]).reset_index(drop=True)
    external_df = external_df.dropna(subset=[cfg.TARGET_COL]).reset_index(drop=True)
    y_train = train_df[cfg.TARGET_COL].astype(int)
    y_ext = external_df[cfg.TARGET_COL].astype(int)

    # 生成裂孔聚类特征
    train_df, external_df, cluster_model = create_break_cluster(train_df, external_df, cfg)

    # 特征列表 = 非裂孔术前变量 + 裂孔聚类特征
    log_feats = cfg.LOG_FEATURES
    num_feats = cfg.NUM_FEATURES
    cat_feats = cfg.CAT_FEATURES + [cfg.BREAK_CLUSTER_COL]
    all_feats = log_feats + num_feats + cat_feats

    # 验证
    missing = [f for f in all_feats if f not in train_df.columns]
    if missing:
        raise ValueError(f"❌ 训练集缺少特征: {missing}")

    X_train = train_df[all_feats].copy()
    X_ext = external_df[all_feats].copy()
    for c in num_feats + log_feats:
        X_train[c] = pd.to_numeric(X_train[c], errors='coerce')
        X_ext[c] = pd.to_numeric(X_ext[c], errors='coerce')

    print(f"\n  原术前模型裂孔特征数: {len(cfg.BREAK_NUMERIC_FEATURES) + len(cfg.BREAK_CATEGORICAL_FEATURES)} 个")
    print(f"    被替换的特征: {cfg.BREAK_NUMERIC_FEATURES + cfg.BREAK_CATEGORICAL_FEATURES}")
    print(f"  裂孔聚类特征数: 1 个 ({cfg.BREAK_CLUSTER_COL})")
    print(f"  总特征数: {len(all_feats)} 个 (原模型: {len(all_feats) + 2} 个)")
    print(f"  训练集: {len(X_train)} 例, 外部集: {len(X_ext)} 例")

    # --------------------------------------------------------
    # 2. 构建预处理器 & 训练
    # --------------------------------------------------------
    preprocessor = make_column_transformer(log_feats, num_feats, cat_feats)
    trainer = MultiModelTrainer(preprocessor, random_state=cfg.RANDOM_STATE)
    results = trainer.train_all(X_train, y_train, X_ext, y_ext,
                                cv_folds=cfg.CV_FOLDS, cv_repeats=cfg.CV_REPEATS)

    best_model_cluster = results['final_model']
    best_name_cluster = results['best_model_name']
    threshold_cluster = results['optimal_threshold']

    # 获取裂孔聚类模型的预测概率
    prob_train_cluster = results['prob_train']
    prob_ext_cluster = results['prob_external']
    prob_oof_cluster = results.get('prob_oof', None)

    # --------------------------------------------------------
    # 3. 裂孔聚类模型风险分层
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("📊 裂孔聚类模型 — 风险分层分析")
    print(f"{'='*60}")

    #训练集风险分层：优先使用 OOF 概率（与 NRI 数据源保持一致）
    prob_train_cluster_for_strat = prob_oof_cluster if prob_oof_cluster is not None else prob_train_cluster

    risk_strat_cluster = {
        'train': compute_risk_strata(y_train, prob_train_cluster_for_strat, threshold_cluster),
        'external': compute_risk_strata(y_ext, prob_ext_cluster, threshold_cluster)
    }

    for ds_name, strat in risk_strat_cluster.items():
        print(f"\n  [{ds_name}] 三分位风险分层:")
        for g, label in enumerate(strat['tertile']['labels']):
            s = strat['tertile']['stats'][g]
            print(f"    {label}: n={s['n']}, 事件={s['events']}, "
                  f"事件率={s['event_rate']:.1%}, 平均概率={s['mean_prob']:.3f}")

    # --------------------------------------------------------
    # 4. 加载术前模型 → 计算 NRI
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("📊 NRI 分析 — 原术前模型 vs 裂孔聚类模型 比较")
    print(f"{'='*60}")

    nri_available = False
    nri_train, nri_ext = None, None
    prob_train_preop, prob_ext_preop = None, None
    preop_threshold = None
    preop_oof_probs = None

    try:
        preop_pkg = joblib.load(cfg.PREOP_MODEL_PKL)
        print(f"  ✓ 已加载术前模型: {cfg.PREOP_MODEL_PKL}")

        preop_model = preop_pkg['best_model']
        preop_threshold = preop_pkg['optimal_threshold']

        preop_log, preop_num, preop_cat = extract_feature_lists_from_pkg(preop_pkg)
        preop_features = preop_log + preop_num + preop_cat

        print(f"  ✓ 原术前模型特征: {preop_features}")
        print(f"  ✓ 裂孔聚类模型特征: {log_feats + num_feats + cat_feats}")

        # 用原术前特征子集预测
        X_train_preop = train_df[preop_features].copy()
        X_ext_preop = external_df[preop_features].copy()
        for col in preop_num + preop_log:
            X_train_preop[col] = pd.to_numeric(X_train_preop[col], errors='coerce')
            X_ext_preop[col] = pd.to_numeric(X_ext_preop[col], errors='coerce')

        prob_train_preop = preop_model.predict_proba(X_train_preop)[:, 1]
        prob_ext_preop = preop_model.predict_proba(X_ext_preop)[:, 1]

        # 尝试获取术前模型的 OOF 预测
        preop_oof_probs = None
        if 'predictions' in preop_pkg and isinstance(preop_pkg['predictions'], dict):
            preop_oof_probs = preop_pkg['predictions'].get('internal_oof_probs', None)
        if preop_oof_probs is None and 'results' in preop_pkg:
            preop_results = preop_pkg['results']
            preop_oof_probs = preop_results.get('prob_oof', None)

        # 判断是否可用 OOF
        use_oof = (preop_oof_probs is not None and prob_oof_cluster is not None)

        # ---- 计算 NRI ----
        # 训练集 NRI：优先使用 OOF 预测（公平比较）
        if use_oof:
            print(f"  ✓ 训练集 NRI 使用 OOF 预测（公平比较）")
            nri_train = compute_nri(y_train, preop_oof_probs, prob_oof_cluster,
                                    threshold=preop_threshold)
        else:
            print(f"  ⚠️ OOF 预测不可用，训练集 NRI 使用 resubstitution（仅供参考）")
            nri_train = compute_nri(y_train, prob_train_preop, prob_train_cluster,
                                    threshold=preop_threshold)

        # 外部验证集 NRI
        nri_ext = compute_nri(y_ext, prob_ext_preop, prob_ext_cluster,
                              threshold=preop_threshold)

        # 打印结果
        for ds_name, nri_res in [('训练集', nri_train), ('外部验证集', nri_ext)]:
            print(f"\n  [{ds_name}] NRI 结果:")
            cf = nri_res['cfNRI']
            print(f"    cfNRI:  {cf['value']:.4f} (95%CI: {cf['95CI'][0]:.4f}-{cf['95CI'][1]:.4f}), p={cf['p_value']:.4f}")
            print(f"      事件组: {cf['events_component']:.4f}, 非事件组: {cf['non_events_component']:.4f}")

            if 'categorical_NRI' in nri_res:
                cat = nri_res['categorical_NRI']
                print(f"    分类NRI: {cat['value']:.4f} (95%CI: {cat['95CI'][0]:.4f}-{cat['95CI'][1]:.4f}), p={cat['p_value']:.4f}")

            idi = nri_res['IDI']
            print(f"    IDI:    {idi['value']:.4f} (95%CI: {idi['95CI'][0]:.4f}-{idi['95CI'][1]:.4f}), p={idi['p_value']:.4f}")

        # ----AUC 对比 ----
        prob_train_preop_for_auc = preop_oof_probs if use_oof else prob_train_preop
        prob_train_cluster_for_auc = prob_oof_cluster if use_oof else prob_train_cluster
        auc_train_preop = roc_auc_score(y_train, prob_train_preop_for_auc)
        auc_ext_preop = roc_auc_score(y_ext, prob_ext_preop)
        auc_train_cluster = roc_auc_score(y_train, prob_train_cluster_for_auc)
        auc_ext_cluster = roc_auc_score(y_ext, prob_ext_cluster)
        auc_source_label = 'OOF' if use_oof else 'resub'
        print(f"\n  AUC 对比 (训练集数据源: {auc_source_label}):")
        print(f"    训练集:   原模型 AUC={auc_train_preop:.4f} → 聚类模型 AUC={auc_train_cluster:.4f} (Δ={auc_train_cluster-auc_train_preop:+.4f})")
        print(f"    外部集:   原模型 AUC={auc_ext_preop:.4f} → 聚类模型 AUC={auc_ext_cluster:.4f} (Δ={auc_ext_cluster-auc_ext_preop:+.4f})")

        nri_available = True

    except FileNotFoundError:
        print(f"  ⚠️ 未找到术前模型文件: {cfg.PREOP_MODEL_PKL}")
        print(f"  请先运行 2_模型训练.py，或修改 PREOP_MODEL_PKL 路径")
        use_oof = False

    # --------------------------------------------------------
    # 5. 保存所有数据包
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("💾 保存敏感性分析数据包...")
    print(f"{'='*60}")

    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_ext_arr = y_ext.values if hasattr(y_ext, 'values') else np.array(y_ext)

    curve_data_cluster = results.get('curve_data', {})

    # A. 裂孔聚类模型完整数据包
    sensitivity_package = {
        'best_model': best_model_cluster,
        'best_model_name': best_name_cluster,
        'optimal_threshold': threshold_cluster,
        'results': results,
        'datasets': {
            'X_train': X_train, 'y_train': y_train,
            'X_ext': X_ext, 'y_ext': y_ext
        },
        # 同时保存两种格式的特征信息，确保下游兼容
        'feature_info': {
            'log_features': log_feats,
            'num_features': num_feats,
            'cat_features': cat_feats
        },
        'feature_names': {
            'log': log_feats,
            'numeric': num_feats,
            'categorical': cat_feats
        },
        'predictions': {
            'train_probs': prob_train_cluster,
            'external_probs': prob_ext_cluster,
            'internal_oof_probs': prob_oof_cluster,
        },
        'curve_data': curve_data_cluster,
        'risk_stratification': risk_strat_cluster,
        #聚类模型信息
        'cluster_model': cluster_model,
        'metadata': {
            'timestamp': timestamp,
            'analysis_type': 'sensitivity_break_cluster',
            'description': '敏感性分析：裂孔聚类特征替换多个裂孔变量',
            'replaced_features': cfg.BREAK_NUMERIC_FEATURES + cfg.BREAK_CATEGORICAL_FEATURES,
            'new_feature': cfg.BREAK_CLUSTER_COL,
            'n_clusters': cluster_model['n_clusters'],
            'internal_validation_method': 'OOF (Out-of-Fold)',
        }
    }

    sens_pkl = f"{cfg.OUTPUT_DIR}/Sensitivity_BreakCluster_Package_{timestamp}.pkl"
    joblib.dump(sensitivity_package, sens_pkl)
    print(f"  ✓ 裂孔聚类模型数据包: {sens_pkl}")

    # B. NRI 比较专用数据包

    if nri_available:

        # --- 确定训练集使用的概率（OOF）---
        prob_train_preop_for_pkg = preop_oof_probs if use_oof else prob_train_preop
        prob_train_cluster_for_pkg = prob_oof_cluster if use_oof else prob_train_cluster

        # --- 基于实际使用概率计算训练集风险分层 ---
        risk_strat_train_preop = compute_risk_strata(
            y_train, prob_train_preop_for_pkg, preop_threshold)
        risk_strat_train_cluster = compute_risk_strata(
            y_train, prob_train_cluster_for_pkg, threshold_cluster)

        nri_comparison_package = {
            # 标签
            'y_train': np.array(y_train),
            'y_external': np.array(y_ext),

            # 原术前模型（基线模型）
            'preop_model': {
                'name': 'Original Preoperative Model (separate break features)',
                # 训练集概率使用 OOF（与 NRI 计算一致）
                'prob_train': np.array(prob_train_preop_for_pkg),
                'prob_train_resub': np.array(prob_train_preop),   # 保留 resubstitution 备用
                'prob_external': np.array(prob_ext_preop),
                'threshold': preop_threshold,
                # 风险分层也基于 OOF 概率
                'risk_strat_train': risk_strat_train_preop,
                'risk_strat_external': compute_risk_strata(y_ext, prob_ext_preop, preop_threshold),
            },

            # 裂孔聚类模型
            'cluster_model': {
                'name': 'Break Cluster Model (clustered break features)',
                # 训练集概率使用 OOF（与 NRI 计算一致）
                'prob_train': np.array(prob_train_cluster_for_pkg),
                'prob_train_resub': np.array(prob_train_cluster),  # 保留 resubstitution 备用
                'prob_external': np.array(prob_ext_cluster),
                'threshold': threshold_cluster,
                # 风险分层也基于 OOF 概率
                'risk_strat_train': risk_strat_train_cluster,
                'risk_strat_external': risk_strat_cluster['external'],
            },

            # NRI/IDI 结果
            'nri_results': {
                'train': nri_train,
                'external': nri_ext
            },

            # AUC 对比 — 训练集也统一使用 OOF 概率
            'auc_comparison': {
                'train': {
                    'preop': roc_auc_score(y_train, prob_train_preop_for_pkg),
                    'cluster': roc_auc_score(y_train, prob_train_cluster_for_pkg),
                    'data_source': 'OOF' if use_oof else 'resubstitution',
                },
                # 保留 resubstitution AUC 供参考
                'train_resub': {
                    'preop': roc_auc_score(y_train, prob_train_preop),
                    'cluster': roc_auc_score(y_train, prob_train_cluster),
                    'data_source': 'resubstitution',
                },
                'external': {
                    'preop': roc_auc_score(y_ext, prob_ext_preop),
                    'cluster': roc_auc_score(y_ext, prob_ext_cluster),
                    'data_source': 'direct_prediction',
                }
            },

            'metadata': {
                'timestamp': timestamp,
                'preop_pkl_source': cfg.PREOP_MODEL_PKL,
                'description': 'NRI/IDI 比较数据包：原术前模型 vs 裂孔聚类模型',
                # 明确标注训练集数据来源
                'train_data_source': 'OOF' if use_oof else 'resubstitution',
                'note': '训练集 prob_train / risk_strat_train / auc_comparison.train '
                        '均与 NRI 计算使用相同数据源（OOF 优先）；'
                        'resubstitution 版本保存在 _resub 后缀字段中备查。',
            }
        }

        nri_pkl = f"{cfg.OUTPUT_DIR}/NRI_BreakCluster_Comparison_Data_{timestamp}.pkl"
        joblib.dump(nri_comparison_package, nri_pkl)
        print(f"  ✓ NRI 比较数据包: {nri_pkl}")
        print(f"    训练集数据源: {'OOF' if use_oof else 'resubstitution'}")

        # 同时保存 latest 版本
        joblib.dump(nri_comparison_package, f"{cfg.OUTPUT_DIR}/NRI_BreakCluster_Comparison_Data_latest.pkl")
        joblib.dump(sensitivity_package, f"{cfg.OUTPUT_DIR}/Sensitivity_BreakCluster_Package_latest.pkl")

    # ---- 摘要 ----
    print(f"\n{'='*80}")
    print("✅ 敏感性分析（裂孔聚类）完成！")
    print(f"{'='*80}")
    print(f"\n  📋 分析摘要:")
    print(f"    替换方案: {cfg.BREAK_NUMERIC_FEATURES + cfg.BREAK_CATEGORICAL_FEATURES}")
    print(f"             → {cfg.BREAK_CLUSTER_COL} (K={cluster_model['n_clusters']})")
    print(f"    最佳模型: {best_name_cluster}")
    print(f"    最优阈值: {threshold_cluster:.4f}")
    if nri_available:
        print(f"    外部集 AUC 变化: {auc_ext_preop:.4f} → {auc_ext_cluster:.4f} (Δ={auc_ext_cluster-auc_ext_preop:+.4f})")

    return sensitivity_package


# ============================================================
# 18. 主流程
# ============================================================
def main(cfg=None):

    cfg = cfg or Config()
    _ensure_dirs()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    print("\n" + "=" * 80)
    print("  视网膜裂孔表型聚类 + 敏感性分析 + 生存分析")
    print("=" * 80)

    # ==========================
    # Part A: 聚类分析
    # ==========================
    print("\n" + "▶" * 30 + "  PART A: 聚类分析  " + "◀" * 30)

    # A1. 数据加载
    print("\n[A1] 数据加载与特征筛选")
    print("-" * 70)
    df_raw = pd.read_csv(cfg.TRAIN_DATA, index_col=cfg.INDEX_COL)
    print(f"✅ 内部数据集加载成功: {len(df_raw)} 样本, {len(df_raw.columns)} 特征")
    print(f"   原始缺失值总数: {df_raw.isnull().sum().sum()}")
    tear_features = select_tear_features(df_raw)
    if not tear_features:
        print("❌ 错误: 未找到裂孔相关特征，请检查特征命名")
        return None

    # A2. 预处理
    print("\n[A2] 预处理")
    print("-" * 70)
    X, X_imputed, cat_indices, imputer_cat, imputer_cont, scaler = \
        preprocess_features(df_raw, tear_features)

    # 保留原始填补后数据 (用于画像)
    df_ml = df_raw.copy()
    df_ml[X_imputed.columns.tolist()] = X_imputed.values

    print(f"\n✅ 预处理完成:")
    print(f"   • 连续变量: 中位数填补 + Z-score")
    print(f"   • 分类变量: 众数填补, 保持原值")
    print(f"   • 聚类方法: K-Prototypes (混合距离)")
    print(f"   • 聚类特征矩阵: {X.shape}")

    # 获取结局变量 (用于按风险排序)
    y_event = df_raw[cfg.TARGET_COL] if cfg.TARGET_COL in df_raw.columns else None

    # A3. 树状图
    print("\n[A3] 层次聚类树状图 (Dendrogram)")
    print("-" * 70)
    plot_dendrogram(X, cat_indices)

    # A4. K 值评估
    print("\n[A4] 最佳聚类数评估")
    print("-" * 70)
    best_sil_k, best_ch_k = evaluate_k(X, cat_indices, cfg.K_RANGE, gamma=cfg.GAMMA)

    k_final = cfg.K_FINAL
    if k_final is None:
        k_final = best_sil_k if cfg.AUTO_SELECT_K else int(input(f"请输入最终K值 (推荐: {best_sil_k}): "))
    else:
        print(f"\n🏆 用户指定 K = {k_final}")

    # A5. 聚类执行
    print("\n[A5] 执行最终聚类")
    print("-" * 70)
    kmeans_final, cluster_labels, sorted_clusters = run_clustering(
        X, cat_indices, k_final, y_event=y_event, gamma=cfg.GAMMA)

    # A6. Bootstrap 稳定性
    print("\n[A6] 聚类稳定性评估")
    print("-" * 70)
    mean_ari, aris = bootstrap_stability(X, cat_indices, k_final, cfg.N_BOOTSTRAP,
                                          gamma=cfg.GAMMA)

    # 绘制ARI分布
    plt.figure(figsize=(10, 6))
    plt.hist(aris, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(mean_ari, color='red', ls='--', lw=2, label=f'Mean ARI = {mean_ari:.3f}')
    plt.xlabel('Adjusted Rand Index', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Bootstrap Clustering Stability (100 iterations)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    save_publication_fig("Figure_Stability_ARI.png")
    plt.show()

    # A7. t-SNE
    print("\n[A7] t-SNE 降维可视化")
    print("-" * 70)
    plot_tsne(X, cluster_labels, k_final, cat_indices)

    # A8. 热图
    print("\n[A8] 聚类中心特征热图")
    print("-" * 70)
    plot_cluster_heatmap(kmeans_final, X, tear_features, k_final, cat_indices,
                         sorted_clusters, cluster_labels=cluster_labels)

    # A9. 临床画像
    print("\n[A9] 生成临床画像表")
    print("-" * 70)
    df_profile_src = df_ml
    if cfg.CLINICAL_DATA and os.path.exists(cfg.CLINICAL_DATA):
        df_profile_src = pd.read_csv(cfg.CLINICAL_DATA, index_col=cfg.INDEX_COL).iloc[:len(X)]
    generate_clinical_profile(cluster_labels, df_profile_src, tear_features)

    # A10. KM 生存曲线
    print("\n[A10] Kaplan-Meier 生存分析")
    print("-" * 70)
    km_result = None
    if LIFELINES_AVAILABLE:
        try:
            df_km = df_ml.copy()
            if cfg.CLINICAL_DATA and os.path.exists(cfg.CLINICAL_DATA):
                df_km = pd.read_csv(cfg.CLINICAL_DATA, index_col=cfg.INDEX_COL)
            df_km['Cluster_ID'] = cluster_labels.values if hasattr(cluster_labels, 'values') else cluster_labels
            km_result = plot_km_survival(df_km)
        except Exception as e:
            print(f"⚠️ 生存曲线绘制失败: {str(e)}")


    # A12. 外部验证集
    print("\n[A12] 外部验证集聚类应用")
    print("-" * 70)
    df_ext = assign_external_clusters(
        kmeans_final, X, cfg.EXTERNAL_DATA, tear_features,
        cat_indices, imputer_cat, imputer_cont, scaler,
        sorted_clusters=sorted_clusters)

    # A13. 外部集 KM
    if LIFELINES_AVAILABLE and df_ext is not None and cfg.TARGET_COL in df_ext.columns:
        print("\n[A13] 外部验证集 KM 生存曲线")
        print("-" * 70)
        try:
            plot_km_survival(df_ext, fig_name='Figure_KM_Survival_External.png')
        except Exception as e:
            print(f"⚠️ 外部集生存曲线绘制失败: {str(e)}")

    # A14. 保存内部集带标签
    print("\n[A14] 保存聚类结果")
    print("-" * 70)
    df_ml['Cluster_ID'] = cluster_labels

    # 调整列顺序 (Recurrence -> Cluster_ID -> 其他)
    cols = df_ml.columns.tolist()
    if 'Cluster_ID' in cols:
        cols.remove('Cluster_ID')
    if 'Recurrence' in cols:
        rec_idx = cols.index('Recurrence')
        new_cols = cols[:rec_idx+1] + ['Cluster_ID'] + cols[rec_idx+1:]
        df_ml = df_ml[new_cols]

    out_csv = 'Data_for_ML_Model_with_TearPhenotype.csv'
    df_ml.to_csv(out_csv, encoding='utf-8-sig')
    print(f"✅ 内部数据集已保存: {out_csv}")

    # 聚类报告
    sil = silhouette_score_gower(X, cluster_labels, cat_indices)
    cont_indices_report = [i for i in range(X.shape[1]) if i not in cat_indices]
    ch = calinski_harabasz_score(np.array(X)[:, cont_indices_report], cluster_labels) if cont_indices_report else 0

    cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
    report_lines = [
        "=" * 80,
        "聚类分析完整报告",
        "Clustering Analysis Summary Report",
        "=" * 80,
        f"\n数据集信息:",
        f"  • 内部样本数: {len(df_ml)}",
        f"  • 裂孔特征数: {len(tear_features)}",
        f"  • 最终聚类数: {k_final}",
        f"\n评估指标:",
        f"  • Silhouette Score (Gower): {sil:.3f}",
        f"  • Calinski-Harabasz Index (continuous only): {ch:.1f}",
        f"  • Bootstrap Stability (Mean ARI): {mean_ari:.3f}",
        f"  • 聚类方法: K-Prototypes (混合距离: 连续=欧氏, 分类=汉明)",
        f"\n样本分布:",
    ]
    for cluster_id, count in cluster_dist.items():
        pct = count / len(cluster_labels) * 100
        report_lines.append(f"  • Cluster {cluster_id}: {count:4d} ({pct:5.1f}%)")
    report_lines.extend([
        f"\n生成文件:",
        f"  • 图片: {FIG_DIR}/",
        f"  • 表格: {TABLE_DIR}/",
        f"  • 数据: {out_csv}",
        "\n" + "=" * 80
    ])
    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(TABLE_DIR, 'Clustering_Analysis_Report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n✅ 完整报告已保存: {report_path}")

    # ==========================
    # Part B: 敏感性分析
    # ==========================
    print("\n" + "▶" * 30 + "  PART B: 敏感性分析  " + "◀" * 30)
    sens_result = run_sensitivity_analysis(cfg, timestamp)

    # ==========================
    # 完成
    # ==========================
    print("\n" + "=" * 80)
    print("🎉 全部分析完成!")
    print("=" * 80)

    return {
        'kprot_model': kmeans_final,
        'cluster_labels': cluster_labels,
        'X': X,
        'cat_indices': cat_indices,
        'tear_features': tear_features,
        'k_final': k_final,
        'silhouette_score': sil,
        'stability_ari': mean_ari,
        'sensitivity_package': sens_result,
    }


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":


    results = main()
    if results:
        print("\n💡 聚类结果已保存到 results 变量，可用于后续分析")