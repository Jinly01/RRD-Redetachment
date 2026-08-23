import os
import glob
import joblib
import warnings

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

from scipy.stats import chi2, chi2_contingency
from scipy.special import expit
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, matthews_corrcoef,
)

warnings.filterwarnings('ignore')

# SHAP 为可选依赖（缺失时自动跳过第五部分）
try:
    import shap
    HAS_SHAP = True
except Exception as _e:                                    # pragma: no cover
    HAS_SHAP = False
    print(f"⚠️ 未能导入 shap（{_e}），将跳过 SHAP 模块。")


# ==========================================
# 路径与时间戳（如需切换数据集，改这里即可）
# ==========================================
DATA_PATH = "./model_results"
TIMESTAMP = '20260809_0008'
DPI = 600

# 敏感性分析（术中模型）对应的时间戳
TIMESTAMP_INTRA   = '20260809_0013'

os.makedirs('figures', exist_ok=True)


# ==========================================
# 全局绘图风格（SCI 顶刊）
# ==========================================
def set_sci_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.title_fontsize'] = 9

    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.major.size'] = 3.5
    plt.rcParams['ytick.major.size'] = 3.5

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = '#999999'
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams['legend.borderpad'] = 0.4
    plt.rcParams['legend.labelspacing'] = 0.3
    plt.rcParams['legend.handlelength'] = 1.8
    plt.rcParams['legend.handletextpad'] = 0.5

    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05

set_sci_style()


def sci_legend(ax, **kwargs):
    """统一风格图例（带细框线）"""
    defaults = dict(
        frameon=True, edgecolor='#999999', fancybox=False,
        framealpha=1.0, borderpad=0.4, labelspacing=0.3,
        handlelength=1.8, handletextpad=0.5
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)


# ==========================================
# 配色（沿用原脚本 COLORS_LIST 索引）
#   蓝色=内部验证，红色=外部验证
# ==========================================
COLORS_LIST = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148",
]
C_INT      = COLORS_LIST[3]   # #3C5488 深蓝 — 内部验证
C_EXT      = COLORS_LIST[0]   # #E64B35 红色 — 外部验证
C_OOF_LINE = COLORS_LIST[1]   # #4DBBD5 青色（保留以兼容）
C_EXT_LINE = COLORS_LIST[7]   # #DC0000 深红（保留以兼容）

# SHAP / Figure 2 等模块统一引用的队列配色别名
COLOR_INTERNAL = C_INT
COLOR_EXTERNAL = C_EXT

# 混淆矩阵专用连续色标（白→深蓝 / 白→红）
cmap_int = LinearSegmentedColormap.from_list('cmap_int', ['#FFFFFF', C_INT])
cmap_ext = LinearSegmentedColormap.from_list('cmap_ext', ['#FFFFFF', C_EXT])


# ==========================================================
# 零、变量名 → JAMA-style 展示标签映射
#   （直接沿用建模脚本中的 LABEL_MAP，保证正文/表/图标签完全一致）
# ==========================================================
LABEL_MAP = {
    # ---- 结局 / 时间变量 ----
    'Recurrence':              'Retinal redetachment',
    'Follow_up_Time':          'Follow-up time (mo)',
    'SO_Removal_Time':         'Time to silicone oil removal (mo)',
    'status_cr':               'Competing-risk status',

    # ---- 术前基线特征 ----
    'Diabetes':                'Diabetes',
    'AL':                      'Axial length (mm)',
    'BCVA_Pre':                'Preoperative BCVA (logMAR)',
    'Lens_Status_Pre':         'Preoperative lens status',
    'VH':                      'Vitreous hemorrhage',
    'Macular_status':          'Macula-off detachment',
    'Symptom_Duration':        'Duration of symptoms (d)',
    'PVR_Grade_Pre':           'Preoperative PVR grade',
    'Choroidal_Detachment':    'Choroidal detachment',
    'RD_Extent':               'RD extent (quadrants)',

    # ---- 裂孔特征 ----
    'number_of_breaks':        'Total retinal breaks (No.)',
    'Largest_Break_Diameter':  'Largest break size (DD)',
    'Break_Loc_Inferior':      'Inferior break',
    'Macular_Hole':            'Macular hole',
    'Lattice_Degeneration':    'Lattice degeneration',
    'Atrophic_Holes':          'Atrophic hole',

    # ---- 术中变量 ----
    'Surgery_Duration':        'Operative duration (min)',
    'PFCL':                    'Use of perfluorocarbon liquid',
    'Phacovitrectomy':         'Combined phacovitrectomy',

    # ---- SO removal 时变量 ----
    'BCVA_SOR':                'BCVA at silicone oil removal (logMAR)',
    'SO_Emulsification':       'Silicone oil emulsification',
    'Concurrent_Phaco_SOR':    'Phacoemulsification at silicone oil removal',
    'ERM_SOR':                 'Epiretinal membrane at silicone oil removal',
    'PVR_SOR':                 'PVR at silicone oil removal',

}

# ColumnTransformer 生成的特征名前缀（get_feature_names_out）
_TRANS_PREFIXES = ('log__', 'num__', 'cat__', 'remainder__', 'pipeline-1__',
                   'pipeline-2__', 'pipeline-3__', 'onehot__', 'scaler__')


def _strip_prefix(name):
    """去掉 ColumnTransformer 的 'num__' / 'cat__' 等前缀"""
    s = str(name)
    for p in _TRANS_PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    if '__' in s:                       # 兜底：任何 xxx__ 前缀
        return s.split('__', 1)[1]
    return s


def rename_feature(name):
    """
    将代码变量名映射为 SCI 展示标签，未在映射表中的保持原样。
    额外处理两种 sklearn 产物：
      · 'num__AL'              → 'Axial length (mm)'
      · 'cat__PVR_Grade_Pre_C2'→ 'Preoperative PVR grade: C2'（独热编码水平）
    """
    raw = str(name)
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]

    base = _strip_prefix(raw)
    if base in LABEL_MAP:
        return LABEL_MAP[base]

    # 独热编码：从最长可匹配的前缀切分出「变量_水平」
    for key in sorted(LABEL_MAP, key=len, reverse=True):
        if base.startswith(key + '_'):
            level = base[len(key) + 1:]
            return f"{LABEL_MAP[key]}: {level}"

    return base


def rename_feature_list(names):
    """批量重命名特征列表"""
    return [rename_feature(n) for n in names]


def rename_df_feature_col(df, col='Feature'):
    """就地重命名 DataFrame 中的 Feature 列"""
    if col in df.columns:
        df[col] = df[col].map(rename_feature)
    return df


def rename_df_columns(df):
    """重命名 DataFrame 的列名（用于原始特征矩阵）"""
    return df.rename(columns={c: rename_feature(c) for c in df.columns})


# ==========================================================
# 一、辅助函数（校准曲线 / DCA）
#     ❌ 已删除 intercept_only_recalibration() 与 _safe_logit()
# ==========================================================
def get_cali_stats(y_true, y_prob):
    """计算校准斜率和截距（仅作为报告指标，不做任何概率修正）"""
    p_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    logit_p = np.log(p_clipped / (1 - p_clipped))
    X = logit_p.reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(X, y_true)
    return lr.coef_[0][0], lr.intercept_[0]


def logistic_calibration_curve(y_true, y_prob, n_points=200):
    """
    参数化校准曲线: 用 logistic recalibration 拟合
    比 LOWESS 稳健得多，不受不平衡影响
    返回平滑的校准曲线 (理论拟合线)
    """
    eps = 1e-15
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped))

    # 拟合 logistic recalibration: P(Y=1) = sigmoid(a * logit(p) + b)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)

    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    # 生成平滑曲线
    x_range = np.linspace(y_prob.min(), min(y_prob.max(), 0.95), n_points)
    x_logit = np.log(np.clip(x_range, eps, 1-eps) / (1 - np.clip(x_range, eps, 1-eps)))
    y_fit = expit(slope * x_logit + intercept)

    return x_range, y_fit, slope, intercept


def adaptive_calibration_bins(y_true, y_prob, min_samples=20, max_bins=10):
    """
    自适应等频分bin
    - 保证每个bin最少 min_samples 个样本
    - 使用 Wilson 置信区间 (适合小样本和极端比例)
    - 返回 bin 统计信息
    """
    n = len(y_true)
    n_bins = min(max_bins, max(5, n // (min_samples * 2)))

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(y_prob, quantiles))

    results = {'pred': [], 'true': [], 'n': [], 'n_pos': [],
               'ci_low': [], 'ci_up': [], 'pred_range': []}

    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])

        n_in = mask.sum()
        if n_in < max(5, min_samples // 2):
            continue

        p_hat = y_true[mask].mean()
        mean_pred = y_prob[mask].mean()

        z = 1.96
        denom = 1 + z**2 / n_in
        center = (p_hat + z**2 / (2 * n_in)) / denom
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_in)) / n_in) / denom

        results['pred'].append(mean_pred)
        results['true'].append(p_hat)
        results['n'].append(n_in)
        results['n_pos'].append(int(y_true[mask].sum()))
        results['ci_low'].append(max(0, center - spread))
        results['ci_up'].append(min(1, center + spread))
        results['pred_range'].append((bin_edges[i], bin_edges[min(i+1, len(bin_edges)-1)]))

    for k in results:
        results[k] = np.array(results[k]) if k != 'pred_range' else results[k]

    return results


def bootstrap_logistic_cal_ci(y_true, y_prob, n_bootstrap=1000, ci=95):
    """Bootstrap 置信带 (基于 logistic recalibration 拟合线)"""
    eps = 1e-15
    x_range = np.linspace(max(y_prob.min(), 0.001), min(y_prob.max(), 0.95), 200)
    x_logit = np.log(np.clip(x_range, eps, 1-eps) / (1 - np.clip(x_range, eps, 1-eps)))

    boot_curves = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_b, p_b = y_true[idx], y_prob[idx]

        try:
            logit_b = np.log(np.clip(p_b, eps, 1-eps) / (1 - np.clip(p_b, eps, 1-eps)))
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500)
            lr.fit(logit_b.reshape(-1, 1), y_b)
            y_fit = expit(lr.coef_[0][0] * x_logit + lr.intercept_[0])
            boot_curves.append(y_fit)
        except Exception:
            continue

    boot_curves = np.array(boot_curves)
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_curves, alpha, axis=0)
    upper = np.percentile(boot_curves, 100 - alpha, axis=0)

    return x_range, lower, upper


def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """Hosmer-Lemeshow 拟合优度检验 (标准十分位数法)"""
    sorted_idx = np.argsort(y_prob)
    groups = np.array_split(sorted_idx, n_groups)

    chi2_stat = 0
    for g in groups:
        if len(g) == 0:
            continue
        obs_pos = y_true[g].sum()
        obs_neg = len(g) - obs_pos
        exp_pos = y_prob[g].sum()
        exp_neg = len(g) - exp_pos
        if exp_pos > 0:
            chi2_stat += (obs_pos - exp_pos) ** 2 / exp_pos
        if exp_neg > 0:
            chi2_stat += (obs_neg - exp_neg) ** 2 / exp_neg

    p_value = 1 - chi2.cdf(chi2_stat, n_groups - 2)
    return chi2_stat, p_value


def plot_calibration_publication_v2(
    y_int, prob_int, y_ext, prob_ext,
    save_path='figures/Calibration_Publication_v2.png',
    dpi=600, n_bootstrap=1000
):
    """出版级校准曲线（直接使用原始预测概率，不做任何截距校准）"""

    # ==================== 布局: 主图 + 底部spike图 ====================
    fig = plt.figure(figsize=(3.5, 3.5))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[5.5, 1], hspace=0.05,
        left=0.12, right=0.95, top=0.93, bottom=0.08
    )

    ax_main = fig.add_subplot(gs[0])
    ax_spike = fig.add_subplot(gs[1], sharex=ax_main)

    # ==================== 数据集配置  ====================
    datasets = [
        {'y': y_int, 'prob': prob_int, 'label': 'Internal',
         'color': C_INT, 'fill': C_INT, 'marker': 'o'},
        {'y': y_ext, 'prob': prob_ext, 'label': 'External',
         'color': C_EXT, 'fill': C_EXT, 'marker': 's'},
    ]

    print("\n" + "="*70)
    print("  Publication-Quality Calibration Plot (v2 — legend stats)")
    print("="*70)

    for ds in datasets:
        y, p = ds['y'], ds['prob']

        # --- A. 统计量 ---
        slope, intercept = get_cali_stats(y, p)
        hl_stat, hl_pval = hosmer_lemeshow_test(y, p, n_groups=10)
        brier = brier_score_loss(y, p)
        eo_ratio = p.sum() / y.sum() if y.sum() > 0 else np.inf

        sig = '***' if hl_pval < 0.001 else ('**' if hl_pval < 0.01 else ('*' if hl_pval < 0.05 else ''))

        print(f"\n  {ds['label']}:")
        print(f"    N={len(y)}, Events={int(y.sum())} ({y.mean():.1%})")
        print(f"    Slope={slope:.3f}, Intercept={intercept:.3f}")
        print(f"    Brier={brier:.4f}, E/O={eo_ratio:.3f}")
        print(f"    H-L: χ²={hl_stat:.2f}, p={hl_pval:.4f}")

        # --- B. 构造图例 ---
        legend_label = (
            f"{ds['label']} (Slope = {slope:.2f}, Brier = {brier:.3f})"
        )

        # --- C. 自适应分bin ---
        min_per_bin = max(15, int(len(y) * 0.04))
        bins = adaptive_calibration_bins(y, p, min_samples=min_per_bin, max_bins=10)

        # 点大小编码样本量
        size_min, size_max = 15, 45
        if len(bins['n']) > 0 and bins['n'].max() > bins['n'].min():
            sizes = size_min + (size_max - size_min) * (bins['n'] - bins['n'].min()) / (bins['n'].max() - bins['n'].min())
        else:
            sizes = np.full(len(bins['n']), (size_min + size_max) / 2)

        # 误差线
        ax_main.errorbar(
            bins['pred'], bins['true'],
            yerr=[bins['true'] - bins['ci_low'], bins['ci_up'] - bins['true']],
            fmt='none', ecolor=ds['color'], elinewidth=0.6,
            capsize=1.5, capthick=0.6, alpha=0.45, zorder=3
        )

        # 散点
        ax_main.scatter(
            bins['pred'], bins['true'], s=sizes, marker=ds['marker'],
            facecolors=ds['color'], edgecolors=ds['color'],
            linewidths=0.7, alpha=0.85, zorder=4
        )

        # --- D. Logistic recalibration 拟合曲线 (label 含统计信息) ---
        x_fit, y_fit, _, _ = logistic_calibration_curve(y, p)
        ax_main.plot(x_fit, y_fit, color=ds['color'], lw=1.5,
                     alpha=0.8, zorder=5, label=legend_label)

        # --- E. Bootstrap CI 带 ---
        print(f"    Computing {n_bootstrap} bootstrap CIs...")
        x_ci, ci_low, ci_up = bootstrap_logistic_cal_ci(y, p, n_bootstrap=n_bootstrap)
        ax_main.fill_between(x_ci, ci_low, ci_up, color=ds['color'],
                             alpha=0.15, zorder=1)

    # --- F. 理想校准线 ---
    ax_main.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--',
                 zorder=2, label='Ideal')

    # ==================== 主图美化 ====================
    ax_main.set_xlabel('Predicted probability')
    ax_main.set_ylabel('Observed proportion')
    ax_main.set_xlim([-0.02, 1.02])
    ax_main.set_ylim([-0.02, 1.02])
    ax_main.set_aspect('equal')
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # ---- 图例 (含全部统计信息, 放在 lower right) ----
    sci_legend(ax_main, loc='lower right', fontsize=7)

    # ==================== 底部 Spike Histogram ====================
    spike_bins = np.linspace(0, 1, 41)
    bin_width = spike_bins[1] - spike_bins[0]

    for ds in datasets:
        y, p = ds['y'], ds['prob']
        offset = 0.002 if ds['marker'] == 'o' else -0.002

        counts_pos, _ = np.histogram(p[y == 1], bins=spike_bins)
        centers = (spike_bins[:-1] + spike_bins[1:]) / 2 + offset
        ax_spike.bar(centers, counts_pos, width=bin_width * 0.45,
                     color=ds['color'], alpha=0.6, edgecolor='none')

        counts_neg, _ = np.histogram(p[y == 0], bins=spike_bins)
        neg_scale = max(counts_pos.max(), 1) / max(counts_neg.max(), 1) * 0.8
        ax_spike.bar(centers, -counts_neg * neg_scale, width=bin_width * 0.45,
                     color=ds['color'], alpha=0.25, edgecolor='none')

    ax_spike.axhline(y=0, color='#888888', linewidth=0.6)
    ax_spike.set_xlabel('Predicted probability')
    ax_spike.set_xlim([-0.02, 1.02])
    ax_spike.spines['top'].set_visible(False)
    ax_spike.spines['right'].set_visible(False)
    ax_spike.set_yticks([])

    ax_spike.text(0.01, 0.92, 'Events', transform=ax_spike.transAxes,
                  fontsize=6.5, color='#555555', va='top', fontstyle='italic')
    ax_spike.text(0.01, 0.08, 'Non-events', transform=ax_spike.transAxes,
                  fontsize=6.5, color='#555555', va='bottom', fontstyle='italic')

    # ==================== 保存 ====================
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\n  ✓ Saved: {save_path} / .pdf")
    plt.close()
    return fig


def calculate_net_benefit(y_true, y_pred_prob, threshold):
    """计算净获益"""
    if threshold >= 1.0:
        return 0

    n = len(y_true)
    tp = np.sum((y_pred_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_pred_prob >= threshold) & (y_true == 0))

    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit


def bootstrap_net_benefit(y_true, y_pred_prob, thresholds, n_bootstrap=500):
    """使用 Bootstrap 计算净获益的置信区间"""
    from sklearn.utils import resample

    all_nbs = []
    for _ in range(n_bootstrap):
        indices = resample(np.arange(len(y_true)), random_state=None)
        y_boot = y_true[indices]
        p_boot = y_pred_prob[indices]

        nbs = [calculate_net_benefit(y_boot, p_boot, t) for t in thresholds]
        all_nbs.append(nbs)

    all_nbs = np.array(all_nbs)
    lower = np.percentile(all_nbs, 2.5, axis=0)
    upper = np.percentile(all_nbs, 97.5, axis=0)
    mean = np.mean(all_nbs, axis=0)

    return mean, lower, upper


def calculate_n_high_risk(y_pred_prob, threshold, n_total):
    """计算高风险患者数量（每 1000 人）"""
    return np.sum(y_pred_prob >= threshold) / n_total * 1000


# ==========================================================
# 二、数据加载
#   ✅ 完全对齐建模脚本 Model_Package_<TS>.pkl 的实际字段：
#      best_model / best_model_name / optimal_threshold / preprocessor /
#      X_train / y_train / X_external / y_external /
#      predictions{train_probs, external_probs, internal_oof_probs} /
#      cv_results / final_metrics / bootstrap_analysis / risk_stratification
#   ⚠️ 建模脚本已明确 'calibrated_model': None —— 本脚本同样不做任何校准。
# ==========================================================
print("=" * 60)
print("📂 加载模型与预测数据 ...")
print("=" * 60)

pkl_path = f"{DATA_PATH}/Model_Package_{TIMESTAMP}.pkl"
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"❌ 找不到模型包: {pkl_path}")
full_package = joblib.load(pkl_path)

# ---------- 2.1 模型与阈值 ----------
best_model        = full_package['best_model']
best_model_name   = full_package['best_model_name']
optimal_threshold = full_package['optimal_threshold']      # 原始概率尺度上的锁定阈值
clinical_threshold = optimal_threshold
preprocessor      = full_package.get('preprocessor', None)

assert full_package.get('calibrated_model', None) is None, \
    "⚠️ 模型包中存在 calibrated_model；本脚本按未校准流程设计，请确认。"

# ---------- 2.2 原始（未标准化）特征表 ----------
#   SHAP 眼轴长依赖图需要 mm 尺度的 AL，必须来自这里
X_TRAIN_RAW = full_package.get('X_train',    None)
Y_TRAIN     = full_package.get('y_train',    None)
X_EXT_RAW   = full_package.get('X_external', None)
Y_EXT_PKG   = full_package.get('y_external', None)
if X_TRAIN_RAW is not None:
    print(f"  ✓ 原始训练集特征表: {X_TRAIN_RAW.shape}")
    print(f"  ✓ 原始外部集特征表: {X_EXT_RAW.shape}")
else:
    print("  ⚠️ 模型包中无原始特征表，SHAP 依赖图将退回标准化尺度")

# ---------- 2.3 真实标签（CSV）与预测概率（pickle 原始浮点为准）----------
df_int = pd.read_csv(f"{DATA_PATH}/Internal_Val_OOF_Preds_{best_model_name}_{TIMESTAMP}.csv")
df_ext = pd.read_csv(f"{DATA_PATH}/External_Val_Preds_{best_model_name}_{TIMESTAMP}.csv")
y_int = df_int['True_Label'].values
y_ext = df_ext['True_Label'].values

preds_dict = full_package['predictions']
prob_int = np.asarray(preds_dict['internal_oof_probs'], dtype=float)   # p_dev_oof
prob_ext = np.asarray(preds_dict['external_probs'],     dtype=float)   # p_external_locked

assert len(df_int) == len(prob_int), "❌ 内部验证长度不一致"
assert len(df_ext) == len(prob_ext), "❌ 外部验证长度不一致"

# 一致性核查（CSV 浮点截断导致的微小差异可接受）
_ok_int = np.allclose(df_int['Pred_Prob'].values, prob_int, rtol=1e-4, atol=1e-5)
_ok_ext = np.allclose(df_ext['Pred_Prob'].values, prob_ext, rtol=1e-4, atol=1e-5)
if _ok_int and _ok_ext:
    print("  ✅ CSV 与 Pickle 预测概率一致")
else:
    print("  ⚠️ CSV 与 Pickle 概率存在微小差异（浮点截断），以 Pickle 为准")

# ---------- 2.4 外部 Bootstrap（曲线置信带 + 指标 95% CI）----------
_boot_pkl = f"{DATA_PATH}/Bootstrap_Predictions_{TIMESTAMP}.pkl"
if os.path.exists(_boot_pkl):
    bootstrap_predictions = joblib.load(_boot_pkl)
else:                                   # 回退：模型包内也存了一份
    bootstrap_predictions = full_package['bootstrap_analysis']['predictions']
    print("  ℹ️ 未找到 Bootstrap_Predictions pkl，改用模型包内嵌的 bootstrap 预测")

#   混淆矩阵指标表引用的外部 95% CI（Sensitivity / Specificity / PPV / NPV /
#   F1 / MCC / G_mean / Balanced_Acc），优先用 CSV，其次用模型包内的 summary
df_bootstrap_ext = None
_boot_csv = f"{DATA_PATH}/External_Val_Bootstrap_1000_{TIMESTAMP}.csv"
try:
    if os.path.exists(_boot_csv):
        df_bootstrap_ext = pd.read_csv(_boot_csv, index_col=0)
    elif 'bootstrap_analysis' in full_package:
        df_bootstrap_ext = pd.DataFrame(full_package['bootstrap_analysis']['summary']).T
    if df_bootstrap_ext is not None:
        print(f"  ✓ 外部 Bootstrap 指标表: {len(df_bootstrap_ext)} 个指标（含 95% CI）")
except Exception as e:
    print(f"  ⚠️ 外部 Bootstrap 指标表加载失败: {e}")

# ---------- 2.5 阈值方法（仅供打印溯源）----------
threshold_method = 'Unknown'
if 'cv_results' in full_package and best_model_name in full_package['cv_results']:
    threshold_method = full_package['cv_results'][best_model_name].get(
        'threshold_method', 'Unknown')

prev_int = y_int.mean()
prev_ext = y_ext.mean()

print(f"  ✓ 最佳模型: {best_model_name}")
print(f"  ✓ 内部验证 (OOF): {len(prob_int)} 例，结局率 {prev_int:.1%}")
print(f"  ✓ 外部验证:        {len(prob_ext)} 例，结局率 {prev_ext:.1%}")
print(f"  ✓ 锁定阈值 (原始尺度): {clinical_threshold:.4f}（方法: {threshold_method}）")
print(f"  ✓ 外部 bootstrap 重采样: {len(bootstrap_predictions)} 次")
print("  ℹ️ 本脚本已删除截距校准，所有分析均基于原始预测概率。")


# ==========================================================
# 三、Bootstrap 曲线与置信带
#   内部：对 (y_int, prob_int) 做患者层面 bootstrap 重采样（OOF band）
#   外部：直接用预存的 bootstrap_predictions（locked external band）
#   —— 不做任何模型重拟合
# ==========================================================
print("\n" + "=" * 60)
print("📈 计算 Bootstrap 曲线与 95% 置信带 ...")
print("=" * 60)

mean_fpr    = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
N_BOOT = 1000                      # 内外部保持一致
np.random.seed(42)                 # 复现性

# --- 点估计（确定性，决定主线与图例数字）---
oof_auc = roc_auc_score(y_int, prob_int)
oof_ap  = average_precision_score(y_int, prob_int)
ext_auc = roc_auc_score(y_ext, prob_ext)
ext_ap  = average_precision_score(y_ext, prob_ext)
print(f"  内部 OOF : AUROC = {oof_auc:.3f} | AP = {oof_ap:.3f}")
print(f"  外部     : AUROC = {ext_auc:.3f} | AP = {ext_ap:.3f}")

# --- 内部 OOF bootstrap 重采样 ---
boot_tprs_oof, boot_precs_oof = [], []
boot_auc_oof,  boot_ap_oof    = [], []        # ← 新增：AUROC / AP 的 bootstrap 分布
for _ in range(N_BOOT):
    idx = np.random.choice(len(y_int), size=len(y_int), replace=True)
    yb, pb = y_int[idx], prob_int[idx]
    if len(np.unique(yb)) < 2:
        continue
    fpr_b, tpr_b, _ = roc_curve(yb, pb)
    it = np.interp(mean_fpr, fpr_b, tpr_b); it[0] = 0.0
    boot_tprs_oof.append(it)
    prec_b, rec_b, _ = precision_recall_curve(yb, pb)
    order = np.argsort(rec_b)
    boot_precs_oof.append(np.interp(mean_recall, rec_b[order], prec_b[order]))
    boot_auc_oof.append(roc_auc_score(yb, pb))                # ← 新增
    boot_ap_oof.append(average_precision_score(yb, pb))       # ← 新增

boot_tprs_oof  = np.array(boot_tprs_oof)
boot_precs_oof = np.array(boot_precs_oof)
mean_tpr_oof  = boot_tprs_oof.mean(axis=0)
std_tpr_oof   = boot_tprs_oof.std(axis=0)
upper_tpr_oof = np.minimum(mean_tpr_oof + 1.96 * std_tpr_oof, 1)
lower_tpr_oof = np.maximum(mean_tpr_oof - 1.96 * std_tpr_oof, 0)
mean_prec_oof  = boot_precs_oof.mean(axis=0)
std_prec_oof   = boot_precs_oof.std(axis=0)
upper_prec_oof = np.minimum(mean_prec_oof + 1.96 * std_prec_oof, 1)
lower_prec_oof = np.maximum(mean_prec_oof - 1.96 * std_prec_oof, 0)

# --- 外部 bootstrap（预存重采样预测）---
tprs_external, precs_external = [], []
boot_auc_ext,  boot_ap_ext    = [], []        # ← 新增：AUROC / AP 的 bootstrap 分布
for boot_pred in bootstrap_predictions:
    y_true_b = boot_pred['true_labels']
    y_pred_b = boot_pred['pred_probs']
    if len(np.unique(y_true_b)) < 2:
        continue
    fpr_e, tpr_e, _ = roc_curve(y_true_b, y_pred_b)
    it = np.interp(mean_fpr, fpr_e, tpr_e); it[0] = 0.0
    tprs_external.append(it)
    prec_e, rec_e, _ = precision_recall_curve(y_true_b, y_pred_b)
    order = np.argsort(rec_e)
    precs_external.append(np.interp(mean_recall, rec_e[order], prec_e[order]))
    boot_auc_ext.append(roc_auc_score(y_true_b, y_pred_b))            # ← 新增
    boot_ap_ext.append(average_precision_score(y_true_b, y_pred_b))   # ← 新增

tprs_external  = np.array(tprs_external)
precs_external = np.array(precs_external)
mean_tpr_ext  = tprs_external.mean(axis=0)
std_tpr_ext   = tprs_external.std(axis=0)
upper_tpr_ext = np.minimum(mean_tpr_ext + 1.96 * std_tpr_ext, 1)
lower_tpr_ext = np.maximum(mean_tpr_ext - 1.96 * std_tpr_ext, 0)
mean_prec_ext  = precs_external.mean(axis=0)
std_prec_ext   = precs_external.std(axis=0)
upper_prec_ext = np.minimum(mean_prec_ext + 1.96 * std_prec_ext, 1)
lower_prec_ext = np.maximum(mean_prec_ext - 1.96 * std_prec_ext, 0)

# --- 经验（原始）曲线：内部 OOF / 外部 locked ---
fpr_oof_orig, tpr_oof_orig, _ = roc_curve(y_int, prob_int)
fpr_ext_orig, tpr_ext_orig, _ = roc_curve(y_ext, prob_ext)
prec_oof_orig, rec_oof_orig, _ = precision_recall_curve(y_int, prob_int)
prec_ext_orig, rec_ext_orig, _ = precision_recall_curve(y_ext, prob_ext)

prevalence_int = y_int.mean()
prevalence_ext = y_ext.mean()
print(f"  ✓ band 完成（内部 n_boot={len(boot_tprs_oof)}，外部 n_boot={len(tprs_external)}）")


# --- AUROC / AP 的 95% 百分位 CI，以及 Δ(External − Internal) ---
#     用于「内部 × 外部叠加图」的图例与标题；不改变任何点估计。
_RNG_COMB = np.random.RandomState(2026)   # 独立随机流，避免扰动上游 np.random.seed(42)


def _pct_ci(samples, lo=2.5, hi=97.5):
    """bootstrap 分布的百分位 95% CI；样本过少时返回 nan（图例自动省略 CI）"""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 20:
        return np.array([np.nan, np.nan])
    return np.percentile(s, [lo, hi])


def _delta_ci(samples_int, samples_ext, n_pair=4000):
    """内部与外部是相互独立的两个队列（非配对），因此直接对两组 bootstrap
    分布做随机配对求差，得到 Δ(External − Internal) 的经验 95% CI。
    返回 (Δ 的 bootstrap 均值, [CI 下限, CI 上限])。"""
    a = np.asarray(samples_int, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(samples_ext, dtype=float); b = b[np.isfinite(b)]
    if a.size < 20 or b.size < 20:
        return np.nan, np.array([np.nan, np.nan])
    d = b[_RNG_COMB.randint(0, b.size, n_pair)] - a[_RNG_COMB.randint(0, a.size, n_pair)]
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])


auc_int_ci = _pct_ci(boot_auc_oof)
auc_ext_ci = _pct_ci(boot_auc_ext)
ap_int_ci  = _pct_ci(boot_ap_oof)
ap_ext_ci  = _pct_ci(boot_ap_ext)

delta_auc = ext_auc - oof_auc              # 点估计之差（图中显示的 Δ）
delta_ap  = ext_ap  - oof_ap
_, d_auc_ci = _delta_ci(boot_auc_oof, boot_auc_ext)
_, d_ap_ci  = _delta_ci(boot_ap_oof,  boot_ap_ext)

print(f"  内部 : AUROC = {oof_auc:.3f} ({auc_int_ci[0]:.3f}–{auc_int_ci[1]:.3f}) | "
      f"AP = {oof_ap:.3f} ({ap_int_ci[0]:.3f}–{ap_int_ci[1]:.3f})")
print(f"  外部 : AUROC = {ext_auc:.3f} ({auc_ext_ci[0]:.3f}–{auc_ext_ci[1]:.3f}) | "
      f"AP = {ext_ap:.3f} ({ap_ext_ci[0]:.3f}–{ap_ext_ci[1]:.3f})")
print(f"  Δ(Ext − Int): AUROC = {delta_auc:+.3f} ({d_auc_ci[0]:+.3f} to {d_auc_ci[1]:+.3f}) | "
      f"AP = {delta_ap:+.3f} ({d_ap_ci[0]:+.3f} to {d_ap_ci[1]:+.3f})")


# ==========================================================
# 四、正文六图 A–F
# ==========================================================
FIG_SINGLE = (3.6, 3.6)    # 单图（ROC / PR），与校准图 3.5×3.5 尺度一致

print("\n" + "=" * 60)
print("🎨 生成正文图 ...")
print("=" * 60)
print("\n[A] ROC — Internal (OOF) 单图 ...")
# ==================== 图 A：ROC Internal（单图）====================
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ax.fill_between(mean_fpr, lower_tpr_oof, upper_tpr_oof, color=C_INT, alpha=0.15,
                label='95% Bootstrap CI')
ax.plot(fpr_oof_orig, tpr_oof_orig, color=C_INT, lw=1.5,
        label=f'OOF (AUROC = {oof_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
sci_legend(ax, loc='lower right')
plt.tight_layout()
plt.savefig('figures/ROC_Internal.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_Internal.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/ROC_Internal.png / .pdf")
plt.close()

print("\n[B] ROC — External (locked) 单图 ...")
# ==================== 图 B：ROC External（单图）====================
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ax.fill_between(mean_fpr, lower_tpr_ext, upper_tpr_ext, color=C_EXT, alpha=0.15,
                label='95% Bootstrap CI')
ax.plot(fpr_ext_orig, tpr_ext_orig, color=C_EXT, lw=1.5,
        label=f'External (AUROC = {ext_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
sci_legend(ax, loc='lower right')
plt.tight_layout()
plt.savefig('figures/ROC_External.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_External.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/ROC_External.png / .pdf")
plt.close()

print("\n[C] PR — Internal (OOF) 单图 ...")
# ==================== 图 C：PR Internal（单图）====================
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ax.fill_between(mean_recall, lower_prec_oof, upper_prec_oof, color=C_INT, alpha=0.15,
                label='95% Bootstrap CI')
ax.plot(rec_oof_orig, prec_oof_orig, color=C_INT, lw=1.5,
        label=f'OOF (AP = {oof_ap:.3f})')
ax.axhline(y=prevalence_int, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_int:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
sci_legend(ax, loc='upper right')
plt.tight_layout()
plt.savefig('figures/PR_Internal.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/PR_Internal.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/PR_Internal.png / .pdf")
plt.close()

print("\n[D] PR — External (locked) 单图 ...")
# ==================== 图 D：PR External（单图）====================
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ax.fill_between(mean_recall, lower_prec_ext, upper_prec_ext, color=C_EXT, alpha=0.15,
                label='95% Bootstrap CI')
ax.plot(rec_ext_orig, prec_ext_orig, color=C_EXT, lw=1.5,
        label=f'External (AP = {ext_ap:.3f})')
ax.axhline(y=prevalence_ext, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_ext:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
sci_legend(ax, loc='upper right')
plt.tight_layout()
plt.savefig('figures/PR_External.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/PR_External.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/PR_External.png / .pdf")
plt.close()


# ==========================================================
# 四补、图 A+B / 图 C+D —— 内部 × 外部「同图叠加」可视化（新增）
#   · 图 A+B → figures/ROC_Internal_vs_External.(png|pdf)
#   · 图 C+D → figures/PR_Internal_vs_External.(png|pdf)
#
#   设计要点（与单图 A–D 完全同源，正文数字保证一致）：
#     - 主线：内部 = OOF 经验曲线（蓝 C_INT），外部 = locked 经验曲线（红 C_EXT）；
#     - 阴影：各自的 95% bootstrap 置信带，alpha 调低以免双带互相压色；
#     - 图例：AUROC / AP 点估计 + 95% 百分位 CI；
#             队列 n 与事件数默认【不】显示（COMB_SHOW_N=False），请在图注中补充；
#     - 标题：Δ(External − Internal) 及其非配对 bootstrap 95% CI
#             （跨 0 → 内外部表现差异无统计学意义）；
#     - 工作点：锁定阈值 clinical_threshold 在两条曲线上的位置（空心圆）。
#   全部开关见下方 COMB_* 常量，按需一键关闭。
# ==========================================================
COMB_FIGSIZE      = (4.0, 4.0)   # 叠加图信息量更大，较单图 3.6×3.6 略放大
COMB_SHOW_BAND    = True         # 是否绘制 95% bootstrap 置信带
COMB_SHOW_OP      = False        # 是否标注锁定阈值工作点
COMB_SHOW_CI_TEXT = True         # 图例中是否附 95% CI
COMB_SHOW_DELTA   = False        # 图例标题是否显示 Δ(External − Internal)
COMB_SHOW_N       = False        # 图例中是否标注各队列 n 与事件数（关闭 → 单行图例）
COMB_LS_INT       = '-'          # 内部线型
COMB_LS_EXT       = '-'          # 外部线型（如需黑白友好，可改为 '--'）
COMB_ALPHA_BAND   = 0.13         # 置信带透明度

#   队列在图例中的显示名（同时用于数值表 Cohort 列，保证图表口径一致）
COMB_NAME_INT     = 'Internal (OOF)'   # 若正文不想出现 OOF，改成 'Internal'
COMB_NAME_EXT     = 'External'         # 按要求去掉原来的 '(locked)'

#   图例位置：可选 'lower right' / 'upper right' / 'lower left' / 'upper left' /
#            'below'（轴下方）/ 'right'（轴右侧）
#   · ROC 曲线恒在对角线上方 → 右下角必为空白，图例放图内最省版面；
#   · PR 曲线形状随患病率与模型强弱变化，图内任何角落都可能被曲线/置信带压住，
#     故默认放到坐标轴下方，保证不遮挡；若两图需并排作为同一张图的 a/b 面板、
#     希望高度完全一致，把下面两个都设成 'below' 即可。
COMB_LEGEND_ROC   = 'lower right'
COMB_LEGEND_PR    = 'upper right'

_LEGEND_POS_MAP = {
    'lower right': dict(loc='lower right'),
    'upper right': dict(loc='upper right'),
    'lower left':  dict(loc='lower left'),
    'upper left':  dict(loc='upper left'),
    'below':       dict(loc='upper center', bbox_to_anchor=(0.5, -0.16)),
    'right':       dict(loc='upper left',   bbox_to_anchor=(1.02, 1.0)),
}


def _u_minus(s):
    """ASCII 连字符 → 排版负号 U+2212（与坐标轴 '1 − Specificity' 风格统一）"""
    return s.replace('-', '\u2212')


def _fmt_metric(name, point, ci):
    """图例正文：'AUROC = 0.812 (0.751–0.869)'"""
    if COMB_SHOW_CI_TEXT and np.all(np.isfinite(ci)):
        return f'{name} = {point:.3f} ({ci[0]:.3f}–{ci[1]:.3f})'
    return f'{name} = {point:.3f}'


def _cohort_label(cohort_name, y, metric_name, point, ci):
    """图例条目：
         COMB_SHOW_N=False → 'External: AUROC = 0.812 (0.751–0.869)'（单行）
         COMB_SHOW_N=True  → 'External: n = 412, 58 events\\nAUROC = 0.812 (...)'"""
    metric = _fmt_metric(metric_name, point, ci)
    if COMB_SHOW_N:
        return f'{cohort_name}: n = {len(y)}, {int(y.sum())} events\n{metric}'
    return f'{cohort_name}: {metric}'


def _fmt_delta(name, point, ci):
    """图例标题：'ΔAUROC (Ext − Int) = +0.021 (−0.048 to +0.091)'"""
    if not COMB_SHOW_DELTA:
        return None
    if np.all(np.isfinite(ci)):
        return _u_minus(f'Δ{name} (Ext - Int) = {point:+.3f} '
                        f'({ci[0]:+.3f} to {ci[1]:+.3f})')
    return _u_minus(f'Δ{name} (Ext - Int) = {point:+.3f}')


def _op_point(y_true, y_prob, threshold, kind='roc'):
    """锁定阈值下的工作点：ROC → (1 − Spec, Sens)；PR → (Recall, Precision)"""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    return (1 - spec, sens) if kind == 'roc' else (sens, prec)


def _comb_legend(ax, extra_handles, pos, title=None):
    """统一图例：曲线句柄 + 置信带/工作点/患病率代理句柄，文字左对齐；
    pos 见 _LEGEND_POS_MAP（图内四角 / 轴下方 / 轴右侧）
    单行标签时收紧 labelspacing，避免图例过于松散"""
    h, l = ax.get_legend_handles_labels()
    h = list(h) + list(extra_handles)
    l = list(l) + [a.get_label() for a in extra_handles]
    kw = dict(_LEGEND_POS_MAP.get(pos, _LEGEND_POS_MAP['lower right']))
    leg = sci_legend(ax, handles=h, labels=l, fontsize=6.5,
                     title=title, title_fontsize=6.5, borderpad=0.45,
                     labelspacing=(0.5 if COMB_SHOW_N else 0.35),
                     handlelength=1.6, handletextpad=0.5, **kw)
    try:                                   # 旧版 matplotlib 无 alignment 参数
        leg._legend_box.align = 'left'
    except Exception:
        pass
    return leg


def _band_handle():
    return mpl.patches.Patch(facecolor='#9E9E9E', edgecolor='none', alpha=0.35,
                             label='95% bootstrap CI')


def _op_handle():
    return Line2D([], [], marker='o', ms=4.5, mfc='white', mec='#4D4D4D', mew=1.3,
                  ls='none', label=f'Locked threshold = {clinical_threshold:.3f}')


# ==================== 图 A+B：ROC Internal vs External ====================
print("\n[A+B] ROC — Internal vs External（叠加）...")
fig, ax = plt.subplots(figsize=COMB_FIGSIZE)

ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--', zorder=1)

if COMB_SHOW_BAND:
    ax.fill_between(mean_fpr, lower_tpr_oof, upper_tpr_oof,
                    color=C_INT, alpha=COMB_ALPHA_BAND, lw=0, zorder=2)
    ax.fill_between(mean_fpr, lower_tpr_ext, upper_tpr_ext,
                    color=C_EXT, alpha=COMB_ALPHA_BAND, lw=0, zorder=2)

ax.plot(fpr_oof_orig, tpr_oof_orig, color=C_INT, lw=1.6, ls=COMB_LS_INT, zorder=4,
        label=_cohort_label(COMB_NAME_INT, y_int, 'AUROC', oof_auc, auc_int_ci))
ax.plot(fpr_ext_orig, tpr_ext_orig, color=C_EXT, lw=1.6, ls=COMB_LS_EXT, zorder=4,
        label=_cohort_label(COMB_NAME_EXT, y_ext, 'AUROC', ext_auc, auc_ext_ci))

_extra = [_band_handle()] if COMB_SHOW_BAND else []
if COMB_SHOW_OP:
    for _y, _p, _c in [(y_int, prob_int, C_INT), (y_ext, prob_ext, C_EXT)]:
        _x0, _y0 = _op_point(_y, _p, clinical_threshold, kind='roc')
        ax.plot([_x0], [_y0], marker='o', ms=4.5, mfc='white', mec=_c, mew=1.3,
                ls='none', zorder=6)
    _extra.append(_op_handle())

ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
_comb_legend(ax, _extra, COMB_LEGEND_ROC,
             title=_fmt_delta('AUROC', delta_auc, d_auc_ci))
plt.tight_layout()
plt.savefig('figures/ROC_Internal_vs_External.png', dpi=DPI, bbox_inches='tight')
plt.savefig('figures/ROC_Internal_vs_External.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/ROC_Internal_vs_External.png / .pdf")
plt.close()


# ==================== 图 C+D：PR Internal vs External ====================
print("\n[C+D] PR — Internal vs External（叠加）...")
fig, ax = plt.subplots(figsize=COMB_FIGSIZE)

if COMB_SHOW_BAND:
    ax.fill_between(mean_recall, lower_prec_oof, upper_prec_oof,
                    color=C_INT, alpha=COMB_ALPHA_BAND, lw=0, zorder=2)
    ax.fill_between(mean_recall, lower_prec_ext, upper_prec_ext,
                    color=C_EXT, alpha=COMB_ALPHA_BAND, lw=0, zorder=2)

# 各队列的事件率基线（PR 曲线的“随机”参考线，随患病率变化）
ax.axhline(prevalence_int, color=C_INT, ls=':', lw=0.9, alpha=0.75, zorder=3)
ax.axhline(prevalence_ext, color=C_EXT, ls=':', lw=0.9, alpha=0.75, zorder=3)

ax.plot(rec_oof_orig, prec_oof_orig, color=C_INT, lw=1.6, ls=COMB_LS_INT, zorder=4,
        label=_cohort_label(COMB_NAME_INT, y_int, 'AP', oof_ap, ap_int_ci))
ax.plot(rec_ext_orig, prec_ext_orig, color=C_EXT, lw=1.6, ls=COMB_LS_EXT, zorder=4,
        label=_cohort_label(COMB_NAME_EXT, y_ext, 'AP', ext_ap, ap_ext_ci))

_extra = [_band_handle()] if COMB_SHOW_BAND else []
_extra.append(Line2D([], [], color='#4D4D4D', ls=':', lw=0.9,
                     label=(f'Prevalence: Int {prevalence_int:.3f} / '
                            f'Ext {prevalence_ext:.3f}')))
if COMB_SHOW_OP:
    for _y, _p, _c in [(y_int, prob_int, C_INT), (y_ext, prob_ext, C_EXT)]:
        _x0, _y0 = _op_point(_y, _p, clinical_threshold, kind='pr')
        ax.plot([_x0], [_y0], marker='o', ms=4.5, mfc='white', mec=_c, mew=1.3,
                ls='none', zorder=6)
    _extra.append(_op_handle())

ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision (PPV)')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
_comb_legend(ax, _extra, COMB_LEGEND_PR,
             title=_fmt_delta('AP', delta_ap, d_ap_ci))
plt.tight_layout()
plt.savefig('figures/PR_Internal_vs_External.png', dpi=DPI, bbox_inches='tight')
plt.savefig('figures/PR_Internal_vs_External.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/PR_Internal_vs_External.png / .pdf")
plt.close()


# ---------- 叠加图对应的数值表（便于正文/审稿直接引用）----------
# 注：图例已不再显示 n 与事件数，此表仍保留 N / Events 两列，信息不丢失
try:
    _sens_i, _prec_i = _op_point(y_int, prob_int, clinical_threshold, kind='pr')
    _sens_e, _prec_e = _op_point(y_ext, prob_ext, clinical_threshold, kind='pr')
    _fpr_i, _ = _op_point(y_int, prob_int, clinical_threshold, kind='roc')
    _fpr_e, _ = _op_point(y_ext, prob_ext, clinical_threshold, kind='roc')
    _comb_tbl = pd.DataFrame([
        {'Cohort': COMB_NAME_INT, 'N': len(y_int), 'Events': int(y_int.sum()),
         'Prevalence': prevalence_int,
         'AUROC': oof_auc, 'AUROC_CI_low': auc_int_ci[0], 'AUROC_CI_high': auc_int_ci[1],
         'AP': oof_ap, 'AP_CI_low': ap_int_ci[0], 'AP_CI_high': ap_int_ci[1],
         'Threshold': clinical_threshold, 'Sensitivity_at_thr': _sens_i,
         'Specificity_at_thr': 1 - _fpr_i, 'Precision_at_thr': _prec_i},
        {'Cohort': COMB_NAME_EXT, 'N': len(y_ext), 'Events': int(y_ext.sum()),
         'Prevalence': prevalence_ext,
         'AUROC': ext_auc, 'AUROC_CI_low': auc_ext_ci[0], 'AUROC_CI_high': auc_ext_ci[1],
         'AP': ext_ap, 'AP_CI_low': ap_ext_ci[0], 'AP_CI_high': ap_ext_ci[1],
         'Threshold': clinical_threshold, 'Sensitivity_at_thr': _sens_e,
         'Specificity_at_thr': 1 - _fpr_e, 'Precision_at_thr': _prec_e},
        {'Cohort': 'Δ (External − Internal)', 'N': np.nan, 'Events': np.nan,
         'Prevalence': prevalence_ext - prevalence_int,
         'AUROC': delta_auc, 'AUROC_CI_low': d_auc_ci[0], 'AUROC_CI_high': d_auc_ci[1],
         'AP': delta_ap, 'AP_CI_low': d_ap_ci[0], 'AP_CI_high': d_ap_ci[1],
         'Threshold': clinical_threshold, 'Sensitivity_at_thr': _sens_e - _sens_i,
         'Specificity_at_thr': _fpr_i - _fpr_e, 'Precision_at_thr': _prec_e - _prec_i},
    ])
    _comb_csv = 'figures/ROC_PR_Internal_vs_External_Metrics.csv'
    _comb_tbl.to_csv(_comb_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ 已保存: {_comb_csv}")
except Exception as e:
    print(f"  ⚠️ 叠加图数值表导出失败: {e}")


    


# ==================== 面板 E：校准曲线（原始概率）====================
print("\n[E] 校准曲线（原始预测概率，无截距校准）...")
plot_calibration_publication_v2(
    y_int, prob_int, y_ext, prob_ext,
    save_path=f'figures/Calibration_Publication_v2_{TIMESTAMP}.png',
    n_bootstrap=1000
)


# ==================== 面板 F：决策曲线分析（原始概率）====================
print("\n[F] 决策曲线分析（原始预测概率，无截距校准）...")

# Treat-all 基线（各自结局率）与阈值网格
threshs = np.arange(0, 1.01, 0.01)
all_nb_int = prevalence_int - (1 - prevalence_int) * (threshs / (1 - threshs + 1e-10))
all_nb_ext = prevalence_ext - (1 - prevalence_ext) * (threshs / (1 - threshs + 1e-10))

fig = plt.figure(figsize=(3.5, 5))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5], hspace=0.3)
ax_main = fig.add_subplot(gs[0])
ax_num = fig.add_subplot(gs[1], sharex=ax_main)

dca_data = {}

for y, p, label, col in [(y_int, prob_int, 'Internal', C_INT),
                         (y_ext, prob_ext, 'External', C_EXT)]:
    try:
        nbs_mean, nbs_lower, nbs_upper = bootstrap_net_benefit(y, p, threshs, n_bootstrap=500)
    except Exception as e:
        print(f"  ⚠️ {label} Bootstrap 失败: {e}, 使用点估计")
        nbs_mean = np.array([calculate_net_benefit(y, p, t) for t in threshs])
        nbs_lower = nbs_mean.copy()
        nbs_upper = nbs_mean.copy()

    ax_main.plot(threshs, nbs_mean, color=col, lw=1.5, label=label, alpha=0.9)
    ax_main.fill_between(threshs, nbs_lower, nbs_upper, color=col, alpha=0.15)

    n_high_risk = [calculate_n_high_risk(p, t, len(y)) for t in threshs]
    ax_num.plot(threshs, n_high_risk, color=col, lw=1.5, alpha=0.8)

    dca_data[label] = {
        'net_benefit': nbs_mean, 'ci_lower': nbs_lower,
        'ci_upper': nbs_upper, 'n_high_risk': n_high_risk
    }
    print(f"  ✓ {label} 完成")

ax_main.plot(threshs, all_nb_int, color=C_INT, ls='--', lw=1.0,
             alpha=0.45, label=f'Treat all Int ({prevalence_int:.1%})')
ax_main.plot(threshs, all_nb_ext, color=C_EXT, ls='--', lw=1.0,
             alpha=0.45, label=f'Treat all Ext ({prevalence_ext:.1%})')
ax_main.axhline(y=0, color='black', lw=0.8, ls='-.', label='Treat none', alpha=0.7)
ax_main.axvline(clinical_threshold, color='#333333', ls='--', lw=1.2,
                alpha=0.9, label=f'Locked threshold = {clinical_threshold:.3f}')
ax_main.set_xlim([0, 0.5])
y_max_candidates = [all_nb_int.max(), all_nb_ext.max()]
for k in dca_data:
    y_max_candidates.append(max(dca_data[k]['net_benefit']))
y_max = max(y_max_candidates)
ax_main.set_ylim([-0.05, y_max + 0.05])
ax_main.set_ylabel('Net benefit')
ax_main.tick_params(axis='x', labelbottom=False)          # 主面板隐藏x标签
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
sci_legend(ax_main, loc='upper right', fontsize=7,
           handlelength=1.8, handletextpad=0.5, labelspacing=0.35, borderpad=0.4)
ax_num.set_ylabel('Predicted risk ≥\nthreshold per 1000', fontsize=7)
ax_num.set_xlabel('Threshold probability')
ax_num.set_ylim([0, 1000])
ax_num.spines['top'].set_visible(False)
ax_num.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/DCA_Analysis.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/DCA_Analysis.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/DCA_Analysis.png / .pdf")
plt.close()

print("\n" + "=" * 60)
print("✅ 正文六图已全部生成（均为单图，含 PNG + PDF）：")
print("   A   : figures/ROC_Internal.png")
print("   B   : figures/ROC_External.png")
print("   C   : figures/PR_Internal.png")
print("   D   : figures/PR_External.png")
print("   A+B : figures/ROC_Internal_vs_External.png   ← 新增（内外部叠加）")
print("   C+D : figures/PR_Internal_vs_External.png    ← 新增（内外部叠加）")
print("         figures/ROC_PR_Internal_vs_External_Metrics.csv（对应数值表）")
print(f"   E   : figures/Calibration_Publication_v2_{TIMESTAMP}.png")
print("   F   : figures/DCA_Analysis.png")
print("=" * 60)


# ==========================================================
# 五、混淆矩阵（内部 + 外部，均使用原始概率与原始锁定阈值）
# ==========================================================
print("\n" + "=" * 60)
print("🧩 绘制混淆矩阵 ...")
print("=" * 60)


def plot_confusion_matrix(y_true, y_prob, threshold, dataset_name, color=None,
                          save_path=None, cmap='Blues'):
    """绘制混淆矩阵热图（阈值为原始概率尺度上的锁定阈值）"""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # 计算百分比（按真实类别行归一）
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建标注文本
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    # 绘图 — 关闭 sns 自带 cbar, 后面手动添加等高 colorbar
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=False,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['Non-recurrence', 'Recurrence'],
                yticklabels=['Non-recurrence', 'Recurrence'],
                ax=ax)

    # 用 make_axes_locatable 切出等高 colorbar 轴
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(ax.collections[0], cax=cax, label='')
    cbar.outline.set_visible(False)          # 去掉 colorbar 黑色外框线

    plt.sca(ax)
    plt.title(f'{dataset_name}', fontsize=10, pad=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  ✓ 已保存: {save_path}")
    plt.close()

    return cm


cm_int = plot_confusion_matrix(y_int, prob_int, clinical_threshold,
                               'Internal',
                               save_path='figures/Confusion_Matrix_Internal.png',
                               cmap='Blues')

cm_ext = plot_confusion_matrix(y_ext, prob_ext, clinical_threshold,
                               'External',
                               save_path='figures/Confusion_Matrix_External.png',
                               cmap='Oranges')


# ---------------- 并排混淆矩阵对比（版本 1：行归一百分比）----------------
print("\n[混淆矩阵] 绘制并排对比图 ...")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for idx, (y_true, y_prob, threshold, dataset_name, color, ax) in enumerate([
    (y_int, prob_int, clinical_threshold, 'Internal', C_INT, axes[0]),
    (y_ext, prob_ext, clinical_threshold, 'External', C_EXT, axes[1])
]):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    cmap = 'Blues' if idx == 0 else 'Oranges'

    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=True,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['Non-recurrence', 'Recurrence'],
                yticklabels=['Non-recurrence', 'Recurrence'],
                annot_kws={"size": 14},
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_title(f'{dataset_name}', fontsize=15, pad=10)
    ax.set_ylabel('True Label', fontsize=13, labelpad=10)
    ax.set_xlabel('Predicted Label', fontsize=13, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

plt.tight_layout()
plt.savefig('figures/Confusion_Matrix_Comparison.png', dpi=DPI, bbox_inches='tight')
plt.savefig('figures/Confusion_Matrix_Comparison.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/Confusion_Matrix_Comparison.png")
plt.close()



# ---------------- 混淆矩阵指标详细对比表 ----------------
print("\n[混淆矩阵] 生成指标详细对比表 ...")

# 若上游提供了外部 bootstrap 指标表（df_bootstrap_ext），则附上 95% CI
_df_boot_ext = df_bootstrap_ext          # 来自 External_Val_Bootstrap_1000_<TS>.csv


def _ci_str(key):
    """从外部 bootstrap 指标表安全取 95% CI 字符串
       （建模脚本保存的指标键：AUC / Brier / AP / Sensitivity / Specificity /
         PPV / NPV / F1 / MCC / G_mean / Balanced_Acc）"""
    if _df_boot_ext is None or key not in _df_boot_ext.index:
        return 'N/A'
    if '95%_CI_Lower' not in _df_boot_ext.columns:
        return 'N/A'
    lo = _df_boot_ext.loc[key, '95%_CI_Lower']
    hi = _df_boot_ext.loc[key, '95%_CI_Upper']
    return f"{lo:.4f}-{hi:.4f}"


metrics_comparison = []

for dataset_name, y_true, y_p, thresh in [
    ('Internal', y_int, prob_int, clinical_threshold),
    ('External', y_ext, prob_ext, clinical_threshold)
]:
    y_pred = (y_p >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    g_mean = np.sqrt(sens * spec)
    balanced_acc = (sens + spec) / 2.0

    # 仅外部验证附 bootstrap 95% CI
    if dataset_name == 'External':
        sens_ci, spec_ci = _ci_str('Sensitivity'), _ci_str('Specificity')
        ppv_ci,  npv_ci  = _ci_str('PPV'),         _ci_str('NPV')
        f1_ci,   mcc_ci  = _ci_str('F1'),          _ci_str('MCC')
        gmean_ci, balacc_ci = _ci_str('G_mean'),   _ci_str('Balanced_Acc')
    else:
        sens_ci = spec_ci = ppv_ci = npv_ci = f1_ci = mcc_ci = gmean_ci = balacc_ci = 'N/A'

    metrics_comparison.append({
        'Dataset': dataset_name,
        'Threshold': thresh,
        'True Positive (TP)': tp,
        'True Negative (TN)': tn,
        'False Positive (FP)': fp,
        'False Negative (FN)': fn,
        'Sensitivity': f"{sens:.4f}",
        'Sensitivity 95% CI': sens_ci,
        'Specificity': f"{spec:.4f}",
        'Specificity 95% CI': spec_ci,
        'PPV': f"{ppv:.4f}",
        'PPV 95% CI': ppv_ci,
        'NPV': f"{npv:.4f}",
        'NPV 95% CI': npv_ci,
        'Accuracy': f"{acc:.4f}",
        'F1-Score': f"{f1:.4f}",
        'F1-Score 95% CI': f1_ci,
        'MCC': f"{mcc:.4f}",
        'MCC 95% CI': mcc_ci,
        'G-mean': f"{g_mean:.4f}",
        'G-mean 95% CI': gmean_ci,
        'Balanced Accuracy': f"{balanced_acc:.4f}",
        'Balanced Accuracy 95% CI': balacc_ci,
    })

df_metrics_comparison = pd.DataFrame(metrics_comparison)
df_metrics_comparison.to_csv('figures/Confusion_Matrix_Metrics_Comparison.csv', index=False)
print("  ✓ 已保存: figures/Confusion_Matrix_Metrics_Comparison.csv")

print("\n" + "=" * 80)
print("混淆矩阵指标详细对比（原始概率 + 原始锁定阈值）")
print("=" * 80)
print(df_metrics_comparison.to_string(index=False))
print("=" * 80 + "\n")


# ==========================================================================================
# 六、Figure 2A / 2B — 风险梯度 & 事件集中度
# ==========================================================================================

# ---------------- 可调参数 ----------------
LOW_CUT, HIGH_CUT = 0.05, 0.15          # 与正文一致的固定切点（原始概率尺度）
OUTPUT_DIR = "figures"
TIER_NAMES = ["Low", "Moderate", "High"]   # 与正文统一：Moderate（非 Medium）
PATIENT_COLOR = "#9DC3E6"                # 患者占比（浅蓝）
EVENT_COLOR   = "#C00000"                # 事件占比 / 事件（红）
Z_95 = 1.959963984540054                 # 95% 双侧正态分位数
_DPI = DPI
os.makedirs(OUTPUT_DIR, exist_ok=True)

y_int_arr = np.asarray(y_int)
y_ext_arr = np.asarray(y_ext)
p_int_arr = np.asarray(prob_int)
p_ext_arr = np.asarray(prob_ext)

# 队列定义: (显示名, 真实标签, 预测概率, 颜色)
COHORTS = [
    ("Development",          y_int_arr, p_int_arr, COLOR_INTERNAL),
    ("External validation",  y_ext_arr, p_ext_arr, COLOR_EXTERNAL),
]


# ------------------------------------------------------------------
# 二项比例 95% CI —— Wilson 评分区间
#   相比正态近似(Wald)，在样本量小或事件率接近 0（如 Low 组）时更稳健，
#   且区间始终落在 [0, 1] 内，适合作再脱离率这类小比例的不确定性展示。
# ------------------------------------------------------------------
def _wilson_ci(k, n, z=Z_95):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


# ------------------------------------------------------------------
# 风险分组(固定切点) → 各组 n/events/event_rate/95%CI 及整体 χ² p
# ------------------------------------------------------------------
def _risk_groups(y_true, y_prob, low_cut=LOW_CUT, high_cut=HIGH_CUT):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    masks = [y_prob < low_cut,
             (y_prob >= low_cut) & (y_prob < high_cut),
             y_prob >= high_cut]
    groups, cont = [], []
    for m in masks:
        n = int(m.sum())
        ev = int(y_true[m].sum()) if n else 0
        lo, hi = _wilson_ci(ev, n)
        groups.append({"n": n, "events": ev, "non_events": n - ev,
                       "event_rate": ev / n if n else 0.0,
                       "ci_low": lo, "ci_high": hi})
        cont.append([ev, n - ev])
    try:
        _, p, _, _ = chi2_contingency(np.array(cont))
    except Exception:
        p = np.nan
    return groups, p


def _p_str(p):
    """仅返回 p 的数值部分: '<0.001' / '0.023' / 'N/A'。"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _p_label(p, prefix="\u03c7\u00b2 p"):
    """
    规范化 p 值写法（避免出现 'χ² p = <0.001' 这类等号+不等号并存）：
        p < 0.001  →  'χ² p <0.001'
        其他       →  'χ² p = 0.023'
    """
    s = _p_str(p)
    return f"{prefix} {s}" if s.startswith("<") else f"{prefix} = {s}"


# ==================================================================
# Figure 2A — 风险梯度: 点图 + 95% CI
# ==================================================================
def plot_figure2a(cohorts=COHORTS, low_cut=LOW_CUT, high_cut=HIGH_CUT):
    ranges = [f"<{low_cut:.0%}",
              f"{low_cut:.0%}\u2013{high_cut:.0%}",
              f"\u2265{high_cut:.0%}"]
    x = np.arange(3)
    n_c = len(cohorts)
    span = 0.30
    offsets = np.linspace(-span, span, n_c) if n_c > 1 else np.array([0.0])

    data = []
    for name, yt, yp, color in cohorts:
        groups, p = _risk_groups(yt, yp, low_cut, high_cut)
        rate = np.array([g["event_rate"] for g in groups]) * 100
        lo = np.array([g["ci_low"] for g in groups]) * 100
        hi = np.array([g["ci_high"] for g in groups]) * 100
        yerr = np.clip(np.vstack([rate - lo, hi - rate]), 0, None)   # 2×3
        tot_n = sum(g["n"] for g in groups)
        data.append((name, rate, yerr, hi, tot_n, p, color))

    ymax = max(h.max() for _, _, _, h, _, _, _ in data)             # 由 CI 上界定上限

    fig, ax = plt.subplots(figsize=(4, 3.5))
    for off, (name, rate, yerr, hi, tot_n, p, color) in zip(offsets, data):
        xc = x + off
        ax.plot(xc, rate, ls="--", lw=1.0, color=color, alpha=0.45, zorder=2)
        ax.errorbar(xc, rate, yerr=yerr, fmt="o", ms=7, capsize=4,
                    elinewidth=1.4, capthick=1.4, color=color,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4,
                    label=f"{name} (n={tot_n:,}; {_p_label(p)})")
        for cx, r, h in zip(xc, rate, hi):
            ax.text(cx, h + ymax * 0.025, f"{r:.1f}%", ha="center", va="bottom",
                    fontsize=8, color=color, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({r})" for t, r in zip(TIER_NAMES, ranges)])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Observed 12-month redetachment rate (%)")
    ax.set_ylim(0, ymax * 1.22)
    ax.set_title("Observed 12-month redetachment rate by risk group",
                 fontsize=10, pad=10)
    ax.yaxis.grid(True, ls=":", lw=0.6, alpha=0.5, zorder=0)
    sci_legend(ax, loc="upper left")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "Figure2A_Risk_Gradient.png")
    plt.savefig(out, dpi=_DPI)
    plt.savefig(out.replace(".png", ".pdf"))                        # 矢量版备用
    plt.close()
    print(f"  \u2713 Figure 2A 已保存: {out}")
    return out


# ==================================================================
# Figure 2B — 事件集中度: 上下对齐的横向条形图
# ==================================================================
def plot_figure2b(cohorts=COHORTS, low_cut=LOW_CUT, high_cut=HIGH_CUT):
    n = len(cohorts)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.4 * n),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    y, bh = np.arange(3), 0.38

    legend_handles = None
    for ax, (name, yt, yp, _color) in zip(axes, cohorts):
        groups, p = _risk_groups(yt, yp, low_cut, high_cut)
        ns = np.array([g["n"] for g in groups], float)
        ev = np.array([g["events"] for g in groups], float)
        ps = ns / ns.sum() * 100 if ns.sum() else np.zeros(3)
        es = ev / ev.sum() * 100 if ev.sum() else np.zeros(3)

        ax.barh(y - bh / 2, ps, bh, color=PATIENT_COLOR, edgecolor="white",
                linewidth=0.8, label="% of patients", zorder=3)
        ax.barh(y + bh / 2, es, bh, color=EVENT_COLOR, edgecolor="white",
                linewidth=0.8, label="% of events", zorder=3)
        for cy, v in zip(y - bh / 2, ps):
            ax.text(v + 1.0, cy, f"{v:.1f}%", va="center", ha="left", fontsize=8)
        for cy, v in zip(y + bh / 2, es):
            ax.text(v + 1.0, cy, f"{v:.1f}%", va="center", ha="left", fontsize=8)

        # High 组（事件占比最大者）富集倍数
        hi = int(np.argmax(es))
        enr = es[hi] / ps[hi] if ps[hi] > 0 else np.nan
        ax.annotate(f"{enr:.1f}\u00d7 enrichment", (es[hi], y[hi] + bh / 2),
                    xytext=(34, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=EVENT_COLOR,
                    fontweight="bold")
        ax.text(0.985, 0.50,
                f"{TIER_NAMES[hi]} risk: {ps[hi]:.0f}% of patients\n"
                f"capture {es[hi]:.0f}% of events\n"
                f"{_p_label(p)}",
                transform=ax.transAxes, va="center", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3F3",
                          ec=EVENT_COLOR, lw=0.8))

        ax.set_yticks(y)
        ax.set_yticklabels(TIER_NAMES)
        ax.invert_yaxis()                       # Low 在上、High 在下（与 2A 顺序一致）
        ax.set_xlim(0, 100)
        ax.set_title(f"{name} (n={int(ns.sum()):,}, events={int(ev.sum())})",
                     fontsize=10, loc="left")
        ax.xaxis.grid(True, ls=":", lw=0.6, alpha=0.5, zorder=0)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[-1].set_xlabel("Proportion (%)")
    fig.suptitle("Event concentration by risk group", fontsize=12, y=0.99)
    fig.legend(legend_handles, legend_labels, loc="lower center",
               ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = os.path.join(OUTPUT_DIR, "Figure2B_Event_Concentration.png")
    plt.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Figure 2B 已保存: {out}")
    return out


print(f"\n{'='*60}")
print("\U0001F4CA 绘制 Figure 2A / 2B (风险梯度 & 事件集中度)...")
print(f"{'='*60}")
print(f"  切点: Low <{LOW_CUT:.0%} | Moderate {LOW_CUT:.0%}\u2013{HIGH_CUT:.0%} | High \u2265{HIGH_CUT:.0%}")
for name, yt, yp, _c in COHORTS:
    g, p = _risk_groups(yt, yp)
    rates = " / ".join(
        f"{x['event_rate']:.1%} [{x['ci_low']:.1%}\u2013{x['ci_high']:.1%}]" for x in g)
    print(f"  [{name}] 事件率 Low/Moderate/High = {rates}  ({_p_label(p)})")

plot_figure2a()
plot_figure2b()
print(f"{'='*60}\n")


# ============================================================
# 七、特征选择频率可视化（LASSO 棒棒糖图）
# ============================================================
print("=" * 60)
print("🍭 绘制 LASSO 特征选择稳定性棒棒糖图 ...")
print("=" * 60)


def plot_lollipop_chart(data, model_name, title_suffix='', save_path=None):
    """棒棒糖图：展示 LASSO 在交叉验证中选择每个特征的稳定性"""
    fig, ax = plt.subplots(figsize=(4.5, max(4, len(data) * 0.4)))

    colors = []
    for freq in data['Selection_Freq']:
        if freq >= 0.8:
            colors.append('#2ecc71')
        elif freq >= 0.5:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')

    y_pos = np.arange(len(data))
    ax.hlines(y=y_pos, xmin=0, xmax=data['Selection_Freq'],
              color='gray', alpha=0.4, linewidth=1.2)
    ax.scatter(data['Selection_Freq'], y_pos,
               color=colors, s=60, alpha=0.85,
               edgecolors='white', linewidth=0.8, zorder=4)

    for i, (freq, count, total) in enumerate(zip(
            data['Selection_Freq'], data['Selection_Count'], data['Total_Folds'])):
        ax.text(freq + 0.05, i, f'{freq:.1%}', va='center', fontsize=7)

    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.3, label='High Stability (80%)')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.3, label='Medium Stability (50%)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['Feature'], fontsize=10)
    ax.set_xlabel('LASSO Selection Frequency')
    ax.set_xlim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sci_legend(ax, loc='lower right')
    ax.grid(axis='x', alpha=0.15, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


try:
    # ---------- 数据加载 ----------
    package = joblib.load(f"{DATA_PATH}/Model_Package_{TIMESTAMP}.pkl")
    best_model_name_lp = package['best_model_name']
    freq_df = pd.read_csv(f"{DATA_PATH}/Feature_Selection_Frequency_{TIMESTAMP}.csv")
    rename_df_feature_col(freq_df)                     # ✅ 映射为 SCI 展示标签

    SCENARIOS = {
        'Original Preop Model': {
            'timestamp': TIMESTAMP,
            'package': package,
            'best_model': best_model_name_lp,
            'freq_df': freq_df,
        },
    }

    # ---------- 敏感性分析模型（可选）----------
    try:
        pkg_intra   = joblib.load(f"{DATA_PATH}/Sensitivity_Model_Package_latest.pkl")

        best_intra   = pkg_intra['best_model_name']

        freq_intra   = pd.read_csv(f"{DATA_PATH}/Feature_Selection_Frequency_{TIMESTAMP_INTRA}.csv")
        rename_df_feature_col(freq_intra)              # ✅ 映射为 SCI 展示标签

        SCENARIOS['Pre+Intraoperative Model'] = {
            'timestamp': TIMESTAMP_INTRA, 'package': pkg_intra,
            'best_model': best_intra, 'freq_df': freq_intra,
        }
        
        print(f"  术中模型: {best_intra}, 特征数: {len(freq_intra[freq_intra['Model']==best_intra])}")
    except FileNotFoundError as e:
        print(f"  ⚠️ 敏感性分析数据缺失（{e}），仅绘制主模型棒棒糖图。")

    print(f"  主模型: {best_model_name_lp}, "
          f"特征数: {len(freq_df[freq_df['Model']==best_model_name_lp])}")

    # ---------- 逐场景出图 ----------
    for label, cfg in SCENARIOS.items():
        df_sub = cfg['freq_df']
        df_sub = df_sub[df_sub['Model'] == cfg['best_model']].copy()
        df_sub = df_sub.sort_values('Selection_Freq', ascending=True)
        safe = label.replace(' ', '_').replace('+', '')
        plot_lollipop_chart(df_sub, cfg['best_model'],
                            title_suffix=f'{label}',
                            save_path=f'figures/LASSO_Lollipop_{safe}.png')
        print(f"  ✅ Lollipop saved: {label}")

except FileNotFoundError as e:
    print(f"  ⚠️ 跳过棒棒糖图（缺少文件）: {e}")


# ============================================================================
# 八、SHAP 解释性分析
#   ⚠️ 本模块与截距校准无关；阈值一律使用原始尺度 clinical_threshold
# ============================================================================

# ---------- SHAP 密集图专用较小字号（局部覆盖 set_sci_style）----------
SHAP_STYLE = {
    'font.family':        'Arial',
    'font.size':          8,
    'axes.titlesize':     9,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    7,
    'legend.title_fontsize': 7,
    'axes.linewidth':     0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'axes.grid':          False,
    'legend.frameon':     False,
    'legend.borderpad':   0.3,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'figure.dpi':         150,
}

# SHAP 局部样式覆盖（在 set_sci_style 基础上最小修改）
SHAP_RC = {
    'font.size':            8,
    'axes.titlesize':       9,
    'axes.labelsize':       9,
    'xtick.labelsize':      7,
    'ytick.labelsize':      8,
    'axes.linewidth':       0.8,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'xtick.major.width':    0.8,
    'ytick.major.width':    0.8,
    'xtick.major.size':     3.5,
    'ytick.major.size':     3.5,
    'xtick.minor.visible':  False,
    'ytick.minor.visible':  False,
    'axes.grid':            False,
    'legend.frameon':       False,
    'savefig.dpi':          600,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.05,
}

# ---------- 统一配色（Nature / Lancet 风格）----------
C_BLUE    = '#2166AC'
C_RED     = '#B2182B'
C_GREEN   = '#1B7837'
C_ORANGE  = '#E08214'
C_GRAY    = '#636363'
C_LIGHT   = '#D9D9D9'
C_POS     = '#D6604D'          # 正向 (风险↑)
C_NEG     = '#4393C3'          # 负向 (保护↓)
C_BAR     = COLORS_LIST[1]     # 条形填充
C_ANNOT   = '#636363'          # 灰色 — 数值标注文字

AL_RAW_KEY = 'AL'              # 建模时眼轴长的原始（未标准化）列名
N_BOOT_AL  = 500               # 依赖图 bootstrap 次数


# ---------- 辅助函数 ----------
def despine(ax, top=True, right=True):
    """移除指定边框"""
    if top:    ax.spines['top'].set_visible(False)
    if right:  ax.spines['right'].set_visible(False)


def sci_ax(ax, xlabel='', ylabel='', title=''):
    """快速设置通用 SCI 轴属性"""
    despine(ax)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    if title:  ax.set_title(title, fontsize=9, pad=8)
    ax.tick_params(axis='both', labelsize=7, width=0.6)


def run_shap_module():
    """完整 SHAP 分析与出图（Panel G/H、Fig3A–3D、眼轴长依赖图）"""

    print(f"\n{'='*60}")
    print("🔍 开始 SHAP 分析...")
    print(f"{'='*60}\n")

    # 保存当前 rcParams 以便 SHAP 结束后恢复
    _saved_rcParams = mpl.rcParams.copy()
    mpl.rcParams.update(SHAP_STYLE)

    os.makedirs('figures/SHAP_Dependence', exist_ok=True)

    try:
        # ------------------------------------------
        # 步骤 0: 加载 SHAP 专用数据包
        # ------------------------------------------
        print("[0/9] 加载 SHAP 分析数据包...")

        shap_files = glob.glob(f"{DATA_PATH}/SHAP_Analysis_Data_{TIMESTAMP}.pkl")
        if not shap_files:
            raise FileNotFoundError("❌ 未找到 SHAP 数据包！请先运行建模代码。")

        latest_shap_file = max(shap_files, key=os.path.getctime)
        print(f"  📦 加载: {os.path.basename(latest_shap_file)}")

        shap_package = joblib.load(latest_shap_file)

        X_train_transformed = shap_package['X_train_transformed']
        X_ext_transformed   = shap_package['X_ext_transformed']
        feature_names       = shap_package['feature_names']

        # 稳健重命名：rename_feature_list 无效时，用 rename_feature 逐个兜底
        _raw  = list(feature_names)
        _disp = list(rename_feature_list(_raw))
        if _disp == _raw:                              # 没改成功
            _disp = [rename_feature(f) for f in _raw]
        feature_names = _disp
        print("✓ 展示标签:", feature_names)

        classifier = shap_package['best_model_step']

        # 阈值一致性检查（原始概率尺度，不做任何校准）
        assert np.isclose(shap_package['optimal_threshold'], clinical_threshold), \
            (f"阈值不一致! shap={shap_package['optimal_threshold']}, "
             f"main={clinical_threshold}")

        print(f"  ✓ 训练集: {X_train_transformed.shape}")
        print(f"  ✓ 外部验证集: {X_ext_transformed.shape}")
        print(f"  ✓ 特征数量: {len(feature_names)}")
        print(f"  ✓ 分类器类型: {type(classifier).__name__}")

        y_train    = Y_TRAIN
        y_ext_shap = Y_EXT_PKG

        # ------------------------------------------
        # 步骤 1: 准备 SHAP 分析数据
        # ------------------------------------------
        print("\n[1/9] 准备 SHAP 分析数据...")

        X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        X_ext_df   = pd.DataFrame(X_ext_transformed,   columns=feature_names)

        print(f"  ✓ 训练集特征矩阵: {X_train_df.shape}")
        print(f"  ✓ 外部验证集特征矩阵: {X_ext_df.shape}")

        SHAP_SAMPLE_SIZE = 1000
        if len(X_train_df) > SHAP_SAMPLE_SIZE:
            print(f"  ℹ️ 样本量较大，随机采样 {SHAP_SAMPLE_SIZE} 个样本进行分析...")
            from sklearn.model_selection import train_test_split
            X_sample, _, y_sample, _ = train_test_split(
                X_train_df, y_train,
                train_size=SHAP_SAMPLE_SIZE,
                stratify=y_train,
                random_state=42
            )
        else:
            X_sample = X_train_df
            y_sample = y_train
            print(f"  ✓ 使用全部 {len(X_sample)} 个样本")

        # ------------------------------------------
        # 步骤 2: 创建 SHAP 解释器
        # ------------------------------------------
        print("\n[2/9] 创建 SHAP 解释器...")

        actual_classifier = classifier
        print(f"  ✓ 模型类型: {type(actual_classifier).__name__}")
        print(f"  🔍 是否有 predict_proba: {hasattr(actual_classifier, 'predict_proba')}")
        print(f"  🔍 是否有 decision_function: {hasattr(actual_classifier, 'decision_function')}")

        model_type = type(actual_classifier).__name__

        # ============================================================
        # 线性模型 SHAP 输出尺度开关
        #   "logodds"     -> LinearExplainer：SHAP 值在对数几率(log-odds)尺度，
        #                    解析解、精确、秒级完成，是逻辑回归的理论正确解释空间。
        #                    可加性：sum(shap)+base == logit(p) == decision_function
        #   "probability" -> KernelExplainer 解释 predict_proba：SHAP 值在概率尺度，
        #                    对临床更直观（"该特征使预测风险 +0.15"），但为近似值。
        #                    可加性：sum(shap)+base == p == predict_proba
        # ⚠️ 两种尺度的数值范围/图轴含义不同，切换后所有 SHAP 数值都会随之改变。
        # ============================================================
        SHAP_LINEAR_OUTPUT_SPACE = "logodds"      # 可改为 "probability"

        LINEAR_MODELS = [
            'LogisticRegression', 'LogisticRegressionCV',
            'RidgeClassifier', 'RidgeClassifierCV',
            'SGDClassifier', 'Perceptron', 'LinearDiscriminantAnalysis',
        ]

        if model_type in ['LinearSVC', 'SVC']:
            print("  ℹ️ 使用 LinearExplainer (SVM模式)...")
            try:
                explainer = shap.LinearExplainer(
                    actual_classifier, X_sample,
                    feature_perturbation="interventional"
                )
                print("  ✓ LinearExplainer 创建完成")
            except Exception as e:
                print(f"  ⚠️ LinearExplainer 失败 ({e})，转为 KernelExplainer...")
                if hasattr(actual_classifier, 'decision_function'):
                    predict_fn = lambda x: expit(actual_classifier.decision_function(x))
                    print("  ✓ 使用 decision_function + sigmoid")
                else:
                    raise ValueError("SVM模型没有decision_function")
                background_data = X_sample.sample(min(100, len(X_sample)), random_state=42)
                explainer = shap.KernelExplainer(predict_fn, background_data)

        elif model_type in LINEAR_MODELS and SHAP_LINEAR_OUTPUT_SPACE == "logodds":
            # ✅ 逻辑回归等线性模型：LinearExplainer（解析解，精确且秒级完成）
            print(f"  ℹ️ 检测到线性模型 {model_type}，使用 LinearExplainer (线性模型模式)...")
            print("  ⚠️ SHAP 值在 log-odds(对数几率)尺度；可加性 sum(shap)+base = logit(p)")
            explainer = shap.LinearExplainer(
                actual_classifier, X_sample,
                feature_perturbation="interventional"
            )
            _base = float(np.ravel(explainer.expected_value)[0])
            print(f"  ✓ LinearExplainer 创建完成 (base/期望值 = {_base:.4f} log-odds)")

        elif model_type in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
            print("  ℹ️ 使用 TreeExplainer...")
            explainer = shap.TreeExplainer(actual_classifier)
            print("  ✓ TreeExplainer 创建完成")

        else:
            # 通用 / 概率尺度：KernelExplainer 解释 predict_proba（近似值）
            if model_type in LINEAR_MODELS:
                print(f"  ℹ️ 线性模型 {model_type}，按概率尺度输出 → KernelExplainer(predict_proba)...")
            else:
                print("  ℹ️ 使用 KernelExplainer (通用模式)...")
            print("  ⏳ 初始化中（需要几分钟）...")
            background_data = X_sample.sample(min(200, len(X_sample)), random_state=42)

            if hasattr(actual_classifier, 'predict_proba'):
                predict_fn = lambda x: actual_classifier.predict_proba(x)[:, 1]
                print("  ✓ 使用 predict_proba")
            elif hasattr(actual_classifier, 'decision_function'):
                predict_fn = lambda x: expit(actual_classifier.decision_function(x))
                print("  ✓ 使用 decision_function + sigmoid")
            else:
                raise ValueError("模型没有predict_proba或decision_function")

            test_pred = predict_fn(background_data.iloc[:5].values)
            print(f"  🔍 预测测试: {test_pred}")
            print(f"  🔍 预测范围: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

            explainer = shap.KernelExplainer(predict_fn, background_data)
            print("  ✓ KernelExplainer 创建完成")

        # ------------------------------------------
        # 步骤 3: 计算 SHAP 值
        # ------------------------------------------
        print("\n[3/9] 计算 SHAP 值...")
        print("  ⏳ 这可能需要几分钟...")

        explainer_type = type(explainer).__name__
        print(f"  ℹ️ Explainer类型: {explainer_type}")

        if "Tree" in explainer_type:
            shap_values = explainer(X_sample, check_additivity=False)
        elif "Linear" in explainer_type:
            shap_values = explainer(X_sample)
        else:
            print(f"  ℹ️ 检测到 {explainer_type}，使用兼容模式计算...")
            raw_values = explainer.shap_values(X_sample)

            if isinstance(raw_values, list):
                vals = raw_values[1]
                base_val = explainer.expected_value[1]
            else:
                vals = raw_values
                base_val = explainer.expected_value

            shap_values = shap.Explanation(
                values=vals,
                base_values=base_val,
                data=X_sample.values,
                feature_names=list(feature_names)
            )
            print(f"  🔍 SHAP值范围: [{vals.min():.4f}, {vals.max():.4f}]")
            print(f"  🔍 SHAP值均值: {np.abs(vals).mean():.4f}")
            print(f"  🔍 非零SHAP值: {(np.abs(vals) > 1e-6).sum()} / {vals.size}")

        # 提取正类 SHAP 值
        shap_values_pos = shap_values
        if len(shap_values.values.shape) == 3:
            print("  ✓ 检测到多维 SHAP 值，正在提取正类...")
            shap_values_pos = shap.Explanation(
                values=shap_values.values[:, :, 1],
                base_values=(shap_values.base_values[:, 1]
                             if shap_values.base_values.ndim > 1 else shap_values.base_values),
                data=shap_values.data,
                feature_names=shap_values.feature_names
            )

        print(f"  ✓ 最终 SHAP 值矩阵形状: {shap_values_pos.values.shape}")

        if np.abs(shap_values_pos.values).max() < 1e-6:
            print("  ⚠️⚠️⚠️ 警告：所有SHAP值接近0！模型可能有问题！")
        else:
            print("  ✓ SHAP值计算成功！")

        # ------------------------------------------
        # 步骤 4: 准备出图所需数据
        # ------------------------------------------
        pred_probs = classifier.predict_proba(X_sample)[:, 1]
        high_risk_idx = int(np.argmax(pred_probs))
        low_risk_idx  = int(np.argmin(pred_probs))
        high_risk_prob = pred_probs[high_risk_idx]
        low_risk_prob  = pred_probs[low_risk_idx]

        mean_abs_shap = np.abs(shap_values_pos.values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': mean_abs_shap
        }).sort_values('SHAP_Importance', ascending=False)

        print(f"  High-risk sample: idx={high_risk_idx}, prob={high_risk_prob:.4f}")
        print(f"  Low-risk  sample: idx={low_risk_idx},  prob={low_risk_prob:.4f}")

        # ══════════════════════════════════════════════════════════════
        # Panel G — Feature Importance（左，显示 y 轴标签）
        # ══════════════════════════════════════════════════════════════
        print("\n[Panel G] Feature Importance...")

        set_sci_style()
        mpl.rcParams.update(SHAP_RC)

        fig_g, ax_g = plt.subplots(figsize=(3.5, 2.75))

        top_n   = 5
        imp_top = importance_df.head(top_n).iloc[::-1].reset_index(drop=True)
        vals    = imp_top['SHAP_Importance'].values
        names   = imp_top['Feature'].values
        n       = len(vals)

        cmap_shap  = mcm.get_cmap('RdBu_r')
        norm_vals  = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        bar_colors = [cmap_shap(0.55 + 0.40 * v) for v in norm_vals]

        spacing = 0.8                     # 条形中心间距，越小间距越窄
        y_pos = np.arange(n) * spacing

        bars = ax_g.barh(
            y_pos, vals,
            color=bar_colors,
            edgecolor='none',
            height=0.55,
        )

        ax_g.set_yticks(y_pos)
        ax_g.set_yticklabels(names, fontsize=8)
        ax_g.tick_params(axis='y', left=False)
        ax_g.set_ylim(-spacing * 0.6, y_pos[-1] + spacing * 0.6)

        # ── x 轴 ──
        x_max = vals.max()
        ax_g.set_xlim(0, x_max * 1.30)
        ax_g.set_xlabel('Mean |SHAP|', fontsize=9, labelpad=4)
        ax_g.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune='both'))
        ax_g.tick_params(axis='x', width=0.8, size=3.5, labelsize=7)

        for bar, val in zip(bars, vals):
            ax_g.text(
                val + x_max * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}',
                va='center', ha='left',
                fontsize=7, color='#444444',
            )

        ax_g.spines['top'].set_visible(False)
        ax_g.spines['right'].set_visible(False)
        ax_g.spines['left'].set_visible(True)
        ax_g.spines['bottom'].set_linewidth(0.8)
        ax_g.set_title('')

        plt.tight_layout()
        plt.savefig('figures/FigG_Feature_Importance.png', dpi=600, bbox_inches='tight')
        plt.savefig('figures/FigG_Feature_Importance.pdf', bbox_inches='tight')
        print("  ✓ FigG_Feature_Importance.png / .pdf")
        plt.close()

        # ══════════════════════════════════════════════════════════════
        # Panel H — SHAP Beeswarm（右，不显示 y 轴标签）
        # ══════════════════════════════════════════════════════════════
        print("\n[Panel H] SHAP Beeswarm...")

        set_sci_style()
        mpl.rcParams.update(SHAP_RC)

        fig_h, ax_h = plt.subplots(figsize=(4, 3))

        shap.summary_plot(
            shap_values_pos.values,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=5,
            plot_size=None,
            color_bar=True,
            plot_type='dot',
        )

        ax_h = plt.gca()
        ax_h.set_title('')

        # ── y 轴：隐藏标签，保留刻度位置与 G 对齐 ──
        ax_h.set_yticklabels(['' for _ in ax_h.get_yticklabels()])
        ax_h.tick_params(axis='y', left=False)

        # ── colorbar 美化 ──
        if len(fig_h.axes) > 1:
            cbar_ax = fig_h.axes[-1]
            cbar_ax.tick_params(labelsize=7, width=0.4, size=3)
            cbar_ax.set_ylabel('Feature value', fontsize=8, labelpad=3)
            cbar_ax.spines[:].set_linewidth(0.4)

        ax_h.spines['top'].set_visible(False)
        ax_h.spines['right'].set_visible(False)
        ax_h.spines['left'].set_visible(False)
        ax_h.spines['bottom'].set_linewidth(1.0)
        ax_h.tick_params(axis='x', width=0.8, size=3.5, labelsize=8)
        ax_h.set_xlabel('SHAP value', fontsize=9, labelpad=5)
        ax_h.axvline(0, color='#999999', lw=0.8, ls='--', zorder=0)

        plt.tight_layout()
        plt.savefig('figures/FigH_SHAP_Beeswarm.png', dpi=600, bbox_inches='tight')
        plt.savefig('figures/FigH_SHAP_Beeswarm.pdf', bbox_inches='tight')
        print("  ✓ FigH_SHAP_Beeswarm.png / .pdf")
        plt.close()

        # ══════════════════════════════════════════════════════════════
        # Panel A — SHAP Summary Beeswarm（全特征）
        # ══════════════════════════════════════════════════════════════
        print("\n[Panel A] SHAP Summary Beeswarm...")

        fig_a, ax_a = plt.subplots(figsize=(5.5, 3))

        shap.summary_plot(
            shap_values_pos.values,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
            plot_size=None,
            color_bar=True,
        )

        ax_a = plt.gca()
        sci_ax(ax_a, xlabel='SHAP value', ylabel='')
        ax_a.set_title('SHAP Summary', fontsize=9, pad=8)

        if len(fig_a.axes) > 1:
            cbar_ax = fig_a.axes[-1]
            cbar_ax.tick_params(labelsize=6)
            cbar_ax.set_ylabel('Feature value', fontsize=7, labelpad=3)

        plt.tight_layout()
        plt.savefig('figures/Fig3A_SHAP_Summary.png', dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/Fig3A_SHAP_Summary.pdf', bbox_inches='tight')
        print("  ✓ Fig3A_SHAP_Summary.png/.pdf")
        plt.close()

        # ══════════════════════════════════════════════════════════════
        # Panel B — Feature Importance（Top 10）
        # ══════════════════════════════════════════════════════════════
        print("\n[Panel B] Feature Importance...")

        fig_b, ax_b = plt.subplots(figsize=(5.5, 3))

        top_n = min(10, len(importance_df))
        imp_plot = importance_df.head(top_n).iloc[::-1]

        bars = ax_b.barh(
            range(len(imp_plot)),
            imp_plot['SHAP_Importance'],
            color=C_BAR,
            edgecolor='white',
            linewidth=0.3,
            height=0.65,
            alpha=0.85,
        )

        ax_b.set_yticks(range(len(imp_plot)))
        ax_b.set_yticklabels(imp_plot['Feature'], fontsize=7)
        sci_ax(ax_b, xlabel='Mean |SHAP value|')
        ax_b.set_title('Feature Importance', fontsize=9, pad=8)

        x_max = imp_plot['SHAP_Importance'].max()
        for i, (bar, val) in enumerate(zip(bars, imp_plot['SHAP_Importance'])):
            ax_b.text(val + x_max * 0.025, i,
                      f'{val:.4f}', va='center', fontsize=6.5, color=C_ANNOT)

        plt.tight_layout()
        plt.savefig('figures/Fig3B_Feature_Importance.png', dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/Fig3B_Feature_Importance.pdf', bbox_inches='tight')
        print("  ✓ Fig3B_Feature_Importance.png/.pdf")
        plt.close()

        # ══════════════════════════════════════════════════════════════
        # Panel C — Waterfall (High-risk)
        # ══════════════════════════════════════════════════════════════
        print(f"\n[Panel C] Waterfall — High-risk (prob={high_risk_prob:.3f})...")

        fig_c = plt.figure(figsize=(5, 3))
        shap.waterfall_plot(shap_values_pos[high_risk_idx], show=False, max_display=15)

        all_ax_c = fig_c.get_axes()
        ax_c = all_ax_c[0]
        ax_c.set_title(
            f'High-risk sample (Predicted Probability = {high_risk_prob:.3f})',
            fontsize=10, pad=10
        )
        despine(ax_c)

        for a in all_ax_c:
            a.tick_params(labelsize=10)
            for txt in a.texts:
                txt.set_fontsize(10)

        if len(all_ax_c) > 2:
            for label in all_ax_c[2].get_xticklabels()[1:]:
                label.set_visible(False)

        plt.tight_layout()
        plt.savefig('figures/Fig3C_Waterfall_HighRisk.png', dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/Fig3C_Waterfall_HighRisk.pdf', bbox_inches='tight')
        print("  ✓ Fig3C_Waterfall_HighRisk.png/.pdf")
        plt.close()

        # ══════════════════════════════════════════════════════════════
        # Panel D — Waterfall (Low-risk)
        # ══════════════════════════════════════════════════════════════
        print(f"\n[Panel D] Waterfall — Low-risk (prob={low_risk_prob:.3f})...")

        fig_d = plt.figure(figsize=(5, 3))
        shap.waterfall_plot(shap_values_pos[low_risk_idx], show=False, max_display=15)

        all_ax_d = fig_d.get_axes()
        ax_d = all_ax_d[0]
        ax_d.set_title(
            f'Low-risk sample (Predicted Probability = {low_risk_prob:.3f})',
            fontsize=10, pad=10
        )
        despine(ax_d)

        for a in all_ax_d:
            a.tick_params(labelsize=10)
            for txt in a.texts:
                txt.set_fontsize(10)

        if len(all_ax_d) > 2:
            for label in all_ax_d[2].get_xticklabels()[1:]:
                label.set_visible(False)

        plt.tight_layout()
        plt.savefig('figures/Fig3D_Waterfall_LowRisk.png', dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/Fig3D_Waterfall_LowRisk.pdf', bbox_inches='tight')
        print("  ✓ Fig3D_Waterfall_LowRisk.png/.pdf")
        plt.close()

        # ------------------------------------------
        # 步骤 8b: 眼轴长 SHAP 依赖图 — Development vs External（并排）
        # ------------------------------------------
        print("\n[8b/14] 绘制 Axial-length SHAP Dependence (Development vs External)...")

        RNG_AL = np.random.default_rng(42)

        # ---------- 0. 稳健定位眼轴长特征 ----------
        def _locate_feature(feature_names, raw_key, keywords=(), display_fallback=None):
            """
            在 feature_names 中稳健定位特征，返回 (idx, label)。
            依次尝试：rename 后的标准标签 → 原始键本身 → 关键词模糊匹配（大小写不敏感）。
            """
            fn = [str(f) for f in feature_names]

            # (a) rename 后的展示标签
            label = display_fallback
            try:
                label = rename_feature(raw_key)
            except Exception:
                pass
            if label and label in fn:
                return fn.index(label), label

            # (b) 原始键本身（万一列表没被重命名）
            if raw_key in fn:
                return fn.index(raw_key), raw_key

            # (c) 关键词模糊匹配
            hits = [i for i, f in enumerate(fn)
                    if f.strip().lower() == raw_key.lower()
                    or any(k.lower() in f.lower() for k in keywords)]
            if len(hits) == 1:
                return hits[0], fn[hits[0]]
            if len(hits) > 1:                          # 多个候选时优先含首个关键词者
                for i in hits:
                    if keywords and keywords[0].lower() in fn[i].lower():
                        return i, fn[i]
                return hits[0], fn[hits[0]]

            raise ValueError(
                "❌ 无法在 feature_names 中定位眼轴长特征。\n"
                f"   rename_feature({raw_key!r}) = {label!r}\n"
                f"   现有特征名 = {fn}\n"
                "   → 请检查 AL_RAW_KEY 是否与建模时的原始列名一致。"
            )

        _al_idx, AL_LABEL = _locate_feature(
            feature_names, AL_RAW_KEY, keywords=('axial',),
            display_fallback='Axial length (mm)')
        print(f"  ✓ 眼轴长特征已定位: idx={_al_idx}, label={AL_LABEL!r}")

        # ---------- 1. 计算外部队列 SHAP 值 ----------
        _expl_type = type(explainer).__name__
        if "Tree" in _expl_type:
            _sv_ext = explainer(X_ext_df, check_additivity=False)
        elif "Linear" in _expl_type:
            _sv_ext = explainer(X_ext_df)
        else:
            _raw_ext = explainer.shap_values(X_ext_df)
            if isinstance(_raw_ext, list):
                _v, _b = _raw_ext[1], explainer.expected_value[1]
            else:
                _v, _b = _raw_ext, explainer.expected_value
            _sv_ext = shap.Explanation(values=_v, base_values=_b,
                                       data=X_ext_df.values,
                                       feature_names=list(feature_names))

        shap_values_ext = _sv_ext
        if len(_sv_ext.values.shape) == 3:
            shap_values_ext = shap.Explanation(
                values=_sv_ext.values[:, :, 1],
                base_values=(_sv_ext.base_values[:, 1]
                             if _sv_ext.base_values.ndim > 1 else _sv_ext.base_values),
                data=_sv_ext.data,
                feature_names=_sv_ext.feature_names
            )
        print(f"  ✓ 外部 SHAP 矩阵: {shap_values_ext.values.shape}")

        # ---------- 2. 取回 mm 尺度的眼轴长（建模已标准化，需用原始特征表）----------
        def _raw_al(df_raw, pos=None):
            """从未标准化的原始特征表中取 AL(mm)；pos 为需要对齐的行位置。"""
            if df_raw is None:
                return None
            d = pd.DataFrame(df_raw).reset_index(drop=True)
            col = next((c for c in (AL_RAW_KEY, AL_LABEL) if c in d.columns), None)
            if col is None:
                col = next((c for c in d.columns
                            if 'axial' in str(c).lower() or str(c).strip().lower() == 'al'),
                           None)
            if col is None:
                return None
            v = pd.to_numeric(d[col], errors='coerce').to_numpy(dtype=float)
            return v if pos is None else v[np.asarray(pos, dtype=int)]

        # 原始（未标准化）特征表 —— 来自 Model_Package 的 X_train / X_external
        _X_train_raw = X_TRAIN_RAW
        _X_ext_raw   = X_EXT_RAW

        al_dev = _raw_al(_X_train_raw, pos=X_sample.index.to_numpy())
        al_ext = _raw_al(_X_ext_raw)

        if al_dev is None or al_ext is None:
            print("  ⚠️ 未找到原始 AL 列，改用标准化尺度绘图（横轴单位非 mm）")
            al_dev = X_sample.iloc[:, _al_idx].to_numpy(dtype=float)
            al_ext = X_ext_df.iloc[:, _al_idx].to_numpy(dtype=float)
            AL_AXIS_LABEL = 'Axial length (standardised)'
        else:
            AL_AXIS_LABEL = 'Axial length (mm)'

        shap_al_dev = np.asarray(shap_values_pos.values[:, _al_idx], dtype=float)
        shap_al_ext = np.asarray(shap_values_ext.values[:, _al_idx], dtype=float)

        _m_dev = np.isfinite(al_dev) & np.isfinite(shap_al_dev)
        _m_ext = np.isfinite(al_ext) & np.isfinite(shap_al_ext)
        al_dev, shap_al_dev = al_dev[_m_dev], shap_al_dev[_m_dev]
        al_ext, shap_al_ext = al_ext[_m_ext], shap_al_ext[_m_ext]
        print(f"  ✓ 有效样本: development n={len(al_dev)}, external n={len(al_ext)}")

        # ---------- 3. 平滑、置信带与拐点 ----------
        def _fit_spline(x, y, s_factor=1.0):
            """按 x 去重加权后拟合平滑样条。"""
            xu, inv = np.unique(x, return_inverse=True)
            cnt = np.bincount(inv).astype(float)
            yu = np.bincount(inv, weights=y) / cnt
            if len(xu) < 5:
                return None
            k = 3 if len(xu) > 4 else 1
            s = s_factor * len(xu) * float(np.var(yu))
            return UnivariateSpline(xu, yu, w=np.sqrt(cnt), k=k, s=max(s, 1e-9))

        def _smooth(x, y, grid, s_factor=1.0):
            sp = _fit_spline(x, y, s_factor)
            return np.full_like(grid, np.nan, dtype=float) if sp is None else sp(grid)

        def _boot_band(x, y, grid, n_boot=N_BOOT_AL, s_factor=1.0):
            curves = np.full((n_boot, len(grid)), np.nan)
            n = len(x)
            for b in range(n_boot):
                ii = RNG_AL.integers(0, n, n)
                try:
                    curves[b] = _smooth(x[ii], y[ii], grid, s_factor)
                except Exception:
                    pass
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return (np.nanpercentile(curves, 2.5, axis=0),
                        np.nanpercentile(curves, 97.5, axis=0))

        def _inflection(x, y, grid, s_factor=1.0):
            """
            临床阈值 = 平滑 SHAP 曲线从负(保护)翻转为正(风险)的过零点。
            与独立脚本 "分箱均值 → 首次符号变号 → 线性插值" 方法一致,
            可让 Dev vs Ext 并排图与临床尺度反推图相互对齐。
            """
            sp = _fit_spline(x, y, s_factor)
            if sp is None:
                return np.nan
            smoothed = sp(grid)
            # 限定核心区,避免边缘外推伪影
            core = (grid >= np.percentile(x, 5)) & (grid <= np.percentile(x, 95))
            if core.sum() < 2:
                return np.nan
            gc, yc = grid[core], smoothed[core]

            # 首次负→正符号变换
            sign = np.sign(yc)
            sign[sign == 0] = 1
            crossings = np.where(np.diff(sign) > 0)[0]

            if len(crossings) == 0:
                # 全程未过零 → 用 |SHAP| 最小处兜底
                return float(gc[int(np.nanargmin(np.abs(yc)))])

            i = crossings[0]
            x1, x2, y1, y2 = gc[i], gc[i + 1], yc[i], yc[i + 1]
            if y2 == y1:
                return float(x1)
            return float(x1 - y1 * (x2 - x1) / (y2 - y1))

        _g_dev = np.linspace(np.percentile(al_dev, 1), np.percentile(al_dev, 99), 200)
        _g_ext = np.linspace(np.percentile(al_ext, 1), np.percentile(al_ext, 99), 200)

        curve_dev = _smooth(al_dev, shap_al_dev, _g_dev)
        curve_ext = _smooth(al_ext, shap_al_ext, _g_ext)
        lo_dev, hi_dev = _boot_band(al_dev, shap_al_dev, _g_dev)
        lo_ext, hi_ext = _boot_band(al_ext, shap_al_ext, _g_ext)

        infl_dev = _inflection(al_dev, shap_al_dev, _g_dev)
        infl_ext = _inflection(al_ext, shap_al_ext, _g_ext)
        print(f"  ✓ Model-derived inflection: development={infl_dev:.2f}, external={infl_ext:.2f}")

        # ---------- 4. 绘图（沿用主图风格：set_sci_style + SHAP_RC）----------
        set_sci_style()
        mpl.rcParams.update(SHAP_RC)

        fig_al, axes_al = plt.subplots(1, 2, figsize=(6.7, 2.9), sharey=True)

        _panels = [
            (axes_al[0], al_dev, shap_al_dev, _g_dev, curve_dev, lo_dev, hi_dev,
             infl_dev, COLOR_INTERNAL, 'Development cohort', ''),
            (axes_al[1], al_ext, shap_al_ext, _g_ext, curve_ext, lo_ext, hi_ext,
             infl_ext, COLOR_EXTERNAL, 'External validation cohort', ''),
        ]

        _all_shap = np.concatenate([shap_al_dev, shap_al_ext])
        _pad = 0.16 * (np.nanmax(_all_shap) - np.nanmin(_all_shap) + 1e-9)
        _ylim = (np.nanmin(_all_shap) - _pad, np.nanmax(_all_shap) + _pad)

        for ax, xv, yv, grid, curve, lo, hi, infl, col, ttl, tag in _panels:
            ax.axhline(0, color='#999999', lw=0.8, ls='--', zorder=0)

            ax.scatter(xv, yv, s=7, alpha=0.40, color=col,
                       edgecolors='none', zorder=1, rasterized=True)
            ax.fill_between(grid, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(grid, curve, color=col, lw=1.6, zorder=3)

            # 建模队列拐点在外部面板作浅灰参考线
            if tag == 'b' and np.isfinite(infl_dev):
                ax.axvline(infl_dev, color='#BBBBBB', lw=0.9, ls='-', zorder=1)

            if np.isfinite(infl):
                ax.axvline(infl, color='#444444', lw=1.0, ls=':', zorder=4)
                ax.annotate(f'SHAP threshold\n(neg → pos), {infl:.1f} mm',
                            xy=(infl, _ylim[1]), xytext=(4, -2),
                            textcoords='offset points',
                            ha='left', va='top', fontsize=7, color='#444444')

            # 底部 rug — 展示数据密度
            _rug_y = _ylim[0] + 0.02 * (_ylim[1] - _ylim[0])
            ax.plot(xv, np.full_like(xv, _rug_y), '|', color=col,
                    alpha=0.35, ms=3, mew=0.5, zorder=1)

            ax.set_ylim(*_ylim)
            ax.set_xlabel(AL_AXIS_LABEL, fontsize=9, labelpad=4)
            ax.set_title(f'{ttl} (n = {len(xv)})', fontsize=9, pad=6)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            ax.tick_params(axis='both', width=0.8, size=3.5, labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(-0.02, 1.10, tag, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='top', ha='right')

        axes_al[0].set_ylabel('SHAP value for axial length\n(log-odds)',
                              fontsize=9, labelpad=4)

        # 图例：development 拐点参考线
        if np.isfinite(infl_dev):
            _ref = Line2D([0], [0], color='#BBBBBB', lw=0.9,
                          label=f'Development threshold ({infl_dev:.1f} mm)')
            sci_legend(axes_al[1], handles=[_ref], loc='lower right', fontsize=7)

        plt.tight_layout()
        plt.savefig('figures/FigAL_Dependence_Dev_vs_Ext.png', dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/FigAL_Dependence_Dev_vs_Ext.pdf', bbox_inches='tight')
        print("  ✓ FigAL_Dependence_Dev_vs_Ext.png / .pdf")
        plt.close()

        # ---------- 5. 导出曲线数据（供审稿/复核）----------
        pd.concat([
            pd.DataFrame({'Cohort': 'Development', 'Axial_length': _g_dev,
                          'SHAP_smooth': curve_dev, 'CI_low': lo_dev, 'CI_high': hi_dev,
                          'SHAP_threshold_mm': infl_dev}),
            pd.DataFrame({'Cohort': 'External', 'Axial_length': _g_ext,
                          'SHAP_smooth': curve_ext, 'CI_low': lo_ext, 'CI_high': hi_ext,
                          'SHAP_threshold_mm': infl_ext}),
        ]).to_csv(f'{DATA_PATH}/AL_SHAP_Dependence_Curves_{TIMESTAMP}.csv', index=False)
        print(f"  ✓ 曲线数据已导出: {DATA_PATH}/AL_SHAP_Dependence_Curves_{TIMESTAMP}.csv")

    finally:
        # 恢复全局绘图样式，避免影响后续图形
        mpl.rcParams.update(_saved_rcParams)


if HAS_SHAP:
    try:
        run_shap_module()
    except FileNotFoundError as e:
        print(f"⚠️ 跳过 SHAP 模块（缺少文件）: {e}")
    except KeyError as e:
        print(f"⚠️ 跳过 SHAP 模块（数据包缺少字段 {e}）")
else:
    print("⚠️ 未安装 shap，已跳过 SHAP 模块。")


# ==========================================================================================
# 九、补充图 S1–S4（移植自 comprehensive_analysis.py 的 section1–section4 绘图部分）
# ==========================================================================================

print("\n" + "=" * 60)
print("📎 生成补充图 S1–S4 ...")
print("=" * 60)

# ---------------- 可调参数 ----------------
SUPP_ENABLE       = True          # 一键开关
SUPP_DIR          = 'figures'     # 输出目录（与正文图同级）
SUPP_N_BOOT_CAL   = 500           # 校准带 bootstrap 次数（正文面板 E 用 1000）
SUPP_N_BOOT_DCA   = 300           # DCA 置信带 bootstrap 次数
SUPP_DCA_XMAX     = 0.5           # DCA 横轴上限（与正文 F 一致）
SUPP_MODELS       = ['Plain_LR', 'XGBoost']     # S1–S3 并列比较的两个模型
SUPP_S4_MODELS    = ['XGB_5var_LASSO', 'XGB_10var_Full']

_SUPP_STATS_ROWS = []             # 汇总各图统计量，最后统一导出 CSV


# ---------------- 模型展示名（沿用 LABEL_MAP 的英文风格）----------------
MODEL_LABEL_MAP = {
    'Plain_LR':        'Logistic regression',
    'XGBoost':         'XGBoost',
    'XGB_5var_LASSO':  'XGBoost (5 variables)',
    'XGB_10var_Full':  'XGBoost (10 variables)',
    'LR_Ridge':        'Ridge logistic regression',
    'LR_ElasticNet':   'Elastic-net logistic regression',
    'RandomForest':    'Random forest',
    'LightGBM':        'LightGBM',
    'Stacking':        'Stacking ensemble',
}


def model_label(name):
    """模型代码名 → 出版展示名"""
    return MODEL_LABEL_MAP.get(str(name), str(name).replace('_', ' '))


# ---------------- 模型对比配色 / 标记 ----------------
#   蓝(#3C5488) = 队列「内部」、红(#E64B35) = 队列「外部」，语义已被正文占用；
#   模型对比统一使用 深蓝 → 青绿 → 红 → 橙 的顺序，队列信息由图题与文件名承担。
_MODEL_PALETTE = [COLORS_LIST[3], COLORS_LIST[2], COLORS_LIST[0], COLORS_LIST[4],
                  COLORS_LIST[5], COLORS_LIST[8]]
_MODEL_MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def model_style(names):
    """为一组模型分配稳定的颜色与散点标记"""
    return {n: {'color': _MODEL_PALETTE[i % len(_MODEL_PALETTE)],
                'marker': _MODEL_MARKERS[i % len(_MODEL_MARKERS)]}
            for i, n in enumerate(names)}


# ---------------- 通用小工具 ----------------
def _supp_save(save_path, dpi=DPI):
    """统一保存 PNG + PDF（与正文图一致）"""
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_path.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
    print(f"  ✓ 已保存: {save_path} / .pdf")
    plt.close()


def _find_latest(pattern):
    """按修改时间取最新的匹配文件（S 系列文件用的是自己的时间戳）"""
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def _youden_threshold(y_true, y_prob):
    """开发集 OOF 上的 Youden 阈值，锁定后用于外部（与 comprehensive_analysis 同口径）"""
    fpr, tpr, thr = roc_curve(np.asarray(y_true), np.asarray(y_prob))
    return round(float(thr[np.argmax(tpr - fpr)]), 2)


def _fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


# ==========================================================================================
# 9.1 通用绘图函数（校准 / DCA / 一致性 / 森林图）—— 均为本脚本风格
# ==========================================================================================
def plot_calibration_multi(series, save_path, n_bootstrap=SUPP_N_BOOT_CAL,
                           show_spike=True, note=None, figsize=(3.5, 3.5),
                           legend_loc='lower right', record_tag=None):
    """
    多曲线校准图 —— 与正文面板 E（plot_calibration_publication_v2）同款视觉语法，
    区别仅在于「一条曲线 = 一个模型」而非「一个队列」。

    series : list of dict(y=…, prob=…, label=…, color=…, marker=…)
    note   : 左上角小字注释（如样本量、事件数）
    """
    series = [s for s in series if s.get('y') is not None and len(s['y']) > 0]
    if not series:
        print("  ⚠️ 无可绘制数据，跳过")
        return None

    fig = plt.figure(figsize=figsize)
    if show_spike:
        gs = gridspec.GridSpec(2, 1, height_ratios=[5.5, 1], hspace=0.05,
                               left=0.12, right=0.95, top=0.93, bottom=0.08)
        ax_main  = fig.add_subplot(gs[0])
        ax_spike = fig.add_subplot(gs[1], sharex=ax_main)
    else:
        ax_main, ax_spike = fig.add_subplot(111), None

    np.random.seed(42)                     # 置信带可复现

    for ds in series:
        y = np.asarray(ds['y'], dtype=int)
        p = np.asarray(ds['prob'], dtype=float)
        col, mk = ds['color'], ds.get('marker', 'o')

        # --- A. 统计量（进图例 / 汇总 CSV）---
        slope, intercept = get_cali_stats(y, p)
        hl_stat, hl_pval = hosmer_lemeshow_test(y, p, n_groups=10)
        brier = brier_score_loss(y, p)
        oe    = y.sum() / p.sum() if p.sum() > 0 else np.nan

        print(f"    {ds['label']:<28} N={len(y):>4}  Events={int(y.sum()):>3} "
              f"({y.mean():.1%})  Slope={slope:.3f}  Intercept={intercept:.3f}  "
              f"Brier={brier:.4f}  O:E={oe:.3f}  H-L p={_fmt_p(hl_pval)}")

        _SUPP_STATS_ROWS.append({
            'Figure': record_tag or os.path.basename(save_path),
            'Series': ds['label'], 'N': len(y), 'Events': int(y.sum()),
            'Event_rate': y.mean(), 'Cal_slope': slope, 'Cal_intercept': intercept,
            'Brier': brier, 'OE': oe, 'HL_chi2': hl_stat, 'HL_p': hl_pval,
        })

        # --- B. 自适应等频分 bin + Wilson CI ---
        min_per_bin = max(15, int(len(y) * 0.04))
        bins = adaptive_calibration_bins(y, p, min_samples=min_per_bin, max_bins=10)

        if len(bins['pred']) > 0:
            size_min, size_max = 15, 45
            if bins['n'].max() > bins['n'].min():
                sizes = size_min + (size_max - size_min) * \
                    (bins['n'] - bins['n'].min()) / (bins['n'].max() - bins['n'].min())
            else:
                sizes = np.full(len(bins['n']), (size_min + size_max) / 2)

            ax_main.errorbar(
                bins['pred'], bins['true'],
                yerr=[bins['true'] - bins['ci_low'], bins['ci_up'] - bins['true']],
                fmt='none', ecolor=col, elinewidth=0.6,
                capsize=1.5, capthick=0.6, alpha=0.45, zorder=3)

            ax_main.scatter(bins['pred'], bins['true'], s=sizes, marker=mk,
                            facecolors=col, edgecolors=col,
                            linewidths=0.7, alpha=0.85, zorder=4)

        # --- C. logistic 重校准拟合线（图例带统计量）---
        x_fit, y_fit, _, _ = logistic_calibration_curve(y, p)
        ax_main.plot(x_fit, y_fit, color=col, lw=1.5, alpha=0.85, zorder=5,
                     label=f"{ds['label']} (Slope = {slope:.2f}, Brier = {brier:.3f})")

        # --- D. bootstrap 95% 带 ---
        try:
            x_ci, ci_low, ci_up = bootstrap_logistic_cal_ci(
                y, p, n_bootstrap=n_bootstrap)
            ax_main.fill_between(x_ci, ci_low, ci_up, color=col, alpha=0.15, zorder=1)
        except Exception as e:
            print(f"    ⚠️ {ds['label']} bootstrap 带失败: {e}")

    # --- E. 理想校准线 ---
    ax_main.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--',
                 zorder=2, label='Ideal')

    ax_main.set_ylabel('Observed proportion')
    ax_main.set_xlim([-0.02, 1.02])
    ax_main.set_ylim([-0.02, 1.02])
    ax_main.set_aspect('equal')
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    if note:
        ax_main.text(0.03, 0.97, note, transform=ax_main.transAxes, fontsize=6.5,
                     color='#555555', va='top', ha='left')
    sci_legend(ax_main, loc=legend_loc, fontsize=6.5)

    # --- F. 底部 spike 直方图 ---
    if show_spike:
        ax_main.set_xlabel('')
        ax_main.tick_params(axis='x', labelbottom=False)

        spike_bins = np.linspace(0, 1, 41)
        bw = spike_bins[1] - spike_bins[0]
        offsets = np.linspace(-0.002, 0.002, max(len(series), 2))

        for k, ds in enumerate(series):
            y = np.asarray(ds['y'], dtype=int)
            p = np.asarray(ds['prob'], dtype=float)
            centers = (spike_bins[:-1] + spike_bins[1:]) / 2 + offsets[k]

            counts_pos, _ = np.histogram(p[y == 1], bins=spike_bins)
            ax_spike.bar(centers, counts_pos, width=bw * 0.45,
                         color=ds['color'], alpha=0.6, edgecolor='none')

            counts_neg, _ = np.histogram(p[y == 0], bins=spike_bins)
            neg_scale = max(counts_pos.max(), 1) / max(counts_neg.max(), 1) * 0.8
            ax_spike.bar(centers, -counts_neg * neg_scale, width=bw * 0.45,
                         color=ds['color'], alpha=0.25, edgecolor='none')

        ax_spike.axhline(y=0, color='#888888', linewidth=0.6)
        ax_spike.set_xlabel('Predicted probability')
        ax_spike.set_xlim([-0.02, 1.02])
        ax_spike.spines['top'].set_visible(False)
        ax_spike.spines['right'].set_visible(False)
        ax_spike.set_yticks([])
        ax_spike.text(0.01, 0.92, 'Events', transform=ax_spike.transAxes,
                      fontsize=6.5, color='#555555', va='top', fontstyle='italic')
        ax_spike.text(0.01, 0.08, 'Non-events', transform=ax_spike.transAxes,
                      fontsize=6.5, color='#555555', va='bottom', fontstyle='italic')
    else:
        ax_main.set_xlabel('Predicted probability')

    _supp_save(save_path)
    return fig


def plot_dca_multi(series, save_path, locked_threshold=None, xmax=SUPP_DCA_XMAX,
                   n_bootstrap=SUPP_N_BOOT_DCA, figsize=(3.5, 5), record_tag=None):
    """
    多模型决策曲线 —— 与正文面板 F 同款「主面板 + 每 1000 人高风险数」双层结构。

    series : list of dict(y=…, prob=…, label=…, color=…)
             同一张图内各条曲线应来自同一队列（Treat all 取第一条的结局率）
    """
    series = [s for s in series if s.get('y') is not None and len(s['y']) > 0]
    if not series:
        print("  ⚠️ 无可绘制数据，跳过")
        return None

    threshs = np.arange(0, 1.01, 0.01)
    prevalence = np.asarray(series[0]['y'], dtype=int).mean()
    all_nb = prevalence - (1 - prevalence) * (threshs / (1 - threshs + 1e-10))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5], hspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_num  = fig.add_subplot(gs[1], sharex=ax_main)

    y_max_candidates = [all_nb[threshs <= xmax].max()]

    for ds in series:
        y = np.asarray(ds['y'], dtype=int)
        p = np.asarray(ds['prob'], dtype=float)
        try:
            nb_mean, nb_lo, nb_hi = bootstrap_net_benefit(
                y, p, threshs, n_bootstrap=n_bootstrap)
        except Exception as e:
            print(f"    ⚠️ {ds['label']} Bootstrap 失败: {e}，改用点估计")
            nb_mean = np.array([calculate_net_benefit(y, p, t) for t in threshs])
            nb_lo = nb_hi = nb_mean.copy()

        ax_main.plot(threshs, nb_mean, color=ds['color'], lw=1.5,
                     alpha=0.9, label=ds['label'])
        ax_main.fill_between(threshs, nb_lo, nb_hi, color=ds['color'], alpha=0.15)

        n_high = [calculate_n_high_risk(p, t, len(y)) for t in threshs]
        ax_num.plot(threshs, n_high, color=ds['color'], lw=1.5, alpha=0.8)

        y_max_candidates.append(nb_mean[threshs <= xmax].max())

        # 锁定阈值处的净获益，写入汇总表
        nb_at_lock = (calculate_net_benefit(y, p, locked_threshold)
                      if locked_threshold is not None else np.nan)
        _SUPP_STATS_ROWS.append({
            'Figure': record_tag or os.path.basename(save_path),
            'Series': ds['label'], 'N': len(y), 'Events': int(y.sum()),
            'Event_rate': y.mean(), 'Net_benefit_at_locked_threshold': nb_at_lock,
            'Locked_threshold': locked_threshold,
        })

    ax_main.plot(threshs, all_nb, color='#999999', ls='--', lw=1.0, alpha=0.7,
                 label=f'Treat all ({prevalence:.1%})')
    ax_main.axhline(y=0, color='black', lw=0.8, ls='-.', alpha=0.7, label='Treat none')
    if locked_threshold is not None:
        ax_main.axvline(locked_threshold, color='#333333', ls='--', lw=1.2, alpha=0.9,
                        label=f'Locked threshold = {locked_threshold:.3f}')

    ax_main.set_xlim([0, xmax])
    ax_main.set_ylim([-0.05, max(y_max_candidates) + 0.05])
    ax_main.set_ylabel('Net benefit')
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    sci_legend(ax_main, loc='upper right', fontsize=6.5,
               handlelength=1.8, handletextpad=0.5, labelspacing=0.35, borderpad=0.4)

    ax_num.set_ylabel('Predicted risk ≥\nthreshold per 1000', fontsize=7)
    ax_num.set_xlabel('Threshold probability')
    ax_num.set_ylim([0, 1000])
    ax_num.spines['top'].set_visible(False)
    ax_num.spines['right'].set_visible(False)

    plt.tight_layout()
    _supp_save(save_path)
    return fig


def plot_agreement_scatter(prob_x, prob_y, y_true, xlabel, ylabel,
                           save_path, threshold=None, figsize=(3.5, 3.5),
                           record_tag=None):
    """
    两模型预测概率一致性散点 —— 事件/非事件分色，附 Pearson / Spearman / κ。
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import cohen_kappa_score

    px = np.asarray(prob_x, dtype=float)
    py = np.asarray(prob_y, dtype=float)
    yt = np.asarray(y_true, dtype=int)

    r_p, _ = pearsonr(px, py)
    r_s, _ = spearmanr(px, py)
    if threshold is not None:
        kappa = cohen_kappa_score((px >= threshold).astype(int),
                                  (py >= threshold).astype(int))
        agree = float(np.mean((px >= threshold) == (py >= threshold)))
    else:
        kappa, agree = np.nan, np.nan

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--', zorder=1)

    ax.scatter(px[yt == 0], py[yt == 0], s=10, marker='o',
               facecolors='none', edgecolors='#9E9E9E', linewidths=0.5,
               alpha=0.6, zorder=2, label='Non-events')
    ax.scatter(px[yt == 1], py[yt == 1], s=14, marker='o',
               facecolors=C_EXT, edgecolors='white', linewidths=0.4,
               alpha=0.85, zorder=3, label='Events')

    if threshold is not None:
        ax.axvline(threshold, color='#333333', ls=':', lw=0.8, alpha=0.7, zorder=1)
        ax.axhline(threshold, color='#333333', ls=':', lw=0.8, alpha=0.7, zorder=1)

    stats_txt = (f"Pearson r = {r_p:.3f}\nSpearman ρ = {r_s:.3f}"
                 + (f"\nCohen κ = {kappa:.3f}\nAgreement = {agree:.1%}"
                    if threshold is not None else ""))
    ax.text(0.04, 0.96, stats_txt, transform=ax.transAxes, fontsize=6.5,
            va='top', ha='left', color='#333333',
            bbox=dict(boxstyle='square,pad=0.35', facecolor='white',
                      edgecolor='#999999', linewidth=0.6))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sci_legend(ax, loc='lower right', fontsize=6.5)

    _SUPP_STATS_ROWS.append({
        'Figure': record_tag or os.path.basename(save_path),
        'Series': f'{xlabel} vs {ylabel}', 'N': len(yt), 'Events': int(yt.sum()),
        'Pearson_r': r_p, 'Spearman_rho': r_s, 'Cohen_kappa': kappa,
        'Classification_agreement': agree, 'Locked_threshold': threshold,
    })

    plt.tight_layout()
    _supp_save(save_path)
    return fig


def plot_risk_stratification_bars(series, save_path, n_groups=5,
                                  figsize=(4.2, 3.2), record_tag=None):
    """
    五分位风险分层：各层「观察发生率（Wilson 95% CI）」vs「平均预测风险」。
    series : list of dict(y=…, prob=…, label=…, color=…)
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)

    for k, ds in enumerate(series):
        y = np.asarray(ds['y'], dtype=int)
        p = np.asarray(ds['prob'], dtype=float)
        cuts = np.unique(np.percentile(p, np.linspace(0, 100, n_groups + 1)[1:-1]))
        grp = np.digitize(p, cuts)

        obs, lo_err, hi_err, pred, xs = [], [], [], [], []
        for g in range(n_groups):
            m = grp == g
            if m.sum() == 0:
                continue
            ev, n = int(y[m].sum()), int(m.sum())
            lo, hi = _wilson_ci(ev, n)
            obs.append(ev / n)
            lo_err.append(ev / n - lo)
            hi_err.append(hi - ev / n)
            pred.append(p[m].mean())
            xs.append(g)

            _SUPP_STATS_ROWS.append({
                'Figure': record_tag or os.path.basename(save_path),
                'Series': ds['label'], 'Risk_group': g + 1, 'N': n, 'Events': ev,
                'Observed_risk': ev / n, 'Mean_predicted_risk': p[m].mean(),
                'CI_low': lo, 'CI_high': hi,
            })

        xs = np.asarray(xs, dtype=float) + (k - (n_series - 1) / 2) * width
        ax.bar(xs, obs, width=width * 0.85, color=ds['color'], alpha=0.65,
               edgecolor=ds['color'], linewidth=0.6,
               label=f"{ds['label']} — observed")
        ax.errorbar(xs, obs, yerr=[lo_err, hi_err], fmt='none', ecolor='#444444',
                    elinewidth=0.6, capsize=1.8, capthick=0.6, zorder=4)
        ax.plot(xs, pred, marker='D', ms=3.5, ls='none',
                markerfacecolor='white', markeredgecolor=ds['color'],
                markeredgewidth=0.9, zorder=5,
                label=f"{ds['label']} — predicted")

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels([f'Q{i+1}' for i in range(n_groups)])
    ax.set_xlabel('Predicted-risk quintile')
    ax.set_ylabel('Event rate')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)        # 顶部留白给图例
    sci_legend(ax, loc='upper left', fontsize=6.5,
               ncol=2 if n_series > 1 else 1, columnspacing=1.0)

    plt.tight_layout()
    _supp_save(save_path)
    return fig


def plot_forest_metrics(df, save_path, metrics=None, figsize=(4.6, 4.2),
                        title_note=None, record_tag=None):
    """
    指标森林图：每行一个指标，同一行内并排展示各模型的点估计 + 95% CI。
    df 需含列 Model 及 <metric> / <metric>_CI_lo / <metric>_CI_hi
    """
    metrics = metrics or ['AUROC', 'AP', 'Brier', 'OE',
                          'Sensitivity', 'Specificity', 'PPV', 'NPV']
    metrics = [m for m in metrics if m in df.columns]
    models  = list(df['Model'].unique())
    style   = model_style(models)

    fig, ax = plt.subplots(figsize=figsize)
    n_m = len(models)
    offs = np.linspace(0.22, -0.22, n_m) if n_m > 1 else np.array([0.0])

    for mi, mname in enumerate(models):
        row = df[df['Model'] == mname].iloc[0]
        col = style[mname]['color']
        ys, xs, lo_e, hi_e = [], [], [], []
        for j, met in enumerate(metrics):
            v = row.get(met, np.nan)
            if pd.isna(v):
                continue
            lo = row.get(f'{met}_CI_lo', np.nan)
            hi = row.get(f'{met}_CI_hi', np.nan)
            ys.append(len(metrics) - 1 - j + offs[mi])
            xs.append(float(v))
            lo_e.append(float(v) - float(lo) if not pd.isna(lo) else 0.0)
            hi_e.append(float(hi) - float(v) if not pd.isna(hi) else 0.0)

        ax.errorbar(xs, ys, xerr=[lo_e, hi_e], fmt=style[mname]['marker'],
                    ms=4, color=col, ecolor=col, elinewidth=0.9,
                    capsize=2, capthick=0.7, lw=0, alpha=0.9,
                    label=model_label(mname))

        for x, yy in zip(xs, ys):
            ax.text(x, yy + 0.16, f'{x:.3f}', ha='center', va='bottom',
                    fontsize=6, color='#555555')

    ax.set_yticks(np.arange(len(metrics))[::-1])
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Estimate (95% CI)')

    # 自适应横轴：按实际取值范围留白，避免大片空白压缩可读性
    lo_all, hi_all = [], []
    for met in metrics:
        for _, r in df.iterrows():
            v = r.get(met, np.nan)
            if pd.isna(v):
                continue
            lo_all.append(min(float(v), float(r.get(f'{met}_CI_lo', v))
                              if not pd.isna(r.get(f'{met}_CI_lo', np.nan)) else float(v)))
            hi_all.append(max(float(v), float(r.get(f'{met}_CI_hi', v))
                              if not pd.isna(r.get(f'{met}_CI_hi', np.nan)) else float(v)))
    if lo_all:
        pad = max(0.05, (max(hi_all) - min(lo_all)) * 0.12)
        ax.set_xlim(max(-0.02, min(lo_all) - pad), min(1.08, max(hi_all) + pad))
    else:
        ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-1.15, len(metrics) - 0.45)        # 底部留一行空间给图例
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    if title_note:
        ax.text(0.0, 1.02, title_note, transform=ax.transAxes, fontsize=7,
                color='#555555', va='bottom', ha='left')
    sci_legend(ax, loc='lower right', fontsize=6.5)

    plt.tight_layout()
    _supp_save(save_path)
    return fig


# ==========================================================================================
# 9.2 数据装载：逐样本预测概率（只读盘，不重拟合）
# ==========================================================================================
def load_supp_model_probs(model_names=SUPP_MODELS):
    """
    返回 (probs, y_int_s, y_ext_s)
      probs = {model: {'oof': ndarray, 'ext': ndarray, 'threshold': float}}

    优先级：
      1) HeadToHead_Probs_*.pkl                       ← 建模脚本 3.5 节直接导出
      2) Model_Package 的 cv_results[m]['prob_oof']
         + all_trained_models[m].predict_proba(X_external)
    """
    # ---- 路径 1：Head-to-Head 概率包 ----
    h2h_path = _find_latest(f"{DATA_PATH}/HeadToHead_Probs_*.pkl")
    if h2h_path:
        try:
            h2h = joblib.load(h2h_path)
            y_i = np.asarray(h2h['y_internal'], dtype=int)
            y_e = np.asarray(h2h['y_external'], dtype=int)
            probs = {}
            for m in model_names:
                if f'{m}_oof_probs' in h2h and f'{m}_ext_probs' in h2h:
                    oof = np.asarray(h2h[f'{m}_oof_probs'], dtype=float)
                    ext = np.asarray(h2h[f'{m}_ext_probs'], dtype=float)
                    probs[m] = {'oof': oof, 'ext': ext,
                                'threshold': _youden_threshold(y_i, oof)}
            if len(probs) >= 1:
                print(f"  ✓ 概率来源: {os.path.basename(h2h_path)} "
                      f"（{', '.join(probs)}）")
                return probs, y_i, y_e
        except Exception as e:
            print(f"  ⚠️ 读取 HeadToHead_Probs 失败（{e}），改用模型包回退方案")

    # ---- 路径 2：模型包回退 ----
    cvres  = full_package.get('cv_results', {}) or {}
    models = full_package.get('all_trained_models', {}) or {}
    if X_EXT_RAW is None or Y_TRAIN is None:
        raise FileNotFoundError("模型包缺少原始特征表/标签，无法回退计算 S1–S3 概率")

    y_i = np.asarray(Y_TRAIN, dtype=int)
    y_e = np.asarray(Y_EXT_PKG if Y_EXT_PKG is not None else y_ext, dtype=int)

    probs = {}
    for m in model_names:
        if m not in cvres or 'prob_oof' not in cvres[m] or m not in models:
            print(f"  ⚠️ 模型包中缺少 {m} 的 OOF 概率或已训练管道，跳过")
            continue
        oof = np.asarray(cvres[m]['prob_oof'], dtype=float)
        ext = models[m].predict_proba(X_EXT_RAW)[:, 1]
        probs[m] = {'oof': oof, 'ext': ext,
                    'threshold': _youden_threshold(y_i, oof)}
    print(f"  ✓ 概率来源: Model_Package（{', '.join(probs) if probs else '无'}）")
    return probs, y_i, y_e


# ==========================================================================================
# 9.3 S1 — Logistic regression vs XGBoost 并列比较
# ==========================================================================================
def run_supp_S1(probs, y_i, y_e):
    print("\n[S1] Logistic regression vs XGBoost 并列比较 ...")
    names = [m for m in SUPP_MODELS if m in probs]
    if len(names) < 1:
        print("  ⚠️ 无可用模型，跳过 S1")
        return
    style = model_style(names)
    lock  = probs[names[0]]['threshold']

    for tag, key, y_arr, cohort in [('Internal', 'oof', y_i, 'Internal validation (OOF)'),
                                    ('External', 'ext', y_e, 'External validation')]:
        series = [{'y': y_arr, 'prob': probs[m][key], 'label': model_label(m),
                   'color': style[m]['color'], 'marker': style[m]['marker']}
                  for m in names]

        # --- 校准曲线 ---
        print(f"  · S1 校准曲线 — {cohort}")
        plot_calibration_multi(
            series, f'{SUPP_DIR}/S1_Calibration_{tag}.png',
            note=f'{cohort}\nN = {len(y_arr)}, events = {int(np.sum(y_arr))}',
            record_tag=f'S1_Calibration_{tag}')

        # --- 决策曲线 ---
        print(f"  · S1 决策曲线 — {cohort}")
        plot_dca_multi(
            [{k: v for k, v in s.items() if k != 'marker'} for s in series],
            f'{SUPP_DIR}/S1_DCA_{tag}.png',
            locked_threshold=lock, record_tag=f'S1_DCA_{tag}')

    # --- 一致性散点（需要恰好两个模型）---
    if len(names) >= 2:
        m1, m2 = names[0], names[1]
        for tag, key, y_arr in [('Internal', 'oof', y_i), ('External', 'ext', y_e)]:
            print(f"  · S1 一致性散点 — {tag}")
            plot_agreement_scatter(
                probs[m1][key], probs[m2][key], y_arr,
                xlabel=f'{model_label(m1)} predicted probability',
                ylabel=f'{model_label(m2)} predicted probability',
                save_path=f'{SUPP_DIR}/S1_Agreement_{tag}.png',
                threshold=probs[m1]['threshold'],
                record_tag=f'S1_Agreement_{tag}')


# ==========================================================================================
# 9.4 S2 — 外部验证校准汇总
# ==========================================================================================
def run_supp_S2(probs, y_e):
    print("\n[S2] 外部验证校准汇总 ...")
    names = [m for m in SUPP_MODELS if m in probs]
    if not names:
        print("  ⚠️ 无可用模型，跳过 S2")
        return
    style = model_style(names)

    series = [{'y': y_e, 'prob': probs[m]['ext'], 'label': model_label(m),
               'color': style[m]['color'], 'marker': style[m]['marker']}
              for m in names]

    print("  · S2 外部校准曲线（含 bootstrap 95% 带）")
    plot_calibration_multi(
        series, f'{SUPP_DIR}/S2_External_Calibration.png',
        n_bootstrap=max(SUPP_N_BOOT_CAL, 1000),
        note=f'External validation\nN = {len(y_e)}, events = {int(np.sum(y_e))}',
        record_tag='S2_External_Calibration')

    print("  · S2 五分位风险分层（观察 vs 预测）")
    plot_risk_stratification_bars(
        series, f'{SUPP_DIR}/S2_External_RiskStratification.png',
        n_groups=5, record_tag='S2_External_RiskStratification')


# ==========================================================================================
# 9.5 S3 — 完整病例敏感性分析（外部队列）
# ==========================================================================================
def run_supp_S3(probs):
    print("\n[S3] 完整病例敏感性分析（外部队列）...")

    models = full_package.get('all_trained_models', {}) or {}
    fnames = full_package.get('feature_names', {}) or {}
    all_features = (list(fnames.get('log', [])) + list(fnames.get('numeric', []))
                    + list(fnames.get('categorical', [])))

    if X_EXT_RAW is None or not all_features or not models:
        print("  ⚠️ 模型包缺少原始外部特征表 / 特征清单 / 已训练管道，跳过 S3")
        return

    feats = [f for f in all_features if f in X_EXT_RAW.columns]
    complete_mask = X_EXT_RAW[feats].notnull().all(axis=1)
    y_e_pkg = np.asarray(Y_EXT_PKG if Y_EXT_PKG is not None else y_ext, dtype=int)

    X_cc = X_EXT_RAW.loc[complete_mask]
    y_cc = y_e_pkg[np.asarray(complete_mask)]

    n_total, n_cc = len(X_EXT_RAW), int(complete_mask.sum())
    print(f"  外部队列: {n_total} 例 | 完整病例: {n_cc} ({n_cc/n_total:.1%}) | "
          f"含缺失: {n_total-n_cc} ({1-n_cc/n_total:.1%})")
    print(f"  完整病例事件数: {int(y_cc.sum())} ({y_cc.mean():.1%})")
    print("  ⚠️ 完整病例分析仅为敏感性分析；若缺失非完全随机，删除缺失病例可能引入选择偏倚")

    names = [m for m in SUPP_MODELS if m in models and m in probs]
    if not names or len(y_cc) < 20 or len(np.unique(y_cc)) < 2:
        print("  ⚠️ 完整病例样本不足或模型缺失，跳过 S3")
        return
    style = model_style(names)

    series = []
    for m in names:
        p_cc = models[m].predict_proba(X_cc)[:, 1]
        series.append({'y': y_cc, 'prob': p_cc, 'label': model_label(m),
                       'color': style[m]['color'], 'marker': style[m]['marker']})

    note = (f'External complete cases\nN = {n_cc}/{n_total} ({n_cc/n_total:.0%}), '
            f'events = {int(y_cc.sum())}')

    print("  · S3 校准曲线")
    plot_calibration_multi(series, f'{SUPP_DIR}/S3_CompleteCases_Calibration.png',
                           note=note, record_tag='S3_CompleteCases_Calibration')

    print("  · S3 决策曲线")
    plot_dca_multi([{k: v for k, v in s.items() if k != 'marker'} for s in series],
                   f'{SUPP_DIR}/S3_CompleteCases_DCA.png',
                   locked_threshold=probs[names[0]]['threshold'],
                   record_tag='S3_CompleteCases_DCA')


# ==========================================================================================
# 9.6 S4 — 5 变量 vs 10 变量 XGBoost 敏感性分析
# ==========================================================================================
def _load_s4_probs():
    """兼容嵌套 dict 与扁平键两种导出格式；找不到返回 None"""
    path = _find_latest(f"{DATA_PATH}/S4_Sensitivity_Probs_*.pkl")
    if not path:
        return None
    try:
        pkg = joblib.load(path)
    except Exception as e:
        print(f"  ⚠️ 读取 {os.path.basename(path)} 失败: {e}")
        return None

    y_i = np.asarray(pkg['y_internal'], dtype=int)
    y_e = np.asarray(pkg['y_external'], dtype=int)
    probs = {}
    for m in SUPP_S4_MODELS:
        if isinstance(pkg.get(m), dict):
            d = pkg[m]
            probs[m] = {'oof': np.asarray(d['oof_probs'], dtype=float),
                        'ext': np.asarray(d['ext_probs'], dtype=float),
                        'threshold': float(d.get('threshold', np.nan))}
        elif f'{m}_oof_probs' in pkg:
            oof = np.asarray(pkg[f'{m}_oof_probs'], dtype=float)
            probs[m] = {'oof': oof,
                        'ext': np.asarray(pkg[f'{m}_ext_probs'], dtype=float),
                        'threshold': _youden_threshold(y_i, oof)}
    if not probs:
        return None
    print(f"  ✓ 概率来源: {os.path.basename(path)}")
    return probs, y_i, y_e


def run_supp_S4():
    print("\n[S4] 5 变量 vs 10 变量 XGBoost 敏感性分析 ...")

    # ---------- 路径 A：有逐样本概率包 → 校准 + DCA ----------
    loaded = _load_s4_probs()
    if loaded is not None:
        probs, y_i, y_e = loaded
        names = [m for m in SUPP_S4_MODELS if m in probs]
        style = model_style(names)
        lock  = probs[names[0]]['threshold']

        for tag, key, y_arr, cohort in [
                ('Internal', 'oof', y_i, 'Internal validation (OOF)'),
                ('External', 'ext', y_e, 'External validation')]:
            series = [{'y': y_arr, 'prob': probs[m][key], 'label': model_label(m),
                       'color': style[m]['color'], 'marker': style[m]['marker']}
                      for m in names]

            print(f"  · S4 校准曲线 — {cohort}")
            plot_calibration_multi(
                series, f'{SUPP_DIR}/S4_Sensitivity_Calibration_{tag}.png',
                note=f'{cohort}\nN = {len(y_arr)}, events = {int(np.sum(y_arr))}',
                record_tag=f'S4_Sensitivity_Calibration_{tag}')

            print(f"  · S4 决策曲线 — {cohort}")
            plot_dca_multi(
                [{k: v for k, v in s.items() if k != 'marker'} for s in series],
                f'{SUPP_DIR}/S4_Sensitivity_DCA_{tag}.png',
                locked_threshold=lock if np.isfinite(lock) else None,
                record_tag=f'S4_Sensitivity_DCA_{tag}')
    else:
        print("  ℹ️ 未找到 S4_Sensitivity_Probs_*.pkl（10 变量模型未持久化）")
        print("     → 仅绘制 S4E 森林图；如需校准/决策曲线，请按本节顶部注释在")
        print("       comprehensive_analysis.py 的 section4 末尾补一行 joblib.dump")

    # ---------- 路径 B：森林图（只需 comprehensive_analysis 的 CSV）----------
    csv_path = _find_latest(f"{DATA_PATH}/S4_Sensitivity_5v10_AllMetrics_*.csv")
    if not csv_path:
        print("  ⚠️ 未找到 S4_Sensitivity_5v10_AllMetrics_*.csv，跳过森林图")
        return
    try:
        sens_df = pd.read_csv(csv_path)
        ext_df  = sens_df[sens_df['Dataset'] == 'External'].copy()
        if ext_df.empty:
            print("  ⚠️ CSV 中无 External 行，跳过森林图")
            return
        print(f"  · S4 外部指标森林图（数据: {os.path.basename(csv_path)}）")
        plot_forest_metrics(
            ext_df, f'{SUPP_DIR}/S4_Sensitivity_Forest_External.png',
            title_note='External validation — point estimate (95% CI)',
            record_tag='S4_Sensitivity_Forest_External')

        # Δ 指标打印（与原脚本一致的结论口径）
        try:
            r5  = ext_df[ext_df['Model'] == 'XGB_5var_LASSO'].iloc[0]
            r10 = ext_df[ext_df['Model'] == 'XGB_10var_Full'].iloc[0]
            d_auc = r10['AUROC'] - r5['AUROC']
            d_ap  = r10['AP'] - r5['AP']
            print(f"    ΔAUROC = {d_auc:+.4f} | ΔAP = {d_ap:+.4f} | "
                  f"ΔBrier = {r10['Brier'] - r5['Brier']:+.4f}")
            print("    ✅ 全变量模型无明显改善 → 支持简约五变量模型"
                  if (d_auc < 0.02 and d_ap < 0.02)
                  else "    ⚠️ 全变量模型有潜在改善 → 建议重新审视变量筛选策略")
        except Exception:
            pass
    except Exception as e:
        print(f"  ⚠️ 森林图绘制失败: {e}")


# ==========================================================================================
# 9.7 补充图模块入口
# ==========================================================================================
def run_supplementary_S1_S4():
    set_sci_style()                       # 确保不受 SHAP 局部样式影响
    os.makedirs(SUPP_DIR, exist_ok=True)

    try:
        probs, y_i, y_e = load_supp_model_probs()
    except Exception as e:
        print(f"  ⚠️ 逐样本概率装载失败（{e}），S1–S3 跳过")
        probs, y_i, y_e = {}, None, None

    if probs:
        try:
            run_supp_S1(probs, y_i, y_e)
        except Exception as e:
            print(f"  ⚠️ S1 出错: {e}")
        try:
            run_supp_S2(probs, y_e)
        except Exception as e:
            print(f"  ⚠️ S2 出错: {e}")
        try:
            run_supp_S3(probs)
        except Exception as e:
            print(f"  ⚠️ S3 出错: {e}")

    try:
        run_supp_S4()
    except Exception as e:
        print(f"  ⚠️ S4 出错: {e}")

    # ---- 汇总统计量 CSV（图中所有数字的溯源表）----
    if _SUPP_STATS_ROWS:
        try:
            out_csv = f'{DATA_PATH}/Supp_S1S4_Plot_Stats_{TIMESTAMP}.csv'
            pd.DataFrame(_SUPP_STATS_ROWS).to_csv(
                out_csv, index=False, encoding='utf-8-sig')
            print(f"\n  ✓ 补充图统计量汇总已导出: {out_csv}")
        except Exception as e:
            print(f"  ⚠️ 统计量汇总导出失败: {e}")


if SUPP_ENABLE:
    try:
        run_supplementary_S1_S4()
    except FileNotFoundError as e:
        print(f"⚠️ 跳过补充图 S1–S4（缺少文件）: {e}")
    except Exception as e:
        print(f"⚠️ 补充图 S1–S4 模块出错: {e}")
else:
    print("ℹ️ SUPP_ENABLE = False，已跳过补充图 S1–S4。")




# ==========================================================================================
# 十、补充图 S5 —— 术中变量（手术时长 / PFCL / 联合白内障手术）的增量价值评估
# ==========================================================================================

print("\n" + "=" * 60)
print("📐 生成补充图 S5 —— 术中变量增量价值（审稿意见专用）...")
print("=" * 60)

from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import norm as _norm, chi2 as _chi2

# ---------------- 可调参数 ----------------
INCR_ENABLE      = True                 # 一键开关
INCR_DIR         = 'figures'            # 图与表输出目录（与正文图同级）
INCR_VARS        = ['Surgery_Duration', 'PFCL', 'Phacovitrectomy']   # 三个术中变量
INCR_N_BOOT      = 1000                 # 配对 bootstrap 次数（Δ 指标 95% CI）
INCR_CV_FOLDS    = 5                    # 开发队列重采样：折数（与主流程一致）
INCR_CV_REPEATS  = 20                   # 开发队列重采样：重复次数（与主流程一致）
INCR_SEED        = 42
INCR_ROPE_AUROC  = 0.02                 # ΔAUROC 的「临床可忽略区间」(ROPE)，仅用于图上淡色带
INCR_DCA_XMAX    = SUPP_DCA_XMAX        # 决策曲线横轴上限（与正文 F 一致）
INCR_N_BOOT_CAL  = SUPP_N_BOOT_CAL      # 校准曲线 bootstrap 带
INCR_N_BOOT_DCA  = SUPP_N_BOOT_DCA      # 决策曲线 bootstrap 带

# ---------------- 结果收集器（最后统一导出 CSV）----------------
_INCR_ROWS_SPEC     = []   # 分析设计说明（逐条对应审稿问题）
_INCR_ROWS_METRICS  = []   # 各模型绝对指标 + 95% CI
_INCR_ROWS_DELTAS   = []   # Δ 指标 + 95% CI + p（审稿问题 5 的核心表）
_INCR_ROWS_COEF     = []   # 术中变量的调整后 OR + LRT
_INCR_ROWS_RECLASS  = []   # cfNRI / 分类 NRI / IDI
_INCR_ROWS_DCA      = []   # 全阈值净获益（决策曲线的数字底稿）

# ---------------- 模型展示名（并入 MODEL_LABEL_MAP，供 model_label 复用）----------------
MODEL_LABEL_MAP.update({
    'Preop':            'Preoperative model',
    'Preop_Intraop':    'Pre + intraoperative model',
    'Base_LP':          'Preoperative model (anchored)',
    'Add_Surgery_Duration': 'Preoperative + operative duration',
    'Add_PFCL':             'Preoperative + PFCL',
    'Add_Phacovitrectomy':  'Preoperative + phacovitrectomy',
    'Add_All_Intraop':      'Preoperative + all 3 intraoperative',
})

# 术中变量的短标签（森林图 y 轴用，避免过长）
INCR_SHORT_LABEL = {
    'Base_LP':               'Preoperative model (reference)',
    'Add_Surgery_Duration':  '+ Operative duration',
    'Add_PFCL':              '+ PFCL',
    'Add_Phacovitrectomy':   '+ Phacovitrectomy',
    'Add_All_Intraop':       '+ All three (joint)',
    'Preop_Intraop':         '+ All three (full re-training)',
}

# 队列配色沿用正文语义：蓝 = 内部/开发，红 = 外部
INCR_COHORT_COLOR = {'Development': C_INT, 'External': C_EXT}
INCR_COHORT_MARK  = {'Development': 'o',   'External': 's'}

# 模型对比配色（同一队列内两条曲线）：深蓝 vs 青绿，与 S1–S4 口径一致
INCR_MODEL_COLOR = {'base': COLORS_LIST[3], 'new': COLORS_LIST[2]}


# ==========================================================================================
# 10.1 统计内核 —— IRLS logistic / 校准量 / DeLong / 配对 bootstrap / NRI-IDI
# ==========================================================================================
def _incr_logit(p, eps=1e-6):
    """安全 logit（本脚本无截距校准，这里只是把概率搬到线性尺度作为锚定协变量）"""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _incr_irls(X, y, max_iter=100, tol=1e-9, ridge=1e-8):
    """
    无惩罚 logistic 回归的 IRLS（Newton–Raphson）求解器。

    之所以不直接用 sklearn：本模块要在 bootstrap / 交叉验证里反复拟合上万次
    只有 2–5 列的小设计矩阵，sklearn 的对象开销会占据绝大部分时间；
    IRLS 在数学上与 LogisticRegression(penalty=None) 的 MLE 完全等价
    （ridge=1e-8 仅用于避免完全分离时 Hessian 奇异，对估计值影响可忽略）。

    返回 (beta, cov, loglik, converged)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.zeros(k)
    converged = False

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35, 35)
        mu = expit(eta)
        w = np.clip(mu * (1 - mu), 1e-10, None)
        XtW = X.T * w
        H = XtW @ X + ridge * np.eye(k)
        g = X.T @ (y - mu) - ridge * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    eta = np.clip(X @ beta, -35, 35)
    mu = expit(eta)
    loglik = float(np.sum(y * np.log(np.clip(mu, 1e-12, 1)) +
                          (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1))))
    w = np.clip(mu * (1 - mu), 1e-10, None)
    H = (X.T * w) @ X + ridge * np.eye(k)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    return beta, cov, loglik, converged


def _incr_cal_stats(y, p):
    """
    校准斜率与截距（Cox 校准）：logit(观察) ~ a·logit(预测) + b。
    与正文 get_cali_stats() 数学等价，这里改用 IRLS 以便在 bootstrap 中高速调用。
    """
    lp = _incr_logit(p)
    X = np.column_stack([np.ones(len(lp)), lp])
    beta, _, _, _ = _incr_irls(X, y)
    return float(beta[1]), float(beta[0])          # slope, intercept


def _incr_midrank(x):
    """DeLong 所需的 mid-rank（并列取平均秩）"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _incr_fast_delong(pred_sorted, n_pos):
    """Sun & Xu (2014) 快速 DeLong：返回 (aucs, 协方差矩阵)"""
    m, n = n_pos, pred_sorted.shape[1] - n_pos
    pos = pred_sorted[:, :m]
    neg = pred_sorted[:, m:]
    k = pred_sorted.shape[0]

    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _incr_midrank(pos[r, :])
        ty[r, :] = _incr_midrank(neg[r, :])
        tz[r, :] = _incr_midrank(pred_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:                                   # np.cov 对单行返回标量
        sx = np.atleast_2d(sx); sy = np.atleast_2d(sy)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_base, prob_new):
    """
    两条相关 ROC 曲线的 DeLong 检验（同一批患者、两套预测 → 必须用相关样本方法）。

    返回 dict：AUROC_base / AUROC_new（各带 95% CI）、Delta、SE、z、p、95% CI。
    """
    y = np.asarray(y_true, dtype=int)
    p1 = np.asarray(prob_base, dtype=float)
    p2 = np.asarray(prob_new, dtype=float)

    order = (-y).argsort(kind='mergesort')       # 阳性在前
    n_pos = int(y.sum())
    if n_pos < 2 or (len(y) - n_pos) < 2:
        return {k: np.nan for k in
                ['AUROC_base', 'AUROC_base_lo', 'AUROC_base_hi',
                 'AUROC_new', 'AUROC_new_lo', 'AUROC_new_hi',
                 'Delta_AUROC', 'SE', 'z', 'p_value', 'CI_lo', 'CI_hi']}

    pred_sorted = np.vstack((p1, p2))[:, order]
    try:
        aucs, cov = _incr_fast_delong(pred_sorted, n_pos)
    except Exception:
        return {k: np.nan for k in
                ['AUROC_base', 'AUROC_base_lo', 'AUROC_base_hi',
                 'AUROC_new', 'AUROC_new_lo', 'AUROC_new_hi',
                 'Delta_AUROC', 'SE', 'z', 'p_value', 'CI_lo', 'CI_hi']}

    delta = float(aucs[1] - aucs[0])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    z = delta / se if se > 0 else 0.0
    p = float(2 * (1 - _norm.cdf(abs(z))))

    se0, se1 = float(np.sqrt(max(cov[0, 0], 0.0))), float(np.sqrt(max(cov[1, 1], 0.0)))
    return {
        'AUROC_base': float(aucs[0]),
        'AUROC_base_lo': float(aucs[0] - 1.96 * se0),
        'AUROC_base_hi': float(aucs[0] + 1.96 * se0),
        'AUROC_new': float(aucs[1]),
        'AUROC_new_lo': float(aucs[1] - 1.96 * se1),
        'AUROC_new_hi': float(aucs[1] + 1.96 * se1),
        'Delta_AUROC': delta, 'SE': se, 'z': z, 'p_value': p,
        'CI_lo': delta - 1.96 * se, 'CI_hi': delta + 1.96 * se,
    }


# ---- 参与配对 bootstrap 的指标清单（顺序即 CSV / 森林图的行序）----
_INCR_BOOT_METRICS = ['AUROC', 'AP', 'Brier', 'Scaled_Brier',
                      'Cal_slope', 'Cal_intercept', 'OE', 'NB_at_threshold']

_INCR_METRIC_LABEL = {
    'AUROC':           'ΔAUROC',
    'AP':              'ΔAP',
    'Brier':           'ΔBrier',
    'Scaled_Brier':    'ΔScaled Brier (IPA)',
    'Cal_slope':       'ΔCalibration slope',
    'Cal_intercept':   'ΔCalibration intercept',
    'OE':              'ΔO:E ratio',
    'NB_at_threshold': 'ΔNet benefit @ threshold',
}

# 「越大越好」的指标（决定图上箭头注释与结论判断方向）
_INCR_HIGHER_BETTER = {'AUROC': True, 'AP': True, 'Brier': False, 'Scaled_Brier': True,
                       'Cal_slope': None, 'Cal_intercept': None, 'OE': None,
                       'NB_at_threshold': True}


def _incr_metric_battery(y, p, threshold, full=False):
    """一次性算全套指标；full=True 时追加 H-L 与阈值处的分类指标（仅点估计用）"""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out = {'N': len(y), 'Events': int(y.sum()), 'Event_rate': float(y.mean())}

    out['AUROC'] = float(roc_auc_score(y, p))
    out['AP'] = float(average_precision_score(y, p))
    out['Brier'] = float(brier_score_loss(y, p))
    pbar = float(y.mean())
    out['Scaled_Brier'] = (1 - out['Brier'] / (pbar * (1 - pbar))) if 0 < pbar < 1 else np.nan

    slope, intercept = _incr_cal_stats(y, p)
    out['Cal_slope'], out['Cal_intercept'] = slope, intercept
    out['OE'] = float(y.sum() / p.sum()) if p.sum() > 0 else np.nan
    out['NB_at_threshold'] = float(calculate_net_benefit(y, p, threshold))

    if full:
        try:
            hl, hlp = hosmer_lemeshow_test(y, p, n_groups=10)
        except Exception:
            hl, hlp = np.nan, np.nan
        out['HL_chi2'], out['HL_p'] = hl, hlp

        yhat = (p >= threshold).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
        out['TP'], out['FP'], out['TN'], out['FN'] = tp, fp, tn, fn
        out['Sensitivity'] = tp / (tp + fn) if (tp + fn) else np.nan
        out['Specificity'] = tn / (tn + fp) if (tn + fp) else np.nan
        out['PPV'] = tp / (tp + fp) if (tp + fp) else np.nan
        out['NPV'] = tn / (tn + fn) if (tn + fn) else np.nan
    return out


def _incr_boot_indices(y, n_boot, seed):
    """
    分层 bootstrap 的下标序列（事件组与非事件组各自有放回抽样）。

    分层而非简单重采样，是因为外部队列事件数偏少时，简单重采样有一定概率
    抽出 0 个事件，导致 AUROC / AP / 校准斜率无定义；分层可保证每次重采样的
    事件数恒定，Δ 指标的 bootstrap 分布也更稳定。
    """
    y = np.asarray(y, dtype=int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])


def _incr_pct_ci(samples, lo=2.5, hi=97.5):
    s = np.asarray([v for v in samples if np.isfinite(v)], dtype=float)
    if s.size < 20:
        return np.nan, np.nan
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


def _incr_boot_p(samples):
    """双侧 bootstrap p 值：2 × min(P(Δ≤0), P(Δ≥0))，下限截断到 1/n_boot"""
    s = np.asarray([v for v in samples if np.isfinite(v)], dtype=float)
    if s.size < 20:
        return np.nan
    p = 2 * min(float(np.mean(s <= 0)), float(np.mean(s >= 0)))
    return float(min(1.0, max(p, 1.0 / s.size)))


def _incr_paired_bootstrap(y, p_base, p_new, threshold,
                           n_boot=None, seed=INCR_SEED):
    """
    【配对】分层 bootstrap：同一次重采样同时计算 base 与 new 的全套指标，
    再取差值 → 这是 Δ 指标 95% CI 的唯一正确算法。

    返回 (abs_summary, delta_summary, delta_samples)
      abs_summary   : {model: {metric: (点估计, lo, hi)}}
      delta_summary : {metric: {'delta','lo','hi','p'}}
      delta_samples : {metric: ndarray}  ← 供 S5 的 bootstrap 分布图使用
    """
    n_boot = n_boot or INCR_N_BOOT
    y = np.asarray(y, dtype=int)
    p_base = np.asarray(p_base, dtype=float)
    p_new = np.asarray(p_new, dtype=float)

    point_base = _incr_metric_battery(y, p_base, threshold, full=True)
    point_new = _incr_metric_battery(y, p_new, threshold, full=True)

    b_s = {m: [] for m in _INCR_BOOT_METRICS}
    n_s = {m: [] for m in _INCR_BOOT_METRICS}
    d_s = {m: [] for m in _INCR_BOOT_METRICS}

    for idx in _incr_boot_indices(y, n_boot, seed):
        yb = y[idx]
        if yb.sum() < 2 or (len(yb) - yb.sum()) < 2:
            continue
        try:
            mb = _incr_metric_battery(yb, p_base[idx], threshold)
            mn = _incr_metric_battery(yb, p_new[idx], threshold)
        except Exception:
            continue
        for m in _INCR_BOOT_METRICS:
            b_s[m].append(mb[m]); n_s[m].append(mn[m]); d_s[m].append(mn[m] - mb[m])

    abs_summary = {'base': {}, 'new': {}}
    for m in _INCR_BOOT_METRICS:
        lo_b, hi_b = _incr_pct_ci(b_s[m])
        lo_n, hi_n = _incr_pct_ci(n_s[m])
        abs_summary['base'][m] = (point_base[m], lo_b, hi_b)
        abs_summary['new'][m] = (point_new[m], lo_n, hi_n)

    delta_summary = {}
    delta_samples = {}
    for m in _INCR_BOOT_METRICS:
        lo, hi = _incr_pct_ci(d_s[m])
        delta_summary[m] = {
            'delta': point_new[m] - point_base[m],
            'lo': lo, 'hi': hi, 'p': _incr_boot_p(d_s[m]),
            'n_boot_used': len(d_s[m]),
        }
        delta_samples[m] = np.asarray(d_s[m], dtype=float)

    return abs_summary, delta_summary, delta_samples, point_base, point_new


def _incr_nri_idi(y, p_old, p_new, threshold, n_boot=None, seed=INCR_SEED):
    """
    重分类指标：连续 NRI（cfNRI）、锁定阈值下的分类 NRI、IDI。
    点估计公式与「术中特征敏感性分析.py」一致；CI 改用与 Δ 指标同源的
    配对分层 bootstrap（比解析 SE 更稳健，且与本表其它 CI 口径统一）。
    """
    n_boot = n_boot or INCR_N_BOOT
    y = np.asarray(y, dtype=int)
    po = np.asarray(p_old, dtype=float)
    pn = np.asarray(p_new, dtype=float)

    def _core(yy, a, b):
        ev, ne = yy == 1, yy == 0
        n_ev, n_ne = ev.sum(), ne.sum()
        if n_ev == 0 or n_ne == 0:
            return {k: np.nan for k in
                    ['cfNRI', 'cfNRI_ev', 'cfNRI_ne', 'catNRI', 'catNRI_ev',
                     'catNRI_ne', 'IDI', 'IDI_ev', 'IDI_ne']}
        # 连续 NRI
        cf_ev = (np.sum(b[ev] > a[ev]) - np.sum(b[ev] < a[ev])) / n_ev
        cf_ne = (np.sum(b[ne] < a[ne]) - np.sum(b[ne] > a[ne])) / n_ne
        # 分类 NRI（单一锁定切点）
        ca, cb = (a >= threshold).astype(int), (b >= threshold).astype(int)
        ct_ev = (np.sum(cb[ev] > ca[ev]) - np.sum(cb[ev] < ca[ev])) / n_ev
        ct_ne = (np.sum(cb[ne] < ca[ne]) - np.sum(cb[ne] > ca[ne])) / n_ne
        # IDI
        idi_ev = float(np.mean(b[ev]) - np.mean(a[ev]))
        idi_ne = float(np.mean(b[ne]) - np.mean(a[ne]))
        return {'cfNRI': cf_ev + cf_ne, 'cfNRI_ev': cf_ev, 'cfNRI_ne': cf_ne,
                'catNRI': ct_ev + ct_ne, 'catNRI_ev': ct_ev, 'catNRI_ne': ct_ne,
                'IDI': idi_ev - idi_ne, 'IDI_ev': idi_ev, 'IDI_ne': idi_ne}

    point = _core(y, po, pn)
    keys = list(point.keys())
    boots = {k: [] for k in keys}
    for idx in _incr_boot_indices(y, n_boot, seed):
        r = _core(y[idx], po[idx], pn[idx])
        for k in keys:
            boots[k].append(r[k])

    out = {}
    for k in keys:
        lo, hi = _incr_pct_ci(boots[k])
        out[k] = {'value': point[k], 'lo': lo, 'hi': hi, 'p': _incr_boot_p(boots[k])}

    # 重分类计数表（锁定阈值，供正文/附表引用）
    ev, ne = y == 1, y == 0
    ca, cb = (po >= threshold).astype(int), (pn >= threshold).astype(int)
    out['_counts'] = {
        'events_up': int(np.sum(cb[ev] > ca[ev])), 'events_down': int(np.sum(cb[ev] < ca[ev])),
        'nonevents_up': int(np.sum(cb[ne] > ca[ne])), 'nonevents_down': int(np.sum(cb[ne] < ca[ne])),
        'n_events': int(ev.sum()), 'n_nonevents': int(ne.sum()),
    }
    return out


# ==========================================================================================
# 10.2 数据装载 —— 逐样本概率与术中变量原始值（只读盘，不重训主流水线）
# ==========================================================================================
def _incr_load_framework_A():
    """
    Framework A（联合，主分析）：术前模型 vs 术前+术中模型，
    两者均由「术中特征敏感性分析.py」用【与主模型完全相同】的流水线训练得到，
    逐样本概率直接读 NRI_Comparison_Data_*.pkl。

    返回 dict 或 None。
      train 概率优先为 OOF（包内 metadata.train_data_source 会标明）。
    """
    path = (f"{DATA_PATH}/NRI_Comparison_Data_{TIMESTAMP_INTRA}.pkl"
            if os.path.exists(f"{DATA_PATH}/NRI_Comparison_Data_{TIMESTAMP_INTRA}.pkl")
            else _find_latest(f"{DATA_PATH}/NRI_Comparison_Data_*.pkl"))
    if not path:
        print("  ℹ️ 未找到 NRI_Comparison_Data_*.pkl → 跳过 Framework A（全流水线重训的联合模型）")
        return None
    try:
        pkg = joblib.load(path)
    except Exception as e:
        print(f"  ⚠️ 读取 {os.path.basename(path)} 失败: {e}")
        return None

    meta = pkg.get('metadata', {}) or {}
    src = meta.get('train_data_source', 'unknown')
    if src != 'OOF':
        print(f"  ⚠️ 该包的开发队列概率来源为 '{src}' 而非 OOF；"
              f"Framework A 的开发队列 Δ 会带乐观偏倚，已在 CSV 中标注")

    out = {
        'y_dev': np.asarray(pkg['y_train'], dtype=int),
        'y_ext': np.asarray(pkg['y_external'], dtype=int),
        'p_dev_base': np.asarray(pkg['preop_model']['prob_train'], dtype=float),
        'p_dev_new': np.asarray(pkg['intraop_model']['prob_train'], dtype=float),
        'p_ext_base': np.asarray(pkg['preop_model']['prob_external'], dtype=float),
        'p_ext_new': np.asarray(pkg['intraop_model']['prob_external'], dtype=float),
        'thr_base': float(pkg['preop_model'].get('threshold', clinical_threshold)),
        'thr_new': float(pkg['intraop_model'].get('threshold', clinical_threshold)),
        'dev_source': src,
        'preop_model_name': meta.get('preop_model_name', best_model_name),
        'intraop_model_name': meta.get('intraop_model_name', 'N/A'),
        'source_file': os.path.basename(path),
    }
    print(f"  ✓ Framework A 概率来源: {out['source_file']}"
          f"（开发队列={src}；术前={out['preop_model_name']}，"
          f"术前+术中={out['intraop_model_name']}）")
    return out


def _incr_load_intraop_frames():
    """
    Framework B 需要术中变量的【原始值】。按优先级检索：
      1) Sensitivity_Model_Package_*.pkl → datasets{X_train, X_ext}（含术前+术中全部列）
      2) 建模用的原始 CSV（当前目录或 DATA_PATH 下）
    返回 (df_dev, df_ext, source_str) 或 (None, None, None)
    """
    # ---- 路径 1：敏感性分析数据包 ----
    for cand in [f"{DATA_PATH}/Sensitivity_Model_Package_{TIMESTAMP_INTRA}.pkl",
                 f"{DATA_PATH}/Sensitivity_Model_Package_latest.pkl",
                 _find_latest(f"{DATA_PATH}/Sensitivity_Model_Package_*.pkl")]:
        if not cand or not os.path.exists(cand):
            continue
        try:
            pkg = joblib.load(cand)
            ds = pkg.get('datasets', {}) or {}
            d_dev, d_ext = ds.get('X_train', None), ds.get('X_ext', None)
            if d_dev is not None and d_ext is not None and \
               all(v in d_dev.columns for v in INCR_VARS):
                print(f"  ✓ 术中变量原始值来源: {os.path.basename(cand)}")
                return d_dev.reset_index(drop=True), d_ext.reset_index(drop=True), \
                    os.path.basename(cand)
        except Exception as e:
            print(f"  ⚠️ 读取 {os.path.basename(cand)} 失败: {e}")

    # ---- 路径 2：原始 CSV ----
    for dev_name, ext_name in [('Internal_setA_NaN2.csv', 'External_setA_NaN3.csv')]:
        for base in ['.', DATA_PATH]:
            pd_dev, pd_ext = os.path.join(base, dev_name), os.path.join(base, ext_name)
            if os.path.exists(pd_dev) and os.path.exists(pd_ext):
                try:
                    d_dev = pd.read_csv(pd_dev, index_col=0)
                    d_ext = pd.read_csv(pd_ext, index_col=0)
                    d_dev = d_dev.dropna(subset=['Recurrence']).reset_index(drop=True)
                    d_ext = d_ext.dropna(subset=['Recurrence']).reset_index(drop=True)
                    if all(v in d_dev.columns for v in INCR_VARS):
                        print(f"  ✓ 术中变量原始值来源: {pd_dev} / {pd_ext}")
                        return d_dev, d_ext, f'{dev_name} + {ext_name}'
                except Exception as e:
                    print(f"  ⚠️ 读取原始 CSV 失败: {e}")

    print("  ℹ️ 未找到含术中变量的原始数据 → 跳过 Framework B（逐项加入分析）")
    return None, None, None


def _incr_check_alignment(y_ref, y_cand, tag):
    """行序/样本对齐校验：长度一致且标签逐例相同才认为可以放心比较"""
    y_ref = np.asarray(y_ref, dtype=int)
    y_cand = np.asarray(y_cand, dtype=int)
    if len(y_ref) != len(y_cand):
        print(f"  ⚠️ {tag}: 样本量不一致（{len(y_ref)} vs {len(y_cand)}），已跳过该比较")
        return False
    if not np.array_equal(y_ref, y_cand):
        n_diff = int((y_ref != y_cand).sum())
        print(f"  ⚠️ {tag}: 结局标签有 {n_diff} 例不一致（行序可能错位），已跳过该比较")
        return False
    return True


# ==========================================================================================
# 10.3 Framework B —— 以术前模型线性预测值为锚的增量 logistic 回归
#      baseline : logit(p_preop)
#      extended : logit(p_preop) + 候选术中变量
#   · 开发队列：重复分层 5 折 CV（×20）的 out-of-fold 预测（缺失填补与标准化【折内】完成）
#   · 外部队列：系数在整个开发队列锁定后直接应用，无任何外部信息回流
# ==========================================================================================
def _incr_build_Z(df_dev, df_ext, varlist):
    """
    生成候选变量设计矩阵（仅做必要的编码，缺失/标准化留到折内）。
      · 数值型（唯一值 > 2）→ 保留原列，后续按折内均值/标准差标准化（OR 解释为 per 1 SD）
      · 二分类（唯一值 ≤ 2）→ 保留 0/1，不标准化（OR 解释为 1 vs 0）
      · 字符型/多分类       → 按开发集水平做哑变量（drop-first）
    返回 (Z_dev, Z_ext, colnames, is_binary, unit_note)
    """
    Zd, Ze, names, is_bin, units = [], [], [], [], []
    for v in varlist:
        s_d, s_e = df_dev[v], df_ext[v]
        if pd.api.types.is_numeric_dtype(s_d):
            x_d = pd.to_numeric(s_d, errors='coerce').astype(float).values
            x_e = pd.to_numeric(s_e, errors='coerce').astype(float).values
            uniq = np.unique(x_d[~np.isnan(x_d)])
            binary = uniq.size <= 2
            Zd.append(x_d); Ze.append(x_e); names.append(v)
            is_bin.append(binary)
            if binary:
                units.append('per 1 unit (yes vs no)')
            else:
                sd = float(np.nanstd(x_d))
                units.append(f'per 1 SD ({sd:.3g} {("min" if "Duration" in v else "unit")})')
        else:
            levels = sorted(pd.Series(s_d.dropna().astype(str)).unique())[1:]
            for lv in levels:
                Zd.append((s_d.astype(str) == lv).astype(float).values)
                Ze.append((s_e.astype(str) == lv).astype(float).values)
                names.append(f'{v}={lv}')
                is_bin.append(True)
                units.append(f'level "{lv}" vs reference')
    if not names:
        return None, None, [], None, []
    return (np.column_stack(Zd), np.column_stack(Ze), names,
            np.asarray(is_bin, dtype=bool), units)


def _incr_transform(Z_fit, Z_apply_list, is_bin):
    """
    折内预处理：中位数填补 + 连续变量标准化（统计量只用 Z_fit 估计，杜绝泄露）。
    Z_apply_list 为需要用同一组统计量变换的矩阵列表。
    """
    med = np.nanmedian(Z_fit, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Zf = np.where(np.isnan(Z_fit), med, Z_fit)

    mu = np.where(is_bin, 0.0, Zf.mean(axis=0))
    sd = np.where(is_bin, 1.0, Zf.std(axis=0))
    sd = np.where(sd < 1e-12, 1.0, sd)

    outs = []
    for Z in Z_apply_list:
        Za = np.where(np.isnan(Z), med, Z)
        outs.append((Za - mu) / sd)
    return (Zf - mu) / sd, outs


def _incr_fit_framework_B(lp_dev, Z_dev, y_dev, lp_ext, Z_ext, is_bin, colnames,
                          cv_folds=INCR_CV_FOLDS, cv_repeats=INCR_CV_REPEATS,
                          seed=INCR_SEED):
    """
    返回 dict:
      oof_base / oof_new : 开发队列 out-of-fold 预测概率（重复 CV 取均值）
      ext_base / ext_new : 外部队列预测概率（开发队列锁定系数）
      coef_rows          : 候选变量的调整后 OR (95% CI) 与 Wald p
      lrt                : 似然比检验 {chi2, df, p}
    """
    y_dev = np.asarray(y_dev, dtype=int)
    n = len(y_dev)
    oof_base = np.zeros(n); oof_new = np.zeros(n); cnt = np.zeros(n)

    rskf = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats,
                                   random_state=seed)
    for tr, te in rskf.split(np.zeros(n), y_dev):
        # --- baseline：只含锚定的术前线性预测值 ---
        Xtr_b = np.column_stack([np.ones(len(tr)), lp_dev[tr]])
        Xte_b = np.column_stack([np.ones(len(te)), lp_dev[te]])
        beta_b, _, _, _ = _incr_irls(Xtr_b, y_dev[tr])
        oof_base[te] += expit(np.clip(Xte_b @ beta_b, -35, 35))

        # --- extended：锚 + 候选术中变量（折内填补/标准化）---
        Ztr, (Zte,) = _incr_transform(Z_dev[tr], [Z_dev[te]], is_bin)
        Xtr_e = np.column_stack([np.ones(len(tr)), lp_dev[tr], Ztr])
        Xte_e = np.column_stack([np.ones(len(te)), lp_dev[te], Zte])
        beta_e, _, _, _ = _incr_irls(Xtr_e, y_dev[tr])
        oof_new[te] += expit(np.clip(Xte_e @ beta_e, -35, 35))

        cnt[te] += 1

    cnt = np.where(cnt == 0, 1, cnt)
    oof_base /= cnt
    oof_new /= cnt

    # ---- 全开发队列锁定系数 → 外部队列 ----
    Z_dev_t, (Z_ext_t,) = _incr_transform(Z_dev, [Z_ext], is_bin)

    Xd_b = np.column_stack([np.ones(n), lp_dev])
    Xd_e = np.column_stack([np.ones(n), lp_dev, Z_dev_t])
    beta_b, cov_b, ll_b, _ = _incr_irls(Xd_b, y_dev)
    beta_e, cov_e, ll_e, _ = _incr_irls(Xd_e, y_dev)

    ext_base = ext_new = None
    if lp_ext is not None and Z_ext_t is not None and len(lp_ext) > 0:
        Xe_b = np.column_stack([np.ones(len(lp_ext)), lp_ext])
        Xe_e = np.column_stack([np.ones(len(lp_ext)), lp_ext, Z_ext_t])
        ext_base = expit(np.clip(Xe_b @ beta_b, -35, 35))
        ext_new = expit(np.clip(Xe_e @ beta_e, -35, 35))

    # ---- 候选变量的调整后 OR 与 Wald 检验 ----
    coef_rows = []
    se_e = np.sqrt(np.clip(np.diag(cov_e), 0, None))
    for j, nm in enumerate(colnames):
        k = 2 + j                                  # 0=截距, 1=锚定 LP
        b, s = float(beta_e[k]), float(se_e[k])
        z = b / s if s > 0 else np.nan
        coef_rows.append({
            'Variable_code': nm,
            'Variable': rename_feature(nm),
            'Beta': b, 'SE': s,
            'OR': float(np.exp(b)),
            'OR_CI_lo': float(np.exp(b - 1.96 * s)),
            'OR_CI_hi': float(np.exp(b + 1.96 * s)),
            'z': z,
            'Wald_p': float(2 * (1 - _norm.cdf(abs(z)))) if np.isfinite(z) else np.nan,
        })

    # ---- 似然比检验：extended vs baseline（开发队列，自由度=新增变量数）----
    lrt_chi2 = float(2 * (ll_e - ll_b))
    lrt_df = int(len(colnames))
    lrt_p = float(1 - _chi2.cdf(max(lrt_chi2, 0.0), lrt_df)) if lrt_df > 0 else np.nan

    return {
        'oof_base': oof_base, 'oof_new': oof_new,
        'ext_base': ext_base, 'ext_new': ext_new,
        'coef_rows': coef_rows,
        'lrt': {'chi2': lrt_chi2, 'df': lrt_df, 'p': lrt_p},
        'anchor_beta': float(beta_e[1]),
        'n_dev': n,
    }


# ==========================================================================================
# 10.4 S5 专用绘图函数（沿用本脚本的单面板 SCI 风格）
# ==========================================================================================
def _incr_nice_ticks(lo, hi, n=5):
    """在 [lo, hi] 内取整齐刻度（避免 matplotlib 自动刻度侵入右侧注释列）"""
    from matplotlib.ticker import MaxNLocator
    t = MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return [v for v in t if lo - 1e-12 <= v <= hi + 1e-12]


def plot_delta_forest(rows, save_path, xlabel='Δ (with intraoperative − without)',
                      figsize=None, rope=None, rope_panels=None, title_note=None,
                      value_fmt='{:+.4f}', annot_header='Δ (95% CI), p',
                      record_tag=None):
    """
    Δ 指标森林图（本脚本风格）。

      · 竖直零线 = 无增量；点 + 95% CI；CI 跨过零线 → 空心点，不跨 → 实心点，
        一眼分辨「确定的差异」与「与 0 无法区分」；
      · 右侧独立注释列写出 Δ (95% CI) 与 p，点再密也不会与数字重叠；
      · 可选 ROPE 淡色带（临床可忽略区间），支撑「未显示明确增量价值」的表述；
      · rows 可带 'panel' 键：不同 panel 各自独立 x 轴 —— 这一点很关键，
        ΔAUROC（≈0.001 量级）与 Δ校准斜率（≈0.1 量级）若共用一根横轴，
        前者会被压成一个点，图就没有信息量了。

    rows : list of dict(label=…, cohort=…, delta=…, lo=…, hi=…, p=…, panel=…)
    """
    rows = [r for r in rows if np.isfinite(r.get('delta', np.nan))]
    if not rows:
        print("  ⚠️ 无可绘制的 Δ 数据，跳过")
        return None

    # ---- 分组：panel（独立 x 轴） / label（y 行） / cohort（同行 dodge）----
    panels = []
    for r in rows:
        r.setdefault('panel', 'main')
        if r['panel'] not in panels:
            panels.append(r['panel'])
    cohorts = []
    for r in rows:
        if r['cohort'] not in cohorts:
            cohorts.append(r['cohort'])

    panel_rows = {pn: [r for r in rows if r['panel'] == pn] for pn in panels}
    panel_labels = {}
    for pn in panels:
        labs = []
        for r in panel_rows[pn]:
            if r['label'] not in labs:
                labs.append(r['label'])
        panel_labels[pn] = labs

    n_all = sum(len(panel_labels[pn]) for pn in panels)
    if figsize is None:
        figsize = (5.6, max(2.6, 0.46 * n_all + 0.55 * len(panels) + 1.1))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(panels), 1,
                           height_ratios=[len(panel_labels[pn]) + 0.35 for pn in panels],
                           hspace=0.42)
    offs_all = np.linspace(0.19, -0.19, len(cohorts)) if len(cohorts) > 1 else np.array([0.0])

    axes = []
    for pi, pn in enumerate(panels):
        ax = fig.add_subplot(gs[pi])
        axes.append(ax)
        labels = panel_labels[pn]
        n_row = len(labels)
        prs = panel_rows[pn]

        # ---- x 轴范围：数据区 + 右侧注释列 ----
        vals = [0.0]
        for r in prs:
            vals += [r['delta'],
                     r['lo'] if np.isfinite(r.get('lo', np.nan)) else r['delta'],
                     r['hi'] if np.isfinite(r.get('hi', np.nan)) else r['delta']]
        show_rope = (rope is not None and (rope_panels is None or pn in rope_panels))
        if show_rope:
            vals += [rope[0] * 1.18, rope[1] * 1.18]
        d_lo, d_hi = float(np.min(vals)), float(np.max(vals))
        span = max(d_hi - d_lo, 1e-9)
        d_lo -= span * 0.08
        d_hi += span * 0.08
        data_span = d_hi - d_lo
        ax.set_xlim(d_lo, d_hi + data_span * 0.68)          # 右 68% 留给注释列

        # ---- ROPE 带 ----
        if show_rope:
            ax.axvspan(rope[0], rope[1], color='#BBBBBB', alpha=0.13, zorder=0, lw=0)
            for xv in rope:
                ax.axvline(xv, color='#AAAAAA', ls=':', lw=0.7, alpha=0.9, zorder=1)

        ax.axvline(0.0, color='#333333', ls='--', lw=0.9, alpha=0.85, zorder=1)
        ax.axvline(d_hi, color='#DDDDDD', lw=0.7, alpha=0.9, zorder=1)   # 数据区/注释列分隔

        for ci, co in enumerate(cohorts):
            col = INCR_COHORT_COLOR.get(co, COLORS_LIST[ci % len(COLORS_LIST)])
            mk = INCR_COHORT_MARK.get(co, 'o')
            for r in [x for x in prs if x['cohort'] == co]:
                yi = n_row - 1 - labels.index(r['label']) + offs_all[ci]
                lo = r['lo'] if np.isfinite(r.get('lo', np.nan)) else r['delta']
                hi = r['hi'] if np.isfinite(r.get('hi', np.nan)) else r['delta']
                lo_c, hi_c = max(lo, d_lo), min(hi, d_hi)
                crosses_zero = (lo <= 0 <= hi)

                ax.plot([lo_c, hi_c], [yi, yi], color=col, lw=1.0, alpha=0.9,
                        solid_capstyle='butt', zorder=3)
                for xcap, inside in ((lo_c, lo >= d_lo), (hi_c, hi <= d_hi)):
                    if inside:
                        ax.plot([xcap, xcap], [yi - 0.055, yi + 0.055], color=col,
                                lw=0.9, alpha=0.9, zorder=3)
                ax.plot([r['delta']], [yi], marker=mk, ms=4.2,
                        markerfacecolor=('white' if crosses_zero else col),
                        markeredgecolor=col, markeredgewidth=1.0, ls='none', zorder=4)

                # ---- 右侧注释列：Δ (95% CI), p ----
                txt = value_fmt.format(r['delta'])
                if np.isfinite(r.get('lo', np.nan)) and np.isfinite(r.get('hi', np.nan)):
                    txt += (' (' + value_fmt.format(r['lo']).lstrip('+') + ' to '
                            + value_fmt.format(r['hi']).lstrip('+') + ')')
                if np.isfinite(r.get('p', np.nan)):
                    txt += f', p={_fmt_p(r["p"])}'
                ax.text(d_hi + data_span * 0.03, yi, txt, ha='left', va='center',
                        fontsize=5.6, color=col, zorder=5)

        ax.set_yticks(np.arange(n_row)[::-1])
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.62, n_row - 0.34)
        ax.set_xticks(_incr_nice_ticks(d_lo, d_hi))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.13, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)

        head = pn if len(panels) > 1 else (title_note or '')
        if head:
            ax.text(0.0, 1.05, head, transform=ax.transAxes, fontsize=6.8,
                    color='#555555', va='bottom', ha='left', fontstyle='italic')
        if annot_header:
            ax.text(d_hi + data_span * 0.03, n_row - 0.30, annot_header,
                    ha='left', va='center', fontsize=5.8, color='#888888')
        if pi == len(panels) - 1:
            ax.set_xlabel(xlabel)

    if len(panels) > 1 and title_note:
        axes[0].set_title(title_note, fontsize=7, color='#555555', loc='left', pad=16)

    # ---- 统一图例（置于最后一个 panel 下方，绝不遮挡数据）----
    handles = [Line2D([0], [0], marker=INCR_COHORT_MARK.get(c, 'o'),
                      color=INCR_COHORT_COLOR.get(c, '#333333'),
                      markerfacecolor=INCR_COHORT_COLOR.get(c, '#333333'),
                      markeredgecolor=INCR_COHORT_COLOR.get(c, '#333333'),
                      ms=4.2, lw=1.0, label=c) for c in cohorts]
    handles.append(Line2D([0], [0], marker='o', ls='none', ms=4.2,
                          markerfacecolor='white', markeredgecolor='#666666',
                          label='95% CI includes 0'))
    if rope is not None:
        handles.append(Rectangle((0, 0), 1, 1, facecolor='#BBBBBB', alpha=0.35,
                                 edgecolor='none',
                                 label=f'Negligible range (±{abs(rope[1]):g})'))
    sci_legend(axes[-1], handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, -0.30), ncol=min(len(handles), 4), fontsize=6.0,
               columnspacing=1.1)

    plt.tight_layout()
    _supp_save(save_path)
    return fig


def plot_incr_roc_compare(y, p_base, p_new, label_base, label_new, save_path,
                          note=None, figsize=(3.6, 3.6), record_tag=None):
    """两模型 ROC 叠加 + DeLong 检验（同一批患者的两套预测 → 必须用相关 ROC 方法）"""
    y = np.asarray(y, dtype=int)
    d = delong_roc_test(y, p_base, p_new)

    fig, ax = plt.subplots(figsize=figsize)
    for p, lab, col, auc, lo, hi in [
            (p_base, label_base, INCR_MODEL_COLOR['base'],
             d['AUROC_base'], d['AUROC_base_lo'], d['AUROC_base_hi']),
            (p_new, label_new, INCR_MODEL_COLOR['new'],
             d['AUROC_new'], d['AUROC_new_lo'], d['AUROC_new_hi'])]:
        fpr, tpr, _ = roc_curve(y, np.asarray(p, dtype=float))
        ax.plot(fpr, tpr, color=col, lw=1.5, alpha=0.9,
                label=f'{lab} (AUROC = {auc:.3f}, 95% CI {lo:.3f}–{hi:.3f})')

    ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--', zorder=1)

    # 队列信息与 Δ 统计合并进同一个方框，避免与曲线重叠
    box = (note.replace('\n', ' · ') + '\n' if note else '')
    box += (f"ΔAUROC = {d['Delta_AUROC']:+.3f}\n"
            f"95% CI {d['CI_lo']:+.3f} to {d['CI_hi']:+.3f}\n"
            f"DeLong p = {_fmt_p(d['p_value'])}")
    ax.text(0.97, 0.05, box, transform=ax.transAxes, fontsize=6.2,
            va='bottom', ha='right', color='#333333', linespacing=1.35,
            bbox=dict(boxstyle='square,pad=0.35', facecolor='white',
                      edgecolor='#999999', linewidth=0.6))

    ax.set_xlabel('1 − Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sci_legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.20), fontsize=6.0)

    _SUPP_STATS_ROWS.append({
        'Figure': record_tag or os.path.basename(save_path),
        'Series': f'{label_new} vs {label_base}', 'N': len(y), 'Events': int(y.sum()),
        'AUROC_base': d['AUROC_base'], 'AUROC_extended': d['AUROC_new'],
        'Delta_AUROC': d['Delta_AUROC'], 'Delta_AUROC_CI_lo': d['CI_lo'],
        'Delta_AUROC_CI_hi': d['CI_hi'], 'DeLong_p': d['p_value'],
    })

    plt.tight_layout()
    _supp_save(save_path)
    return fig, d


def plot_incr_pr_compare(y, p_base, p_new, label_base, label_new, save_path,
                         note=None, figsize=(3.6, 3.6), record_tag=None):
    """两模型 PR 曲线叠加 —— 事件率较低时，AP 对增量比 AUROC 更敏感"""
    y = np.asarray(y, dtype=int)
    fig, ax = plt.subplots(figsize=figsize)
    aps = {}
    for p, lab, col in [(p_base, label_base, INCR_MODEL_COLOR['base']),
                        (p_new, label_new, INCR_MODEL_COLOR['new'])]:
        p = np.asarray(p, dtype=float)
        prec, rec, _ = precision_recall_curve(y, p)
        ap = float(average_precision_score(y, p))
        aps[lab] = ap
        ax.plot(rec, prec, color=col, lw=1.5, alpha=0.9, label=f'{lab} (AP = {ap:.3f})')

    prev = float(y.mean())
    ax.axhline(prev, color='#999999', ls=':', lw=0.8, label=f'Prevalence = {prev:.3f}')

    box = (note.replace('\n', ' · ') + '\n' if note else '')
    box += f'ΔAP = {aps[label_new] - aps[label_base]:+.3f}'
    ax.text(0.03, 0.04, box, transform=ax.transAxes, fontsize=6.2,
            va='bottom', ha='left', color='#333333', linespacing=1.35,
            bbox=dict(boxstyle='square,pad=0.35', facecolor='white',
                      edgecolor='#999999', linewidth=0.6))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sci_legend(ax, loc='upper right', fontsize=6.2)

    plt.tight_layout()
    _supp_save(save_path)
    return fig


def plot_incr_delta_distribution(samples_by_cohort, save_path, metric_label='ΔAUROC',
                                 figsize=(3.8, 3.0), rope=None, record_tag=None):
    """
    配对 bootstrap 的 Δ 分布图：阶梯直方图 + 2.5/97.5 分位竖线 + 零线。
    回答的是「增量是否稳定地大于 0」，而不是只看一个点估计。
    """
    samples_by_cohort = {k: np.asarray(v, dtype=float)
                         for k, v in samples_by_cohort.items()
                         if v is not None and len(np.asarray(v)) > 20}
    if not samples_by_cohort:
        print("  ⚠️ 无 bootstrap 样本，跳过分布图")
        return None

    all_vals = np.concatenate(list(samples_by_cohort.values()))
    lo_x, hi_x = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
    if rope is not None:
        lo_x, hi_x = min(lo_x, rope[0] * 1.12), max(hi_x, rope[1] * 1.12)
    pad = (hi_x - lo_x) * 0.10 + 1e-9
    bins = np.linspace(lo_x - pad, hi_x + pad, 46)

    fig, ax = plt.subplots(figsize=figsize)
    if rope is not None:
        ax.axvspan(rope[0], rope[1], color='#BBBBBB', alpha=0.13, zorder=0, lw=0)
        for xv in rope:
            ax.axvline(xv, color='#AAAAAA', ls=':', lw=0.7, alpha=0.9, zorder=1)

    for co, s in samples_by_cohort.items():
        col = INCR_COHORT_COLOR.get(co, C_INT)
        lo, hi = np.percentile(s, [2.5, 97.5])
        ax.hist(s, bins=bins, histtype='stepfilled', color=col, alpha=0.18, lw=0, zorder=2)
        ax.hist(s, bins=bins, histtype='step', color=col, lw=1.2, zorder=3,
                label=f'{co}: {np.median(s):+.3f} ({lo:+.3f} to {hi:+.3f})')
        for xv in (lo, hi):
            ax.axvline(xv, color=col, ls=':', lw=0.9, alpha=0.85, zorder=3)

    ax.axvline(0.0, color='#333333', ls='--', lw=1.0, alpha=0.9, zorder=4)
    ax.set_xlabel(f'{metric_label} (pre + intraoperative − preoperative)')
    ax.set_ylabel('Bootstrap resamples')
    ax.set_xlim(bins[0], bins[-1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.13, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    sci_legend(ax, loc='upper left', fontsize=5.8, title='Median (95% CI)')

    plt.tight_layout()
    _supp_save(save_path)
    return fig


def plot_incr_reclass_flow(y, p_base, p_new, threshold, save_path,
                           figsize=(4.6, 2.6), note=None, record_tag=None):
    """
    锁定阈值下的重分类流向图：事件组 / 非事件组分别统计
    「上调到高风险」「下调到低风险」「未改变」的比例。
    真实数据中重分类通常很少（色块极窄），因此右侧同时给出计数注释，
    保证数字在任何情况下都可读。
    """
    y = np.asarray(y, dtype=int)
    ca = (np.asarray(p_base, float) >= threshold).astype(int)
    cb = (np.asarray(p_new, float) >= threshold).astype(int)

    groups = [('Events', y == 1, C_EXT), ('Non-events', y == 0, C_INT)]
    fig, ax = plt.subplots(figsize=figsize)
    ypos, bar_h = np.arange(len(groups))[::-1], 0.46

    for gi, (gname, mask, col) in enumerate(groups):
        n = int(mask.sum())
        up = int(np.sum(cb[mask] > ca[mask]))
        dn = int(np.sum(cb[mask] < ca[mask]))
        same = n - up - dn
        good_is_up = (gname == 'Events')          # 事件组上调才是有益重分类

        segs = [(up, col, 0.85 if good_is_up else 0.28),
                (same, '#CCCCCC', 0.65),
                (dn, col, 0.28 if good_is_up else 0.85)]
        left = 0.0
        for val, scol, alpha in segs:
            if n == 0:
                continue
            frac = val / n
            ax.barh(ypos[gi], frac, left=left, height=bar_h, color=scol,
                    alpha=alpha, edgecolor='white', linewidth=0.5, zorder=3)
            if frac > 0.10:
                ax.text(left + frac / 2, ypos[gi], f'{val} ({frac:.0%})',
                        ha='center', va='center', fontsize=6.0, color='#222222')
            left += frac

        good = up if good_is_up else dn
        bad = dn if good_is_up else up
        txt = (f'n = {n}\nCorrect: {good} ({good / n:.1%})\nWrong: {bad} ({bad / n:.1%})'
               if n else 'n = 0')
        ax.text(1.03, ypos[gi], txt, ha='left', va='center',
                fontsize=5.8, color='#444444', linespacing=1.3)

        _INCR_ROWS_RECLASS.append({
            'Comparison': record_tag or os.path.basename(save_path),
            'Index': f'Reclassification counts — {gname}',
            'Index_code': f'reclass_{gname.lower()}',
            'Value': np.nan, 'CI_lo': np.nan, 'CI_hi': np.nan, 'p_value': np.nan,
            'CI_method': 'counts at locked threshold',
            'Threshold': threshold, 'N': n,
            'Reclassified_up': up, 'Reclassified_down': dn, 'Unchanged': same,
        })

    ax.set_yticks(ypos)
    ax.set_yticklabels([g[0] for g in groups])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, len(groups) - 0.45)
    ax.set_xlabel(f'Proportion reclassified at threshold = {threshold:.3f}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.13, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)

    handles = [Rectangle((0, 0), 1, 1, facecolor='#777777', alpha=0.85,
                         edgecolor='none', label='Correct direction'),
               Rectangle((0, 0), 1, 1, facecolor='#CCCCCC', alpha=0.65,
                         edgecolor='none', label='Unchanged'),
               Rectangle((0, 0), 1, 1, facecolor='#777777', alpha=0.28,
                         edgecolor='none', label='Wrong direction')]
    sci_legend(ax, handles=handles, loc='upper center',
               bbox_to_anchor=(0.42, -0.34), ncol=3, fontsize=5.8)
    if note:
        ax.text(0.0, 1.06, note, transform=ax.transAxes, fontsize=6.2,
                color='#555555', va='bottom', ha='left')

    plt.tight_layout()
    _supp_save(save_path)
    return fig


# ==========================================================================================
# 10.5 单个「比较」的完整评估：绝对指标 + Δ指标(95%CI) + 重分类 + 决策曲线数字
# ==========================================================================================
# Δ 指标在森林图中的分栏（量纲差异大，必须分栏，否则小量纲指标会被压成一个点）
_INCR_PANEL_OF = {
    'AUROC':           'Discrimination and overall accuracy',
    'AP':              'Discrimination and overall accuracy',
    'Brier':           'Discrimination and overall accuracy',
    'Scaled_Brier':    'Discrimination and overall accuracy',
    'Cal_slope':       'Calibration',
    'Cal_intercept':   'Calibration',
    'OE':              'Calibration',
    'NB_at_threshold': 'Clinical utility (decision curve)',
}


def _incr_multimetric_rows(dlt, cohort, metrics=None):
    """把一个队列的 Δ 汇总转成森林图行（带 panel 分栏）"""
    metrics = metrics or ['AUROC', 'AP', 'Brier', 'Scaled_Brier',
                          'Cal_slope', 'Cal_intercept', 'OE', 'NB_at_threshold']
    return [{'label': _INCR_METRIC_LABEL[m], 'cohort': cohort,
             'panel': _INCR_PANEL_OF[m], 'delta': dlt[m]['delta'],
             'lo': dlt[m]['lo'], 'hi': dlt[m]['hi'], 'p': dlt[m]['p']}
            for m in metrics if m in dlt]


def _incr_evaluate_pair(comparison, cohort, y, p_base, p_new, threshold,
                        framework, model_desc, base_label, new_label,
                        n_boot=None, record_dca=True):
    """
    对「同一队列、同一批患者的两套预测概率」做全套比较，并把结果写进各收集器。
    返回 (delta_summary, delta_samples, delong_result)
    """
    y = np.asarray(y, dtype=int)
    p_base = np.asarray(p_base, dtype=float)
    p_new = np.asarray(p_new, dtype=float)

    abs_sum, dlt_sum, dlt_samp, pt_base, pt_new = _incr_paired_bootstrap(
        y, p_base, p_new, threshold, n_boot=n_boot)
    dl = delong_roc_test(y, p_base, p_new)

    # ---- 绝对指标表（两行：base / extended）----
    for tag, label, pt, ab in [('Base', base_label, pt_base, abs_sum['base']),
                               ('Extended', new_label, pt_new, abs_sum['new'])]:
        row = {
            'Comparison': comparison, 'Cohort': cohort, 'Role': tag,
            'Model': label, 'Framework': framework, 'Model_specification': model_desc,
            'N': pt['N'], 'Events': pt['Events'], 'Event_rate': pt['Event_rate'],
            'Threshold': threshold,
        }
        for m in _INCR_BOOT_METRICS:
            v, lo, hi = ab[m]
            row[m] = v
            row[f'{m}_CI_lo'] = lo
            row[f'{m}_CI_hi'] = hi
        for extra in ['HL_chi2', 'HL_p', 'TP', 'FP', 'TN', 'FN',
                      'Sensitivity', 'Specificity', 'PPV', 'NPV']:
            row[extra] = pt.get(extra, np.nan)
        # AUROC 的解析 CI（DeLong）与 bootstrap CI 并列，便于互相核对
        row['AUROC_DeLong_CI_lo'] = dl['AUROC_base_lo'] if tag == 'Base' else dl['AUROC_new_lo']
        row['AUROC_DeLong_CI_hi'] = dl['AUROC_base_hi'] if tag == 'Base' else dl['AUROC_new_hi']
        _INCR_ROWS_METRICS.append(row)

    # ---- Δ 指标表（审稿问题 5 的核心）----
    for m in _INCR_BOOT_METRICS:
        d = dlt_sum[m]
        higher_better = _INCR_HIGHER_BETTER.get(m)
        if higher_better is None:
            favours = 'n/a (target value, not "higher is better")'
        elif (higher_better and d['delta'] > 0) or ((not higher_better) and d['delta'] < 0):
            favours = 'extended (with intraoperative)'
        else:
            favours = 'base (preoperative only)'
        _INCR_ROWS_DELTAS.append({
            'Comparison': comparison, 'Cohort': cohort, 'Framework': framework,
            'Metric': m, 'Metric_label': _INCR_METRIC_LABEL[m],
            'Value_base': pt_base[m], 'Value_extended': pt_new[m],
            'Delta': d['delta'], 'CI_lo': d['lo'], 'CI_hi': d['hi'],
            'p_value': d['p'],
            'CI_method': f"paired stratified bootstrap ({d['n_boot_used']} resamples)",
            'Favours': favours,
            'CI_excludes_zero': (np.isfinite(d['lo']) and np.isfinite(d['hi'])
                                 and not (d['lo'] <= 0 <= d['hi'])),
            'N': pt_base['N'], 'Events': pt_base['Events'], 'Threshold': threshold,
        })

    # ΔAUROC 追加一行 DeLong 解析结果（与 bootstrap 互为印证）
    _INCR_ROWS_DELTAS.append({
        'Comparison': comparison, 'Cohort': cohort, 'Framework': framework,
        'Metric': 'AUROC', 'Metric_label': 'ΔAUROC',
        'Value_base': dl['AUROC_base'], 'Value_extended': dl['AUROC_new'],
        'Delta': dl['Delta_AUROC'], 'CI_lo': dl['CI_lo'], 'CI_hi': dl['CI_hi'],
        'p_value': dl['p_value'], 'CI_method': 'DeLong test (correlated ROC curves)',
        'Favours': ('extended (with intraoperative)' if dl['Delta_AUROC'] > 0
                    else 'base (preoperative only)'),
        'CI_excludes_zero': (np.isfinite(dl['p_value']) and dl['p_value'] < 0.05),
        'N': pt_base['N'], 'Events': pt_base['Events'], 'Threshold': threshold,
    })

    # ---- 重分类（cfNRI / 分类 NRI / IDI）----
    nri = _incr_nri_idi(y, p_base, p_new, threshold, n_boot=n_boot)
    for key, lbl in [('cfNRI', 'Continuous (category-free) NRI'),
                     ('cfNRI_ev', '  cfNRI — events component'),
                     ('cfNRI_ne', '  cfNRI — non-events component'),
                     ('catNRI', 'Categorical NRI (at locked threshold)'),
                     ('catNRI_ev', '  Categorical NRI — events component'),
                     ('catNRI_ne', '  Categorical NRI — non-events component'),
                     ('IDI', 'IDI'),
                     ('IDI_ev', '  IDI — events component'),
                     ('IDI_ne', '  IDI — non-events component')]:
        r = nri[key]
        _INCR_ROWS_RECLASS.append({
            'Comparison': comparison, 'Cohort': cohort, 'Framework': framework,
            'Index': lbl, 'Index_code': key,
            'Value': r['value'], 'CI_lo': r['lo'], 'CI_hi': r['hi'], 'p_value': r['p'],
            'CI_method': 'paired stratified bootstrap',
            'Threshold': threshold, 'N': len(y), 'Events': int(y.sum()),
            **{f'n_{k}': v for k, v in nri['_counts'].items()},
        })

    # ---- 决策曲线的数字底稿（全阈值净获益）----
    if record_dca:
        prev = float(y.mean())
        for t in np.arange(0.01, min(INCR_DCA_XMAX, 0.99) + 1e-9, 0.01):
            _INCR_ROWS_DCA.append({
                'Comparison': comparison, 'Cohort': cohort, 'Framework': framework,
                'Threshold_probability': round(float(t), 4),
                'NB_base': calculate_net_benefit(y, p_base, t),
                'NB_extended': calculate_net_benefit(y, p_new, t),
                'Delta_NB': (calculate_net_benefit(y, p_new, t)
                             - calculate_net_benefit(y, p_base, t)),
                'NB_treat_all': prev - (1 - prev) * (t / (1 - t)),
                'NB_treat_none': 0.0,
                'High_risk_per_1000_base': calculate_n_high_risk(p_base, t, len(y)),
                'High_risk_per_1000_extended': calculate_n_high_risk(p_new, t, len(y)),
            })

    print(f"    {cohort:<12} {comparison:<46} "
          f"ΔAUROC = {dl['Delta_AUROC']:+.4f} "
          f"(95% CI {dl['CI_lo']:+.4f} to {dl['CI_hi']:+.4f}, DeLong p = {_fmt_p(dl['p_value'])}) | "
          f"ΔBrier = {dlt_sum['Brier']['delta']:+.4f} | "
          f"ΔNB = {dlt_sum['NB_at_threshold']['delta']:+.4f}")

    return dlt_sum, dlt_samp, dl


def _incr_pair_figures(cohort, y, p_base, p_new, threshold,
                       label_base, label_new, coh_txt, prefix='S5'):
    """一个队列下的成套对比图：校准 / 决策 / ROC / PR / 一致性 / 重分类"""
    note_txt = f'{coh_txt}\nN = {len(y)}, events = {int(np.sum(y))}'
    series = [
        {'y': y, 'prob': p_base, 'label': label_base,
         'color': INCR_MODEL_COLOR['base'], 'marker': 'o'},
        {'y': y, 'prob': p_new, 'label': label_new,
         'color': INCR_MODEL_COLOR['new'], 'marker': 's'},
    ]

    plot_calibration_multi(
        series, f'{INCR_DIR}/{prefix}C_Calibration_{cohort}.png',
        n_bootstrap=INCR_N_BOOT_CAL, note=note_txt,
        record_tag=f'{prefix}C_Calibration_{cohort}')

    plot_dca_multi(
        [{k: v for k, v in s.items() if k != 'marker'} for s in series],
        f'{INCR_DIR}/{prefix}D_DCA_{cohort}.png',
        locked_threshold=threshold, n_bootstrap=INCR_N_BOOT_DCA,
        record_tag=f'{prefix}D_DCA_{cohort}')

    plot_incr_roc_compare(
        y, p_base, p_new, 'Preoperative', 'Pre + intraoperative',
        f'{INCR_DIR}/{prefix}E_ROC_{cohort}.png', note=note_txt,
        record_tag=f'{prefix}E_ROC_{cohort}')

    plot_incr_pr_compare(
        y, p_base, p_new, 'Preoperative', 'Pre + intraoperative',
        f'{INCR_DIR}/{prefix}F_PR_{cohort}.png', note=note_txt,
        record_tag=f'{prefix}F_PR_{cohort}')

    plot_agreement_scatter(
        p_base, p_new, y,
        xlabel='Preoperative model — predicted risk',
        ylabel='Pre + intraoperative model — predicted risk',
        save_path=f'{INCR_DIR}/{prefix}G_Agreement_{cohort}.png',
        threshold=threshold, record_tag=f'{prefix}G_Agreement_{cohort}')

    plot_incr_reclass_flow(
        y, p_base, p_new, threshold,
        f'{INCR_DIR}/{prefix}H_Reclassification_{cohort}.png',
        note=note_txt.replace('\n', ' · '),
        record_tag=f'{prefix}H_Reclassification_{cohort}')



# ==========================================================================================
# 10.7 S5 主流程
# ==========================================================================================
def run_supp_S5_intraoperative_increment():
    set_sci_style()
    os.makedirs(INCR_DIR, exist_ok=True)

    fa = _incr_load_framework_A()
    df_dev, df_ext, intra_src = _incr_load_intraop_frames()

    headline = {}            # 结论守卫用（取「联合加入」的主分析结果）
    dAUC_samples = {}        # ΔAUROC 的 bootstrap 分布（图 S5I）
    forest_auroc_rows = []   # 图 S5A：逐项 vs 联合
    multimetric_rows = []    # 图 S5B：多指标 Δ（两队列合并成一张图）
    has_external = False

    # ---------------------------------------------------------------
    # A. Framework A —— 全流水线重训的「联合加入」模型（主分析）
    # ---------------------------------------------------------------
    if fa is not None:
        print("\n  [S5-A] Framework A：术前模型 vs 术前+术中模型"
              "（三个术中变量联合加入，与主模型同源的完整建模流水线）")
        model_desc = (f"Same nested-CV pipeline as the primary model "
                      f"(preoperative best model = {fa['preop_model_name']}; "
                      f"pre + intraoperative best model = {fa['intraop_model_name']}); "
                      f"all three intraoperative variables entered jointly")
        thr = clinical_threshold      # 固定在术前模型的锁定阈值上比较，保证同一操作点

        cohorts = [('Development', fa['y_dev'], fa['p_dev_base'], fa['p_dev_new'],
                    'Internal validation (OOF)')]
        if fa['y_ext'] is not None and len(fa['y_ext']) > 0:
            cohorts.append(('External', fa['y_ext'], fa['p_ext_base'], fa['p_ext_new'],
                            'External validation'))
            has_external = True

        for cname, yy, pb, pn, coh_txt in cohorts:
            dlt, samp, dl = _incr_evaluate_pair(
                comparison='Joint (all 3 intraoperative) — full re-training',
                cohort=cname, y=yy, p_base=pb, p_new=pn, threshold=thr,
                framework='A: full pipeline re-training',
                model_desc=model_desc,
                base_label='Preoperative model',
                new_label='Pre + intraoperative model')

            headline[cname] = {'dAUROC': dl['Delta_AUROC'], 'lo': dl['CI_lo'],
                               'hi': dl['CI_hi'], 'p': dl['p_value'],
                               'dBrier': dlt['Brier']['delta'],
                               'dNB': dlt['NB_at_threshold']['delta']}
            dAUC_samples[cname] = samp['AUROC']
            forest_auroc_rows.append({
                'label': INCR_SHORT_LABEL['Preop_Intraop'], 'cohort': cname,
                'delta': dl['Delta_AUROC'], 'lo': dl['CI_lo'], 'hi': dl['CI_hi'],
                'p': dl['p_value']})
            multimetric_rows += _incr_multimetric_rows(dlt, cname)

            _incr_pair_figures(cname, yy, pb, pn, thr,
                               'Preoperative model', 'Pre + intraoperative model',
                               coh_txt)

    # ---------------------------------------------------------------
    # B. Framework B 
    # ---------------------------------------------------------------
    if df_dev is not None:
        print("\n  [S5-B] Framework B：以术前模型线性预测值为锚的增量 logistic 回归"
              "（逐项加入 + 联合加入）")

        y_dev_b = np.asarray(y_int, dtype=int)
        y_ext_b = np.asarray(y_ext, dtype=int)
        ok_dev = (len(df_dev) == len(y_dev_b))
        ok_ext = (len(df_ext) == len(y_ext_b))
        if not ok_dev:
            print(f"  ⚠️ 开发队列行数不匹配（原始表 {len(df_dev)} vs OOF 预测 {len(y_dev_b)}），"
                  f"跳过 Framework B")
        else:
            lp_dev = _incr_logit(prob_int)                 # 开发队列锚 = OOF 概率
            lp_ext = _incr_logit(prob_ext) if ok_ext else None
            if not ok_ext:
                print(f"  ⚠️ 外部队列行数不匹配（原始表 {len(df_ext)} vs 预测 {len(y_ext_b)}），"
                      f"Framework B 仅在开发队列进行")

            var_sets = [(f'Add_{v}', [v]) for v in INCR_VARS] + \
                       [('Add_All_Intraop', list(INCR_VARS))]

            for set_code, varlist in var_sets:
                miss = [v for v in varlist if v not in df_dev.columns]
                if miss:
                    print(f"  ⚠️ 缺少变量 {miss}，跳过 {set_code}")
                    continue

                Zd, Ze, cols, is_bin, units = _incr_build_Z(df_dev, df_ext, varlist)
                if Zd is None:
                    continue

                res = _incr_fit_framework_B(
                    lp_dev, Zd, y_dev_b, lp_ext,
                    (Ze if ok_ext else np.zeros((1, Zd.shape[1]))),
                    is_bin, cols)

                joint = (len(varlist) > 1)
                comparison = ('Joint (all 3 intraoperative) — LP-anchored' if joint
                              else f'Individual: + {rename_feature(varlist[0])}')
                model_desc = ('Logistic regression: logit(preoperative predicted risk) as '
                              'anchor covariate + '
                              + ', '.join(rename_feature(c) for c in cols)
                              + '; median imputation and standardisation of continuous '
                                'covariates fitted inside each cross-validation fold')

                # ---- 系数 / LRT 表 ----
                for cr, un in zip(res['coef_rows'], units):
                    _INCR_ROWS_COEF.append({
                        'Comparison': comparison,
                        'Entry': 'Joint' if joint else 'Individual',
                        'Cohort': 'Development (full-cohort fit)',
                        'Framework': 'B: LP-anchored incremental logistic',
                        **cr, 'Unit': un,
                        'LRT_chi2': res['lrt']['chi2'], 'LRT_df': res['lrt']['df'],
                        'LRT_p': res['lrt']['p'],
                        'Anchor_beta_logit_preop': res['anchor_beta'],
                        'N': res['n_dev'], 'Events': int(y_dev_b.sum()),
                    })
                print(f"    · {comparison:<46} LRT χ²({res['lrt']['df']}) = "
                      f"{res['lrt']['chi2']:.2f}, p = {_fmt_p(res['lrt']['p'])}")

                # ---- 开发队列（OOF）----
                dlt_d, samp_d, dl_d = _incr_evaluate_pair(
                    comparison=comparison, cohort='Development',
                    y=y_dev_b, p_base=res['oof_base'], p_new=res['oof_new'],
                    threshold=clinical_threshold,
                    framework='B: LP-anchored incremental logistic',
                    model_desc=model_desc,
                    base_label='Preoperative model (anchored)',
                    new_label=comparison, record_dca=joint)
                forest_auroc_rows.append({
                    'label': INCR_SHORT_LABEL.get(set_code, comparison),
                    'cohort': 'Development', 'delta': dl_d['Delta_AUROC'],
                    'lo': dl_d['CI_lo'], 'hi': dl_d['CI_hi'], 'p': dl_d['p_value']})

                # ---- 外部队列 ----
                dlt_e = dl_e = samp_e = None
                if ok_ext and res['ext_base'] is not None:
                    has_external = True
                    dlt_e, samp_e, dl_e = _incr_evaluate_pair(
                        comparison=comparison, cohort='External',
                        y=y_ext_b, p_base=res['ext_base'], p_new=res['ext_new'],
                        threshold=clinical_threshold,
                        framework='B: LP-anchored incremental logistic',
                        model_desc=model_desc,
                        base_label='Preoperative model (anchored)',
                        new_label=comparison, record_dca=joint)
                    forest_auroc_rows.append({
                        'label': INCR_SHORT_LABEL.get(set_code, comparison),
                        'cohort': 'External', 'delta': dl_e['Delta_AUROC'],
                        'lo': dl_e['CI_lo'], 'hi': dl_e['CI_hi'], 'p': dl_e['p_value']})

                # 若 Framework A 不可用，用「联合加入」的 Framework B 结果充当结论依据，
                # 并补齐校准 / 决策曲线等成套图，保证审稿人要的图在任何情况下都有。
                if joint and fa is None:
                    headline['Development'] = {
                        'dAUROC': dl_d['Delta_AUROC'], 'lo': dl_d['CI_lo'],
                        'hi': dl_d['CI_hi'], 'p': dl_d['p_value'],
                        'dBrier': dlt_d['Brier']['delta'],
                        'dNB': dlt_d['NB_at_threshold']['delta']}
                    dAUC_samples['Development'] = samp_d['AUROC']
                    multimetric_rows += _incr_multimetric_rows(dlt_d, 'Development')
                    _incr_pair_figures(
                        'Development', y_dev_b, res['oof_base'], res['oof_new'],
                        clinical_threshold, 'Preoperative model (anchored)',
                        'Pre + intraoperative model', 'Internal validation (OOF)')

                    if ok_ext and dlt_e is not None:
                        headline['External'] = {
                            'dAUROC': dl_e['Delta_AUROC'], 'lo': dl_e['CI_lo'],
                            'hi': dl_e['CI_hi'], 'p': dl_e['p_value'],
                            'dBrier': dlt_e['Brier']['delta'],
                            'dNB': dlt_e['NB_at_threshold']['delta']}
                        dAUC_samples['External'] = samp_e['AUROC']
                        multimetric_rows += _incr_multimetric_rows(dlt_e, 'External')
                        _incr_pair_figures(
                            'External', y_ext_b, res['ext_base'], res['ext_new'],
                            clinical_threshold, 'Preoperative model (anchored)',
                            'Pre + intraoperative model', 'External validation')

    # ---------------------------------------------------------------
    # C. 汇总图：S5A 逐项 vs 联合、S5B 多指标 Δ、S5I bootstrap 分布
    # ---------------------------------------------------------------
    if forest_auroc_rows:
        order = [INCR_SHORT_LABEL['Add_Surgery_Duration'],
                 INCR_SHORT_LABEL['Add_PFCL'],
                 INCR_SHORT_LABEL['Add_Phacovitrectomy'],
                 INCR_SHORT_LABEL['Add_All_Intraop'],
                 INCR_SHORT_LABEL['Preop_Intraop']]
        rank = {lab: i for i, lab in enumerate(order)}
        forest_auroc_rows.sort(key=lambda r: (rank.get(r['label'], 99),
                                              0 if r['cohort'] == 'Development' else 1))
        print("\n  · S5A 逐项 vs 联合 ΔAUROC 森林图")
        plot_delta_forest(
            forest_auroc_rows, f'{INCR_DIR}/S5A_Incremental_AUROC_Forest.png',
            xlabel='ΔAUROC versus preoperative model',
            rope=(-INCR_ROPE_AUROC, INCR_ROPE_AUROC),
            title_note='Intraoperative variables entered individually and jointly',
            record_tag='S5A_Incremental_AUROC_Forest')

    if multimetric_rows:
        print("  · S5B 多指标 Δ 森林图（AUROC / AP / Brier / 校准 / 净获益）")
        plot_delta_forest(
            multimetric_rows, f'{INCR_DIR}/S5B_DeltaMetrics_Forest.png',
            xlabel='Δ (pre + intraoperative − preoperative)',
            rope=(-INCR_ROPE_AUROC, INCR_ROPE_AUROC),
            rope_panels=['Discrimination and overall accuracy'],
            title_note=(f'Joint model — paired stratified bootstrap, '
                        f'{INCR_N_BOOT} resamples'),
            record_tag='S5B_DeltaMetrics_Forest')

    if dAUC_samples:
        print("  · S5I ΔAUROC 配对 bootstrap 分布图")
        plot_incr_delta_distribution(
            dAUC_samples, f'{INCR_DIR}/S5I_DeltaAUROC_Bootstrap.png',
            metric_label='ΔAUROC', rope=(-INCR_ROPE_AUROC, INCR_ROPE_AUROC),
            record_tag='S5I_DeltaAUROC_Bootstrap')


    # ---------------------------------------------------------------
    # F. 导出全部 CSV
    # ---------------------------------------------------------------
    exported = []
    for rows, fname in [
            (_INCR_ROWS_SPEC,    f'Table_S5_Analysis_Specification_{TIMESTAMP}.csv'),
            (_INCR_ROWS_DELTAS,  f'Table_S5_Delta_Metrics_{TIMESTAMP}.csv'),
            (_INCR_ROWS_METRICS, f'Table_S5_Absolute_Metrics_{TIMESTAMP}.csv'),
            (_INCR_ROWS_COEF,    f'Table_S5_Coefficients_LRT_{TIMESTAMP}.csv'),
            (_INCR_ROWS_RECLASS, f'Table_S5_Reclassification_{TIMESTAMP}.csv'),
            (_INCR_ROWS_DCA,     f'Table_S5_DecisionCurve_NetBenefit_{TIMESTAMP}.csv')]:
        if not rows:
            continue
        path = os.path.join(INCR_DIR, fname)
        try:
            pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
            exported.append(path)
        except Exception as e:
            print(f"  ⚠️ 导出 {fname} 失败: {e}")

    print("\n  ── S5 导出的详细对比指标 CSV ──")
    for p in exported:
        print(f"     ✓ {p}")


if INCR_ENABLE:
    try:
        _INCR_RESULT = run_supp_S5_intraoperative_increment()
    except FileNotFoundError as e:
        print(f"⚠️ 跳过 S5 术中增量价值分析（缺少文件）: {e}")
        _INCR_RESULT = None
    except Exception as e:
        import traceback
        print(f"⚠️ S5 术中增量价值分析出错: {e}")
        traceback.print_exc()
        _INCR_RESULT = None
else:
    print("ℹ️ INCR_ENABLE = False，已跳过 S5 术中增量价值分析。")
    _INCR_RESULT = None

# S5 复用了 S1–S4 的部分绘图函数，会继续往 _SUPP_STATS_ROWS 追加记录；
# 这里重新导出一次，保证汇总表包含 S5 的行。
if _SUPP_STATS_ROWS:
    try:
        _out_csv = f'{DATA_PATH}/Supp_S1S4_Plot_Stats_{TIMESTAMP}.csv'
        pd.DataFrame(_SUPP_STATS_ROWS).to_csv(_out_csv, index=False, encoding='utf-8-sig')
        print(f"\n  ✓ 补充图统计量汇总（含 S5）已刷新: {_out_csv}")
    except Exception as e:
        print(f"  ⚠️ 统计量汇总刷新失败: {e}")


# ==========================================================
# 十一、完成
# ==========================================================
set_sci_style()
print("\n" + "=" * 60)
print("✅ 全部图形生成完毕")
print("-" * 60)
print("   正文六图  : ROC_Internal / ROC_External / PR_Internal / PR_External /")
print(f"               Calibration_Publication_v2_{TIMESTAMP} / DCA_Analysis")
print("   叠加版    : ROC_Internal_vs_External / PR_Internal_vs_External")
print("               (+ ROC_PR_Internal_vs_External_Metrics.csv)")
print("   混淆矩阵  : Confusion_Matrix_Internal / _External / _Comparison / _Comparison2")
print("   风险分层  : Figure2A_Risk_Gradient / Figure2B_Event_Concentration")
print("   特征稳定性: LASSO_Lollipop_*")
print("   SHAP      : FigG / FigH / Fig3A–3D / FigAL_Dependence_Dev_vs_Ext")
print("   补充图 S1 : S1_Calibration_Internal/_External, S1_DCA_Internal/_External,")
print("               S1_Agreement_Internal/_External")
print("   补充图 S2 : S2_External_Calibration, S2_External_RiskStratification")
print("   补充图 S3 : S3_CompleteCases_Calibration, S3_CompleteCases_DCA")
print("   补充图 S4 : S4_Sensitivity_Calibration_*, S4_Sensitivity_DCA_*,")
print("               S4_Sensitivity_Forest_External")
print("   补充图 S5 : S5A_Incremental_AUROC_Forest, S5B_DeltaMetrics_Forest,")
print("               S5C–S5H (Calibration / DCA / ROC / PR / Agreement / Reclassification),")
print("               S5I_DeltaAUROC_Bootstrap")
print("               Table_S5_*.csv (Analysis_Specification / Delta_Metrics /")
print("               Absolute_Metrics / Coefficients_LRT / Reclassification /")
print("               DecisionCurve_NetBenefit / Conclusion_Statement)")
print("-" * 60)
print("   ⚠️ 全部分析基于原始预测概率与原始锁定阈值。")
print(f"   自检：内部 AUROC={oof_auc:.3f} / AP={oof_ap:.3f}，"
      f"外部 AUROC={ext_auc:.3f} / AP={ext_ap:.3f}")
print("=" * 60)

