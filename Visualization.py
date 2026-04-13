import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from datetime import datetime
from scipy import stats
import shap
from matplotlib import rcParams
from scipy.stats import pearsonr
from itertools import combinations
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
import matplotlib.colors as mcolors
# ==========================================
# 0. 变量名 → SCI 展示标签映射
# ==========================================
LABEL_MAP = {
    # 结局 / 时间变量
    'Recurrence':              'Recurrence',
    'Follow_up_Time':          'Follow-up time (months)',
    'SO_Removal_Time':         'SO removal time (months)',
    'status_cr':               'Competing risk status',
    # 术前基线特征
    'Diabetes':                'Diabetes',
    'AL':                      'Axial length (mm)',
    'BCVA_Pre':                'Preoperative BCVA (logMAR)',
    'Lens_Status_Pre':         'Preoperative lens status',
    'VH':                      'Vitreous hemorrhage',
    'Macular_status':          'Macular status (on/off)',
    'Symptom_Duration':        'Symptom duration (days)',
    'PVR_Grade_Pre':           'Preoperative PVR grade',
    'Choroidal_Detachment':    'Choroidal detachment',
    'RD_Extent':               'RD extent (quadrants)',
    # 裂孔特征
    'number_of_breaks':        'Number of retinal breaks',
    'Largest_Break_Diameter':  'Largest break diameter (DD)',
    'Break_Loc_Inferior':      'Inferior break',
    'Macular_Hole':            'Macular hole',
    'Lattice_Degeneration':    'Lattice degeneration',
    'Atrophic_Holes':          'Atrophic holes',
    # 术中变量
    'Surgery_Duration':        'Surgery duration (min)',
    'PFCL':                    'PFCL use',
    'Phacovitrectomy':         'Phacovitrectomy',
    # SO removal 时变量
    'BCVA_SOR':                'BCVA at SO removal (logMAR)',
    'SO_Emulsification':       'Silicone oil emulsification',
    'Concurrent_Phaco_SOR':    'Concurrent phacoemulsification at SO removal',
    'ERM_SOR':                 'ERM at SO removal',
    'PVR_SOR':                 'PVR at SO removal',
    # 聚类标签
    'Cluster_ID':              'Retinal break phenotype cluster',
    'Break_Cluster':           'Retinal break phenotype cluster',
}

def rename_feature(name):
    """将代码变量名映射为 SCI 展示标签，未在映射表中的保持原样"""
    return LABEL_MAP.get(name, name)

def rename_feature_list(names):
    """批量重命名特征列表"""
    return [rename_feature(n) for n in names]

def rename_df_feature_col(df, col='Feature'):
    """重命名 DataFrame 中的 Feature 列"""
    if col in df.columns:
        df[col] = df[col].map(lambda x: LABEL_MAP.get(x, x))
    return df

def rename_df_columns(df):
    """重命名 DataFrame 的列名（用于特征矩阵）"""
    return df.rename(columns=LABEL_MAP)

# ==========================================
# 1. 核心配置 
# ==========================================
DATA_PATH = "./model_results"
TIMESTAMP = '20260216_1724' 

# 绘图样式配置
DPI = 600
COLOR_INTERNAL = '#2E75B6'
COLOR_EXTERNAL = '#C00000'

# ==========================================
# 极简风格全局设置 
# ==========================================
def set_sci_style():
    """设置全局绘图风格"""
    
    # 1. 字体设置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # PDF/PS 字体嵌入
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # 统一字号 — 顶刊标准: 正文~8pt, 标签~9pt, 标题~10pt
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.title_fontsize'] = 9
    
    # 2. 边框与刻度
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
    
    # 3. 图例：加框线（方便阅读），细边框，无圆角
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = '#999999'
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams['legend.borderpad'] = 0.4
    plt.rcParams['legend.labelspacing'] = 0.3
    plt.rcParams['legend.handlelength'] = 1.8
    plt.rcParams['legend.handletextpad'] = 0.5
    
    # 4. 保存
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05

set_sci_style()

# ------------------------------------------
# 图例辅助函数（统一风格）
# ------------------------------------------
def sci_legend(ax, **kwargs):
    """为指定 ax 添加图例（带细框线）"""
    defaults = dict(
        frameon=True, edgecolor='#999999', fancybox=False,
        framealpha=1.0, borderpad=0.4, labelspacing=0.3,
        handlelength=1.8, handletextpad=0.5
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)

COLORS_LIST = [
    "#E64B35",  # 红色 - 主要对比色
    "#4DBBD5",  # 青色 - 冷色调主色
    "#00A087",  # 绿色 - 中性色
    "#3C5488",  # 深蓝 - 专业色
    "#F39B7F",  # 珊瑚橙 - 柔和对比
    "#8491B4",  # 灰蓝 - 辅助色
    "#91D1C2",  # 薄荷绿 - 清新色
    "#DC0000",  # 深红 - 强调色
    "#7E6148"   # 棕色 - 大地色
]

COLORS_NPG = [
    "#E64B35",  # NPG红
    "#4DBBD5",  # NPG青
    "#00A087",  # NPG绿
    "#3C5488",  # NPG蓝
    "#F39B7F",  # NPG橙
    "#8491B4",  # NPG紫
    "#91D1C2",  # NPG薄荷
    "#DC0000",  # NPG深红
    "#7E6148",  # NPG棕
    "#B09C85"   # 额外：驼色
]

# ==========================================
# 2. 数据加载与阈值确定
# ==========================================
print(f"{'='*60}")
print("📂 正在加载模型分析数据...")
print(f"{'='*60}")

# ------------------------------------------
# 2.1 加载完整模型包 (Model_Package)
# ------------------------------------------
print("\n[1/8] 加载完整模型包...")
pkl_path = f"{DATA_PATH}/Model_Package_{TIMESTAMP}.pkl"
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"❌ 找不到模型包: {pkl_path}")

full_package = joblib.load(pkl_path)

# 提取基础模型信息
best_model = full_package['best_model']
best_model_name = full_package['best_model_name']
optimal_threshold = full_package['optimal_threshold']
preprocessor = full_package['preprocessor']
# ✅ 从模型包中提取原始数据
X_train = full_package['X_train']
y_train = full_package['y_train']
X_ext = full_package['X_external']  # 如果需要
y_ext_true = full_package['y_external']  # 如果需要
print(f"  ✓ 训练集: {X_train.shape}")
print(f"  ✓ 训练集标签: {y_train.shape}")
print(f"  ✓ 外部验证集: {X_ext.shape}")
# 提取校准模型
calibrated_model = full_package.get('calibrated_model', None)

print(f"  ✓ 最佳模型架构: {best_model_name}")
print(f"  ✓ 最优阈值: {optimal_threshold:.4f}")
if calibrated_model:
    print(f"  ✓ 包含校准后模型 (Calibrated Model)")

# ------------------------------------------
# 2.2 加载核心预测数据
# ------------------------------------------
print("\n[2/8] 加载核心预测数据...")

# A. 从 CSV 加载 (用于核对 ID 和 Label)
df_int = pd.read_csv(f"{DATA_PATH}/Internal_Val_OOF_Preds_{best_model_name}_{TIMESTAMP}.csv")
df_ext = pd.read_csv(f"{DATA_PATH}/External_Val_Preds_{best_model_name}_{TIMESTAMP}.csv")
df_cv_detail = pd.read_csv(f"{DATA_PATH}/Internal_Val_CV_Metrics_Detail_{TIMESTAMP}.csv")

# B. 提取真实标签
y_int = df_int['True_Label'].values
y_ext = df_ext['True_Label'].values

# C. 从模型包中提取概率 
# 对应保存代码 Section 7 -> 'predictions' 字典
preds_dict = full_package['predictions']

# 提取校准前的概率 (Uncalibrated)
oof_probs = preds_dict['internal_oof_probs']
ext_probs = preds_dict['external_probs'] 

# 兼容性处理：如果 CSV 中的概率与 Pickle 中的不一致，以 Pickle 为准 (它包含最原始的浮点精度)
prob_int = oof_probs
prob_ext = ext_probs

print(f"  ✓ 内部验证 OOF 预测: {len(df_int)} 例")
print(f"  ✓ 外部验证预测: {len(df_ext)} 例")


# ------------------------------------------
# 2.3 加载 Bootstrap 稳定性分析数据
# ------------------------------------------
print("\n[3/7] 加载 Bootstrap 稳定性分析数据...")
df_bootstrap_summary = pd.read_csv(f"{DATA_PATH}/External_Val_Bootstrap_1000_{TIMESTAMP}.csv", index_col=0)
df_bootstrap_raw = pd.read_csv(f"{DATA_PATH}/External_Val_Bootstrap_RawData_{TIMESTAMP}.csv")
bootstrap_predictions = joblib.load(f"{DATA_PATH}/Bootstrap_Predictions_{TIMESTAMP}.pkl")

print(f"  ✓ Bootstrap 汇总统计: {len(df_bootstrap_summary)} 个指标")
print(f"  ✓ Bootstrap 原始数据: {len(df_bootstrap_raw)} 次采样")
print(f"  ✓ Bootstrap 预测数据: {len(bootstrap_predictions)} 次迭代")

# ------------------------------------------
# 2.4 加载综合指标汇总表
# ------------------------------------------
print("\n[3.5/7] 加载综合指标汇总表...")
comprehensive_summary_path = f"{DATA_PATH}/Comprehensive_Metrics_Summary_{TIMESTAMP}.csv"
if os.path.exists(comprehensive_summary_path):
    df_comprehensive = pd.read_csv(comprehensive_summary_path)
    print(f"  ✓ 综合指标汇总表已加载: {len(df_comprehensive)} 行")
    print(f"    包含指标: {list(df_comprehensive.columns)}")
else:
    df_comprehensive = None
    print("  ⚠️ 综合指标汇总表未找到（可能需重新运行训练代码）")

# 加载所有模型OOF对比表
all_models_oof_path = f"{DATA_PATH}/All_Models_OOF_Comparison_{TIMESTAMP}.csv"
if os.path.exists(all_models_oof_path):
    df_all_models_oof = pd.read_csv(all_models_oof_path)
    print(f"  ✓ 所有模型OOF对比表已加载: {len(df_all_models_oof)} 个模型")
else:
    df_all_models_oof = None
    print("  ⚠️ 所有模型OOF对比表未找到")

# ------------------------------------------
# 2.5 加载错误分析数据
# ------------------------------------------
print("\n[5/8] 加载错误分析数据...")
# 这个依然读取 CSV，因为包含具体的临床特征，CSV 更方便查看
error_file = f"{DATA_PATH}/Error_Deep_Analysis_{TIMESTAMP}.csv"
if os.path.exists(error_file):
    df_error = pd.read_csv(error_file)
    print(f"  ✓ 错误病例分析: {len(df_error)} 例")
    print(f"    - False Positive: {len(df_error[df_error['Error_Type']=='False_Positive'])} 例")
    print(f"    - False Negative: {len(df_error[df_error['Error_Type']=='False_Negative'])} 例")
else:
    print("  ⚠️ 未找到错误分析 CSV 文件")

# ------------------------------------------
# 2.7 加载 DeLong 检验数据包
# ------------------------------------------
print("\n[7/8] 加载 DeLong 检验数据...")
# 对应保存代码 Section 8
try:
    delong_path = f"{DATA_PATH}/DeLong_Data_Package_{TIMESTAMP}.pkl"
    if os.path.exists(delong_path):
        delong_package = joblib.load(delong_path)
        delong_models = delong_package['model_predictions']
        print(f"  ✓ 成功加载 {len(delong_models)} 个模型的预测数据用于 DeLong 检验")
    else:
        print("  ⚠️ DeLong 数据包未找到")
        delong_package = None
except Exception as e:
    print(f"  ⚠️ 加载 DeLong 数据失败: {e}")
    delong_package = None

# ------------------------------------------
# 2.8 阈值确定与验证
# ------------------------------------------
print("\n[8/8] 阈值确定与验证...")

clinical_threshold = optimal_threshold

# 验证 Key 是否存在 (带防御性)
threshold_method = 'Unknown'
if 'cv_results' in full_package and best_model_name in full_package['cv_results']:
    threshold_method = full_package['cv_results'][best_model_name].get('threshold_method', 'Unknown')

print(f"  ✓ 阈值计算方法: {threshold_method}")

# ------------------------------------------
# 2.9 数据完整性检查
# ------------------------------------------
print(f"\n{'='*60}")
print("🔍 数据完整性检查...")
print(f"{'='*60}")

# 检查预测概率长度
assert len(df_int) == len(oof_probs), "❌ 内部验证数据长度不一致"
assert len(df_ext) == len(ext_probs), "❌ 外部验证数据长度不一致"

is_match_int = np.allclose(df_int['Pred_Prob'].values, oof_probs, rtol=1e-4, atol=1e-5)
is_match_ext = np.allclose(df_ext['Pred_Prob'].values, ext_probs, rtol=1e-4, atol=1e-5)

if is_match_int and is_match_ext:
    print("  ✅ CSV 与 Pickle 预测概率一致")
else:
    print("  ⚠️ 警告: CSV 与 Pickle 概率存在微小差异 (通常由 CSV 浮点截断导致，将优先使用 Pickle 数据)")

print(f"\n{'='*60}")
print("📊 数据加载汇总")
print(f"{'='*60}")
print(f"  - 内部验证 (OOF): {len(oof_probs)} 例")
print(f"  - 外部验证: {len(ext_probs)} 例")
print(f"  - 决策阈值: {clinical_threshold:.4f}")
print(f"  - 包含模块: Bootstrap({'✅' if 'bootstrap_analysis' in full_package else '❌'}), "
      f"Balanced({'✅' if 'balanced_subsets_analysis' in full_package else '❌'}), "
      f"DeLong({'✅' if delong_package is not None else '❌'})")

print(f"\n✅ 数据加载完成！准备进行可视化分析...")
print(f"{'='*60}\n")

# ==========================================
# 2.10 概率校准 - 截距校准 (Intercept-Only Recalibration)
# ==========================================

print(f"\n{'='*60}")
print("🔧 概率校准 - 截距校准 (Intercept-Only Recalibration)")
print(f"{'='*60}")

from scipy.special import expit as _expit
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, brier_score_loss

def _safe_logit(p, eps=1e-7):
    """安全的 logit 变换"""
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))

def intercept_only_recalibration(y_true, y_prob, method='mle'):
    """
    截距校准 (Intercept-Only Recalibration)
    固定 slope=1, 仅估计截距偏移量 Δb
    logit(p_calibrated) = logit(p_original) + Δb
    
    Parameters
    ----------
    y_true : array  真实标签 (0/1)
    y_prob : array  原始预测概率
    method : str    'mle' (推荐) | 'prevalence' (快速近似)
    
    Returns
    -------
    delta_b, calibrated_probs, info_dict
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    logit_p = _safe_logit(y_prob)
    
    info = {'method': method, 'n': len(y_true), 'n_events': int(y_true.sum()),
            'prevalence': y_true.mean(), 'original_mean_prob': y_prob.mean()}
    
    if method == 'mle':
        # MLE: 将 logit(p) 作为 offset, 仅拟合截距
        def neg_loglik(delta_b):
            p_cal = _expit(logit_p + delta_b)
            p_cal = np.clip(p_cal, 1e-15, 1 - 1e-15)
            ll = y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal)
            return -np.sum(ll)
        
        result = minimize_scalar(neg_loglik, bounds=(-5, 5), method='bounded')
        delta_b = result.x
        
        # Fisher information → SE
        p_cal = _expit(logit_p + delta_b)
        p_cal_clipped = np.clip(p_cal, 1e-15, 1 - 1e-15)
        fisher_info = np.sum(p_cal_clipped * (1 - p_cal_clipped))
        se_delta_b = 1.0 / np.sqrt(fisher_info) if fisher_info > 0 else np.nan
        
        info['se_delta_b'] = se_delta_b
        info['ci_lower'] = delta_b - 1.96 * se_delta_b
        info['ci_upper'] = delta_b + 1.96 * se_delta_b
        
    elif method == 'prevalence':
        # 快速近似: Δb ≈ logit(observed_prev) - logit(mean_pred)
        obs_prev = np.clip(y_true.mean(), 1e-7, 1 - 1e-7)
        pred_mean = np.clip(y_prob.mean(), 1e-7, 1 - 1e-7)
        delta_b = np.log(obs_prev / (1 - obs_prev)) - np.log(pred_mean / (1 - pred_mean))
        info['se_delta_b'] = np.nan
    else:
        raise ValueError(f"未知方法: {method}")
    
    calibrated_probs = _expit(logit_p + delta_b)
    info['delta_b'] = delta_b
    info['calibrated_mean_prob'] = calibrated_probs.mean()
    
    return delta_b, calibrated_probs, info


def _cal_slope_intercept(y_true, y_prob):
    """计算校准斜率和截距 (logistic recalibration)"""
    from sklearn.linear_model import LogisticRegression
    logit_p = _safe_logit(y_prob).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(logit_p, y_true)
    return lr.coef_[0][0], lr.intercept_[0]

def _hosmer_lemeshow(y_true, y_prob, n_groups=10):
    """Hosmer-Lemeshow 检验"""
    sorted_idx = np.argsort(y_prob)
    groups = np.array_split(sorted_idx, n_groups)
    chi2_stat = 0
    for g in groups:
        if len(g) == 0: continue
        obs_pos, exp_pos = y_true[g].sum(), y_prob[g].sum()
        obs_neg, exp_neg = len(g) - obs_pos, len(g) - exp_pos
        if exp_pos > 0: chi2_stat += (obs_pos - exp_pos)**2 / exp_pos
        if exp_neg > 0: chi2_stat += (obs_neg - exp_neg)**2 / exp_neg
    p_value = 1 - chi2.cdf(chi2_stat, n_groups - 2)
    return chi2_stat, p_value

def _integrated_calibration_index(y_true, y_prob):
    """ICI (Austin & Steyerberg, Stat Med 2019)"""
    from sklearn.linear_model import LogisticRegression
    logit_p = _safe_logit(y_prob).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(logit_p, y_true)
    p_smooth = lr.predict_proba(logit_p)[:, 1]
    return np.mean(np.abs(p_smooth - y_prob))

def calibration_comparison_report(y_true, prob_orig, prob_cal, label=""):
    """打印校准前后指标对比表"""
    report = {}
    for tag, p in [('Original', prob_orig), ('Calibrated', prob_cal)]:
        slope, intercept = _cal_slope_intercept(y_true, p)
        brier = brier_score_loss(y_true, p)
        auc_val = roc_auc_score(y_true, p)
        eo = p.sum() / y_true.sum() if y_true.sum() > 0 else np.inf
        hl_stat, hl_pval = _hosmer_lemeshow(y_true, p)
        ici = _integrated_calibration_index(y_true, p)
        report[tag] = {'AUC': auc_val, 'Brier': brier, 'Slope': slope, 
                        'Intercept': intercept, 'E/O': eo,
                        'H-L χ²': hl_stat, 'H-L p': hl_pval, 'ICI': ici,
                        'Mean Pred': p.mean()}
    
    print(f"\n  {'─'*62}")
    print(f"  截距校准对比 {f'({label})' if label else ''}")
    print(f"  {'─'*62}")
    print(f"  {'Metric':<16s} {'Original':>12s} {'Calibrated':>12s} {'Change':>12s}")
    print(f"  {'─'*52}")
    for m in ['AUC', 'Brier', 'Slope', 'Intercept', 'E/O', 'H-L χ²', 'H-L p', 'ICI', 'Mean Pred']:
        v0, v1 = report['Original'][m], report['Calibrated'][m]
        ch = v1 - v0
        ok = ''
        if m == 'AUC': ok = ' ✓' if abs(ch) < 0.001 else ''
        elif m in ['Brier', 'ICI', 'H-L χ²']: ok = ' ✓' if ch < 0 else ''
        elif m == 'Slope': ok = ' ✓' if abs(v1 - 1.0) < abs(v0 - 1.0) else ''
        elif m == 'Intercept': ok = ' ✓' if abs(v1) < abs(v0) else ''
        elif m == 'E/O': ok = ' ✓' if abs(v1 - 1.0) < abs(v0 - 1.0) else ''
        elif m == 'H-L p': ok = ' ✓' if v1 > v0 else ''
        print(f"  {m:<16s} {v0:>12.4f} {v1:>12.4f} {ch:>+12.4f}{ok}")
    print(f"  {'─'*62}")
    print(f"  Observed Prevalence: {y_true.mean():.2%}  |  AUC不变 = 判别力保持")
    return report

def bootstrap_delta_b_stability(y_true, y_prob, n_bootstrap=500, ci=95):
    """Bootstrap 评估 Δb 的稳定性"""
    delta_bs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        try:
            db, _, _ = intercept_only_recalibration(y_true[idx], y_prob[idx], method='mle')
            delta_bs.append(db)
        except: continue
    delta_bs = np.array(delta_bs)
    alpha = (100 - ci) / 2
    return {'mean': np.mean(delta_bs), 'std': np.std(delta_bs),
            'ci_lower': np.percentile(delta_bs, alpha),
            'ci_upper': np.percentile(delta_bs, 100 - alpha),
            'n_success': len(delta_bs)}

# ------------------------------------------
# 执行截距校准
# ------------------------------------------
prev_int = y_int.mean()
prev_ext = y_ext.mean()
prev_diff = abs(prev_int - prev_ext)

print(f"\n  内部集阳性率: {prev_int:.2%} ({int(y_int.sum())}/{len(y_int)})")
print(f"  外部集阳性率: {prev_ext:.2%} ({int(y_ext.sum())}/{len(y_ext)})")
print(f"  阳性率差异:   {prev_diff:.2%}")

# --- 内部验证集校准 (训练OOF) ---
print(f"\n  [1/4] 内部验证集截距校准...")
delta_b_int, prob_int_cal, info_int = intercept_only_recalibration(y_int, prob_int, method='mle')
print(f"    Δb(internal) = {delta_b_int:.4f}")
if not np.isnan(info_int.get('se_delta_b', np.nan)):
    print(f"    95% CI: [{info_int['ci_lower']:.4f}, {info_int['ci_upper']:.4f}]")

# --- 外部验证集校准 ---
# 策略选择: prevalence差距 > 10% → 在外部集上重新估计截距
if prev_diff > 0.10:
    print(f"\n  [2/4] 外部验证集截距校准 (external_update, prevalence差距 > 10%)...")
    delta_b_ext, prob_ext_cal, info_ext = intercept_only_recalibration(y_ext, prob_ext, method='mle')
    cal_strategy = 'external_update'
    print(f"    Δb(external) = {delta_b_ext:.4f}")
    if not np.isnan(info_ext.get('se_delta_b', np.nan)):
        print(f"    95% CI: [{info_ext['ci_lower']:.4f}, {info_ext['ci_upper']:.4f}]")
else:
    print(f"\n  [2/4] 外部验证集截距校准 (train_then_apply, prevalence差距 ≤ 10%)...")
    logit_ext = _safe_logit(prob_ext)
    prob_ext_cal = _expit(logit_ext + delta_b_int)  # 用内部集的Δb
    delta_b_ext = delta_b_int
    cal_strategy = 'train_then_apply'
    print(f"    使用内部集 Δb = {delta_b_int:.4f} 直接应用")

# --- 校准报告 ---
print(f"\n  [3/4] 生成校准对比报告...")
report_int = calibration_comparison_report(y_int, prob_int, prob_int_cal, label="Internal (OOF)")
report_ext = calibration_comparison_report(y_ext, prob_ext, prob_ext_cal, label="External")

# --- Bootstrap 稳定性 ---
print(f"\n  [4/4] Bootstrap Δb 稳定性评估 (n=500)...")
boot_int = bootstrap_delta_b_stability(y_int, prob_int, n_bootstrap=500)
boot_ext = bootstrap_delta_b_stability(y_ext, prob_ext, n_bootstrap=500)
print(f"    Internal Δb: {boot_int['mean']:.4f} [{boot_int['ci_lower']:.4f}, {boot_int['ci_upper']:.4f}]")
print(f"    External Δb: {boot_ext['mean']:.4f} [{boot_ext['ci_lower']:.4f}, {boot_ext['ci_upper']:.4f}]")

# --- 保存校准后概率 ---
df_int_cal = df_int.copy()
df_int_cal['Pred_Prob_Calibrated'] = prob_int_cal
df_int_cal.to_csv(f"{DATA_PATH}/Internal_Val_OOF_Preds_Calibrated_{TIMESTAMP}.csv", index=False)

df_ext_cal = df_ext.copy()
df_ext_cal['Pred_Prob_Calibrated'] = prob_ext_cal
df_ext_cal.to_csv(f"{DATA_PATH}/External_Val_Preds_Calibrated_{TIMESTAMP}.csv", index=False)

print(f"\n  ✓ 校准后概率已保存至 CSV")

# --- 验证: AUC 不变 ---
auc_int_before = roc_auc_score(y_int, prob_int)
auc_int_after  = roc_auc_score(y_int, prob_int_cal)
auc_ext_before = roc_auc_score(y_ext, prob_ext)
auc_ext_after  = roc_auc_score(y_ext, prob_ext_cal)

print(f"\n  ✅ AUC 验证 (应完全不变):")
print(f"    Internal: {auc_int_before:.6f} → {auc_int_after:.6f}  Δ={auc_int_after-auc_int_before:+.6f}")
print(f"    External: {auc_ext_before:.6f} → {auc_ext_after:.6f}  Δ={auc_ext_after-auc_ext_before:+.6f}")

print(f"\n  校准策略: {cal_strategy}")
print(f"\n{'='*60}")
print("✅ 概率校准完成！")
print(f"{'='*60}\n")

# 将原始阈值转换到校准后的概率尺度
from scipy.special import expit as _expit

# 内部验证集的校准阈值
clinical_threshold_cal_int = float(_expit(_safe_logit(clinical_threshold) + delta_b_int))

# 外部验证集的校准阈值  
clinical_threshold_cal_ext = float(_expit(_safe_logit(clinical_threshold) + delta_b_ext))

print(f"  原始阈值:        {clinical_threshold:.4f}")
print(f"  校准后阈值(Int): {clinical_threshold_cal_int:.4f}")
print(f"  校准后阈值(Ext): {clinical_threshold_cal_ext:.4f}")

# ==========================================
# 3. 单模型高级分析图表 (Discriminative & Clinical)
# ==========================================
print(f"{'='*60}")
print("📊 开始绘制单模型高级分析图表...")
print(f"{'='*60}\n")

# --- A. AUC 柱状图 (Internal vs External) + Bootstrap 误差棒 ---
print("[1/4] 绘制 AUC 对比柱状图...")
fig, ax = plt.subplots(figsize=(6, 6))

# 计算 AUC
from sklearn.metrics import roc_auc_score, auc, roc_curve
auc_int_mean = df_cv_detail['AUC'].mean() if 'AUC' in df_cv_detail.columns else roc_auc_score(y_int, prob_int)
auc_int_std = df_cv_detail['AUC'].std() if 'AUC' in df_cv_detail.columns else 0
auc_ext = roc_auc_score(y_ext, prob_ext)

# 从 Bootstrap 获取外部验证的置信区间
auc_ext_ci_lower = df_bootstrap_summary.loc['AUC', '95%_CI_Lower']
auc_ext_ci_upper = df_bootstrap_summary.loc['AUC', '95%_CI_Upper']
auc_ext_std = df_bootstrap_summary.loc['AUC', 'Std']

bars = ax.bar(['Internal (CV)', 'External'], 
               [auc_int_mean, auc_ext], 
               yerr=[auc_int_std, auc_ext_std], 
               capsize=10, 
               color=[COLOR_INTERNAL, COLOR_EXTERNAL], 
               alpha=0.8, 
               edgecolor='none', 
               width=0.55)

ax.set_ylim([0.6, 1.0])
ax.set_ylabel('AUC (mean ± SD)')
ax.set_title(f'{best_model_name}: Discrimination')

# 添加数值标注
for bar, val, ci_low, ci_up in zip(bars, 
                                     [auc_int_mean, auc_ext], 
                                     [auc_int_mean - 1.96*auc_int_std, auc_ext_ci_lower],
                                     [auc_int_mean + 1.96*auc_int_std, auc_ext_ci_upper]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.06, 
            f'{val:.3f}\n({ci_low:.3f}–{ci_up:.3f})', 
            ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('figures/AUC_Comparison_Bar.png', dpi=DPI)
print(f"  ✓ 已保存: figures/AUC_Comparison_Bar.png")
plt.close()

# --- A. AUC 柱状图 (Internal vs External) + Bootstrap 误差棒 + DeLong 检验 ---
print("[1/4] 绘制 AUC 对比柱状图 (含 DeLong 检验)...")
fig, ax = plt.subplots(figsize=(5, 5))

# 计算 AUC
from sklearn.metrics import roc_auc_score, auc, roc_curve
auc_int_mean = df_cv_detail['AUC'].mean() if 'AUC' in df_cv_detail.columns else roc_auc_score(y_int, prob_int)
auc_int_std = df_cv_detail['AUC'].std() if 'AUC' in df_cv_detail.columns else 0
auc_ext = roc_auc_score(y_ext, prob_ext)

# 从 Bootstrap 获取外部验证的置信区间
auc_ext_ci_lower = df_bootstrap_summary.loc['AUC', '95%_CI_Lower']
auc_ext_ci_upper = df_bootstrap_summary.loc['AUC', '95%_CI_Upper']
auc_ext_std = df_bootstrap_summary.loc['AUC', 'Std']

bars = ax.bar(['Internal (CV)', 'External'], 
               [auc_int_mean, auc_ext], 
               yerr=[auc_int_std, auc_ext_std], 
               capsize=10, 
               color=[COLOR_INTERNAL, COLOR_EXTERNAL], 
               alpha=0.8, 
               edgecolor='none', 
               width=0.55)

ax.set_ylim([0.6, 1.0])
ax.set_ylabel('AUC (mean ± SD)')
ax.set_title(f'{best_model_name}: Discrimination')

# 添加数值标注
for bar, val, ci_low, ci_up in zip(bars, 
                                     [auc_int_mean, auc_ext], 
                                     [auc_int_mean - 1.96*auc_int_std, auc_ext_ci_lower],
                                     [auc_int_mean + 1.96*auc_int_std, auc_ext_ci_upper]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.06, 
            f'{val:.3f}\n({ci_low:.3f}–{ci_up:.3f})', 
            ha='center', va='bottom', fontsize=7)

# 添加 DeLong 检验结果
if delong_package is not None:
    try:
        # 执行 DeLong 检验比较内部和外部 AUC
        from scipy.stats import norm
        
        def delong_test(y_true1, y_pred1, y_true2, y_pred2):
            """
            DeLong 检验比较两个独立数据集的 AUC
            返回 p-value
            """
            from sklearn.metrics import roc_auc_score
            import numpy as np
            
            # 简化版 DeLong 检验 (使用正态近似)
            auc1 = roc_auc_score(y_true1, y_pred1)
            auc2 = roc_auc_score(y_true2, y_pred2)
            
            # 计算 AUC 的标准误
            n1, n2 = len(y_true1), len(y_true2)
            
            # 使用 Hanley-McNeil 方法估计方差
            def auc_variance(y_true, y_pred):
                auc = roc_auc_score(y_true, y_pred)
                n_pos = np.sum(y_true == 1)
                n_neg = np.sum(y_true == 0)
                
                q1 = auc / (2 - auc)
                q2 = 2 * auc**2 / (1 + auc)
                
                var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + 
                       (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
                return var
            
            var1 = auc_variance(y_true1, y_pred1)
            var2 = auc_variance(y_true2, y_pred2)
            
            # Z 检验
            se = np.sqrt(var1 + var2)
            z = (auc1 - auc2) / se
            p_value = 2 * (1 - norm.cdf(abs(z)))
            
            return p_value, z
        
        # 执行检验
        p_value, z_stat = delong_test(y_int, prob_int, y_ext, prob_ext)
        
        # 确定显著性标记
        if p_value < 0.001:
            sig_mark = '***'
        elif p_value < 0.01:
            sig_mark = '**'
        elif p_value < 0.05:
            sig_mark = '*'
        else:
            sig_mark = 'NS'
        
        # 在两个柱子之间添加显著性标注
        y_max = max(auc_int_mean + 1.96*auc_int_std, auc_ext + 1.96*auc_ext_std)
        y_sig = y_max + 0.15
        
        # 绘制连接线
        ax.plot([0, 1], [y_sig, y_sig], 'k-', linewidth=1.5)
        ax.plot([0, 0], [y_sig - 0.01, y_sig], 'k-', linewidth=1.5)
        ax.plot([1, 1], [y_sig - 0.01, y_sig], 'k-', linewidth=1.5)
        
        # 在图例区域添加 p 值说明（放在左上角）
        ax.text(0.02, 0.98, f'DeLong test: p={p_value:.4f}', 
                transform=ax.transAxes, ha='left', va='top',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
        
        print(f"  ✓ DeLong 检验: p={p_value:.4f}, Z={z_stat:.4f}, 标记={sig_mark}")
        
    except Exception as e:
        print(f"  ⚠️ DeLong 检验失败: {e}")

plt.tight_layout()
plt.savefig('figures/AUC_Comparison_Bar2.png', dpi=DPI)
print(f"  ✓ 已保存: figures/AUC_Comparison_Bar2.png")
plt.close()

# --- B. ROC 与 PR 曲线 ---
print("[2/4] 绘制 ROC 与 PR 曲线...")
from sklearn.metrics import precision_recall_curve

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# ROC 曲线
for y, p, label, col in [(y_int, prob_int, 'Internal', COLOR_INTERNAL), 
                         (y_ext, prob_ext, 'External', COLOR_EXTERNAL)]:
    fpr, tpr, _ = roc_curve(y, p)
    auc_score = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=col, lw=1.8, label=f'{label} (AUC = {auc_score:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=0.8)
axes[0].set_xlabel('1 − Specificity')
axes[0].set_ylabel('Sensitivity')
axes[0].set_title('ROC Curve')
sci_legend(axes[0], loc='lower right')

# PR 曲线
for y, p, label, col in [(y_int, prob_int, 'Internal', COLOR_INTERNAL), 
                         (y_ext, prob_ext, 'External', COLOR_EXTERNAL)]:
    prec, rec, _ = precision_recall_curve(y, p)
    auprc = auc(rec, prec)
    axes[1].plot(rec, prec, color=col, lw=1.8, label=f'{label} (AUPRC = {auprc:.3f})')

axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision–Recall Curve')
sci_legend(axes[1], loc='best')

plt.tight_layout()
plt.savefig('figures/ROC_PR_Combined.png', dpi=DPI)
print(f"  ✓ 已保存: figures/ROC_PR_Combined.png")
plt.close()

# --- C. 校准曲线 (Calibration) ---
print("[3/4] 绘制校准曲线...")
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import chi2
# 数据已在 Section 2 加载完毕，无需重复加载

# ============================================================================
# 校准曲线
# ============================================================================

from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # sigmoid 函数
import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------
# 辅助函数
# ------------------------------------------

def get_cali_stats(y_true, y_prob):
    """计算校准斜率和截距"""
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
    # 初始bin数: 确保每bin有足够样本
    n_bins = min(max_bins, max(5, n // (min_samples * 2)))
    
    # 等频分位数
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
        if n_in < max(5, min_samples // 2):  # 最低门槛
            continue
        
        p_hat = y_true[mask].mean()
        mean_pred = y_prob[mask].mean()
        
        # Wilson 置信区间
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
    """
    Bootstrap 置信带 (基于 logistic recalibration 拟合线)
    """
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
        except:
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


# ============================================================================
# 主绘图函数
# ============================================================================
C_INT      = COLORS_LIST[3]   # #3C5488 深蓝 — 内部验证
C_EXT      = COLORS_LIST[0]   # #E64B35 红色 — 外部验证
C_CORRECT  = COLORS_LIST[2]   # #00A087 绿色 — Optimism-Corrected
C_OOF_LINE = COLORS_LIST[1]   # #4DBBD5 青色 — OOF 原始曲线
C_EXT_LINE = COLORS_LIST[7]   # #DC0000 深红 — 外部原始曲线
C_APPARENT = COLORS_LIST[5]   # #8491B4 灰蓝 — Apparent
C_BRIER    = COLORS_LIST[4]   # #F39B7F 珊瑚橙 — Brier / 第三指标

def plot_calibration_publication(
    y_int, prob_int, y_ext, prob_ext,
    save_path='figures/Calibration_Publication.png',
    dpi=600, n_bootstrap=1000
):


    # ==================== 布局: 主图 + 底部spike图 ====================
    fig = plt.figure(figsize=(3.5, 4.5))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[5.5, 1], hspace=0.05,
        left=0.12, right=0.95, top=0.93, bottom=0.08
    )

    ax_main = fig.add_subplot(gs[0])
    ax_spike = fig.add_subplot(gs[1], sharex=ax_main)

    # ==================== 数据集配置 (与 ROC_CI 同色) ====================
    datasets = [
        {'y': y_int, 'prob': prob_int, 'label': 'Internal',
         'color': C_INT, 'fill': C_INT, 'marker': 'o'},
        {'y': y_ext, 'prob': prob_ext, 'label': 'External',
         'color': C_EXT, 'fill': C_EXT, 'marker': 's'},
    ]

    print("\n" + "="*70)
    print("  Publication-Quality Calibration Plot")
    print("="*70)

    stats_lines = []

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

        # --- B. 自适应分bin ---
        min_per_bin = max(15, int(len(y) * 0.04))
        bins = adaptive_calibration_bins(y, p, min_samples=min_per_bin, max_bins=10)

        # 点大小编码样本量 — 缩小至与 3.5in 画幅匹配的比例
        size_min, size_max = 15, 45
        if len(bins['n']) > 0 and bins['n'].max() > bins['n'].min():
            sizes = size_min + (size_max - size_min) * (bins['n'] - bins['n'].min()) / (bins['n'].max() - bins['n'].min())
        else:
            sizes = np.full(len(bins['n']), (size_min + size_max) / 2)

        # 误差线 — 细化线宽 & cap, 与曲线 lw=1.5 协调
        ax_main.errorbar(
            bins['pred'], bins['true'],
            yerr=[bins['true'] - bins['ci_low'], bins['ci_up'] - bins['true']],
            fmt='none', ecolor=ds['color'], elinewidth=0.6,
            capsize=1.5, capthick=0.6, alpha=0.45, zorder=3
        )

        # 散点 — 缩小尺寸, 细化描边
        ax_main.scatter(
            bins['pred'], bins['true'], s=sizes, marker=ds['marker'],
            facecolors=ds['color'], edgecolors=ds['color'],
            linewidths=0.7, alpha=0.85, zorder=4
        )

        # --- C. Logistic recalibration 拟合曲线 (带 label 供图例使用) ---
        x_fit, y_fit, _, _ = logistic_calibration_curve(y, p)
        ax_main.plot(x_fit, y_fit, color=ds['color'], lw=1.5,
                     alpha=0.8, zorder=5, label=ds['label'])

        # --- D. Bootstrap CI 带 ---
        print(f"    Computing {n_bootstrap} bootstrap CIs...")
        x_ci, ci_low, ci_up = bootstrap_logistic_cal_ci(y, p, n_bootstrap=n_bootstrap)
        ax_main.fill_between(x_ci, ci_low, ci_up, color=ds['color'],
                             alpha=0.15, zorder=1)

        # --- E. 统计信息 ---
        hl_p_str = 'p < 0.001' if hl_pval < 0.001 else f'p = {hl_pval:.3f}'
        stats_lines.append(
            f"{ds['label']}: Slope = {slope:.2f}, Int = {intercept:.2f}, "
            f"Brier = {brier:.3f}"
        )
        stats_lines.append(
            f"  H-L χ² = {hl_stat:.2f}, {hl_p_str}{sig}, "
            f"E/O = {eo_ratio:.2f}"
        )

    # --- F. 理想校准线 ---
    ax_main.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--',
                 zorder=2, label='Ideal')

    # ==================== 主图美化 (与 ROC_CI 一致) ====================
    ax_main.set_xlabel('Predicted probability')
    ax_main.set_ylabel('Observed proportion')
    ax_main.set_xlim([-0.02, 1.02])
    ax_main.set_ylim([-0.02, 1.02])
    ax_main.set_aspect('equal')
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # ---- 图例 → upper left, 避免与右下角统计框重叠 ----
    sci_legend(ax_main, loc='upper left')

    # ---- 统计信息框 → lower right, 与 ROC_CI 图例框同风格 ----
    stats_text = '\n'.join(stats_lines)
    ax_main.text(
        0.98, 0.03, stats_text,
        transform=ax_main.transAxes,
        fontsize=5.5, va='bottom', ha='right',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='white',
            alpha=0.92,
            edgecolor='#808080',
            linewidth=0.8
        )
    )

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


# ============================================================================
# 补充图: 诊断面板
# ============================================================================
def plot_calibration_diagnostic(
    y_int, prob_int, y_ext, prob_ext,
    save_path='figures/Calibration_Diagnostic.png',
    dpi=600
):


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ============ Panel A: 概率分布 ============
    ax = axes[0]
    bin_edges = np.linspace(0, 1, 51)

    for y, p, label, col in [(y_int, prob_int, 'Internal', C_INT),
                              (y_ext, prob_ext, 'External', C_EXT)]:
        ax.hist(p[y == 0], bins=bin_edges, density=True, alpha=0.15,
                color=col, label=f'{label} - Non-events')
        ax.hist(p[y == 1], bins=bin_edges, density=True, alpha=0.6,
                color=col, histtype='step', lw=1.5,
                label=f'{label} - Events')

    # 标注"概率沙漠"
    ax.axvspan(0.3, 0.6, alpha=0.08, color='orange', zorder=0)
    ax.text(0.45, ax.get_ylim()[1] * 0.85, 'Sparse\nZone', ha='center',
            fontsize=10, color='darkorange', fontstyle='italic')

    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title('A. Probability distribution', fontsize=13, loc='left')
    sci_legend(ax, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ============ Panel B: 各区间样本量和阳性率 ============
    ax = axes[1]

    edges = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    x_pos = np.arange(len(edges) - 1)
    bar_width = 0.35

    for idx, (y, p, label, col) in enumerate([
        (y_int, prob_int, 'Internal', C_INT),
        (y_ext, prob_ext, 'External', C_EXT)
    ]):
        n_bins_list = []
        pos_rate_list = []
        for i in range(len(edges) - 1):
            mask = (p >= edges[i]) & (p < edges[i+1]) if i < len(edges)-2 else \
                   (p >= edges[i]) & (p <= edges[i+1])
            n_in = mask.sum()
            pos_rate = y[mask].mean() if n_in > 0 else 0
            n_bins_list.append(n_in)
            pos_rate_list.append(pos_rate)

        offset = -bar_width/2 + idx * bar_width
        bars = ax.bar(x_pos + offset, n_bins_list, bar_width,
                      color=col, alpha=0.6, label=f'{label} (n)')

        for j, (bar, pr, n) in enumerate(zip(bars, pos_rate_list, n_bins_list)):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{pr:.0%}', ha='center', va='bottom', fontsize=7,
                        color=col)

    tick_labels = [f'{edges[i]:.0%}-{edges[i+1]:.0%}' for i in range(len(edges)-1)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8.5)
    ax.set_xlabel('Probability range')
    ax.set_ylabel('n')
    ax.set_title('B. Sample size per bin', fontsize=13, loc='left')
    sci_legend(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ============ Panel C: 校准曲线 + 样本量标注 ============
    ax = axes[2]

    for y, p, label, col, marker in [
        (y_int, prob_int, 'Internal', C_INT, 'o'),
        (y_ext, prob_ext, 'External', C_EXT, 's')
    ]:
        slope, intercept = get_cali_stats(y, p)
        min_per_bin = max(15, int(len(y) * 0.04))
        bins = adaptive_calibration_bins(y, p, min_samples=min_per_bin, max_bins=10)

        # 拟合曲线
        x_fit, y_fit, _, _ = logistic_calibration_curve(y, p)
        ax.plot(x_fit, y_fit, color=col, lw=1.5, alpha=0.7)

        # 散点 + 误差线
        ax.errorbar(bins['pred'], bins['true'],
                    yerr=[bins['true'] - bins['ci_low'], bins['ci_up'] - bins['true']],
                    fmt='none', ecolor=col, elinewidth=1.0, capsize=2.5, alpha=0.4)
        ax.scatter(bins['pred'], bins['true'], s=80, marker=marker,
                   facecolors='white', edgecolors=col, linewidths=1.5, alpha=0.9,
                   label=f'{label} (Slope={slope:.2f})')

        # 标注每个点的样本量
        for xp, yp, n in zip(bins['pred'], bins['true'], bins['n']):
            ax.annotate(f'n={int(n)}', (xp, yp), fontsize=6.5, color=col,
                        textcoords='offset points', xytext=(5, 5), alpha=0.7)

    ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed proportion')
    ax.set_title('C. Calibration with n', fontsize=13, loc='left')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    sci_legend(ax, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\n  ✓ Saved: {save_path} / .pdf")
    plt.close()
    return fig

def plot_calibration_publication_v2(
    y_int, prob_int, y_ext, prob_ext,
    save_path='figures/Calibration_Publication_v2.png',
    dpi=600, n_bootstrap=1000
):

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
    sci_legend(ax_main, loc='lower right', fontsize=5.5)

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
# ============================================================================
# 执行
# ============================================================================
os.makedirs('figures', exist_ok=True)

# 图1:  校准后概率
print("\n[1/5] Publication calibration plot (After Intercept Calibration)...")
plot_calibration_publication_v2(
    y_int, prob_int_cal, y_ext, prob_ext_cal,
    save_path=f'figures/Calibration_Publication_v2_Calibrated_{TIMESTAMP}.png',
    n_bootstrap=1000
)

# 图2: 诊断面板 
print("\n[2/5] Diagnostic panel (After Calibration)...")
plot_calibration_diagnostic(
    y_int, prob_int_cal, y_ext, prob_ext_cal,
    save_path=f'figures/Calibration_Diagnostic_Calibrated_{TIMESTAMP}.png'
)


# 图3: 诊断面板 - 校准前后对比
print("\n[3/5] Diagnostic panel (Before vs After Comparison)...")
def plot_calibration_diagnostic_comparison(
    y_int, prob_int_orig, prob_int_cal, y_ext, prob_ext_orig, prob_ext_cal,
    save_path=None, dpi=600
):


    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for row_idx, (pi, pe, row_label) in enumerate([
        (prob_int_orig, prob_ext_orig, 'Before calibration'),
        (prob_int_cal, prob_ext_cal, 'After Calibration')
    ]):
        # Panel A: 概率分布
        ax = axes[row_idx, 0]
        bin_edges = np.linspace(0, 1, 51)
        for y, p, label, col in [(y_int, pi, 'Internal', C_INT),
                                  (y_ext, pe, 'External', C_EXT)]:
            ax.hist(p[y == 0], bins=bin_edges, density=True, alpha=0.15,
                    color=col, label=f'{label} - Non-events')
            ax.hist(p[y == 1], bins=bin_edges, density=True, alpha=0.6,
                    color=col, histtype='step', lw=1.5,
                    label=f'{label} - Events')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        ax.set_title(f'A{row_idx+1}. Probability Distribution ({row_label})', fontsize=12,
                     loc='left')
        sci_legend(ax, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel B: 各区间样本量
        ax = axes[row_idx, 1]
        edges = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
        x_pos = np.arange(len(edges) - 1)
        bar_width = 0.35
        for idx2, (y, p, label, col) in enumerate([
            (y_int, pi, 'Internal', C_INT),
            (y_ext, pe, 'External', C_EXT)
        ]):
            n_bins_list = []
            pos_rate_list = []
            for i in range(len(edges) - 1):
                mask = (p >= edges[i]) & (p < edges[i+1]) if i < len(edges)-2 else \
                       (p >= edges[i]) & (p <= edges[i+1])
                n_in = mask.sum()
                pos_rate = y[mask].mean() if n_in > 0 else 0
                n_bins_list.append(n_in)
                pos_rate_list.append(pos_rate)
            offset = -bar_width/2 + idx2 * bar_width
            bars = ax.bar(x_pos + offset, n_bins_list, bar_width,
                          color=col, alpha=0.6, label=f'{label} (n)')
            for j, (bar, pr, n) in enumerate(zip(bars, pos_rate_list, n_bins_list)):
                if n > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                            f'{pr:.0%}', ha='center', va='bottom', fontsize=6,
                            color=col)
        tick_labels = [f'{edges[i]:.0%}-{edges[i+1]:.0%}' for i in range(len(edges)-1)]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7.5)
        ax.set_xlabel('Probability range')
        ax.set_ylabel('n')
        ax.set_title(f'B{row_idx+1}. Sample Size ({row_label})', fontsize=12, loc='left')
        sci_legend(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel C: 校准曲线
        ax = axes[row_idx, 2]
        for y, p, label, col, marker in [
            (y_int, pi, 'Internal', C_INT, 'o'),
            (y_ext, pe, 'External', C_EXT, 's')
        ]:
            slope, intercept = get_cali_stats(y, p)
            min_per_bin = max(15, int(len(y) * 0.04))
            bins = adaptive_calibration_bins(y, p, min_samples=min_per_bin, max_bins=10)
            x_fit, y_fit, _, _ = logistic_calibration_curve(y, p)
            ax.plot(x_fit, y_fit, color=col, lw=1.5, alpha=0.7)
            ax.errorbar(bins['pred'], bins['true'],
                        yerr=[bins['true'] - bins['ci_low'], bins['ci_up'] - bins['true']],
                        fmt='none', ecolor=col, elinewidth=1.0, capsize=2.5, alpha=0.4)
            ax.scatter(bins['pred'], bins['true'], s=80, marker=marker,
                       facecolors='white', edgecolors=col, linewidths=1.5, alpha=0.9,
                       label=f'{label} (Slope={slope:.2f})')
        ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed proportion')
        ax.set_title(f'C{row_idx+1}. Calibration ({row_label})', fontsize=12, loc='left')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect('equal')
        sci_legend(ax, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Calibration Diagnostic: Before vs After Intercept-Only Recalibration',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path} / .pdf")
    plt.close()

plot_calibration_diagnostic_comparison(
    y_int, prob_int, prob_int_cal, y_ext, prob_ext, prob_ext_cal,
    save_path=f'figures/Calibration_Diagnostic_BeforeAfter_{TIMESTAMP}.png'
)

# 图4: 校准前后叠加对比图
print("\n[4/4] Before vs After overlay plot (External)...")

def plot_calibration_before_after_overlay(
    y_true, prob_orig, prob_cal, label_set="External",
    save_path=None, dpi=600, n_bootstrap=500
):

    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    COLOR_ORIG = '#7B2D8B'     # 深紫 - 校准前主线
    COLOR_CAL = '#1B7837'      # 深绿 - 校准后主线

    fig = plt.figure(figsize=(8, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5.5, 1], hspace=0.05,
                           left=0.12, right=0.95, top=0.93, bottom=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_spike = fig.add_subplot(gs[1], sharex=ax_main)

    datasets = [
        {'p': prob_orig, 'label': 'Before calibration',
         'color': COLOR_ORIG, 'marker': 's'},
        {'p': prob_cal, 'label': 'After calibration',
         'color': COLOR_CAL, 'marker': 'o'},
    ]

    stats_lines = []

    for ds in datasets:
        p = ds['p']
        slope, intercept = get_cali_stats(y_true, p)
        brier = brier_score_loss(y_true, p)
        hl_stat, hl_pval = hosmer_lemeshow_test(y_true, p, n_groups=10)
        eo = p.sum() / y_true.sum() if y_true.sum() > 0 else np.inf

        # 自适应分bin
        min_per_bin = max(15, int(len(y_true) * 0.04))
        bins = adaptive_calibration_bins(y_true, p, min_samples=min_per_bin, max_bins=10)

        if len(bins['pred']) > 0:
            sizes = np.full(len(bins['n']), 80)
            if len(bins['n']) > 1 and bins['n'].max() > bins['n'].min():
                sizes = 50 + 130 * (bins['n'] - bins['n'].min()) / (bins['n'].max() - bins['n'].min())

            ax_main.errorbar(bins['pred'], bins['true'],
                             yerr=[bins['true'] - bins['ci_low'], bins['ci_up'] - bins['true']],
                             fmt='none', ecolor=ds['color'], elinewidth=1.0, capsize=3, alpha=0.4, zorder=3)
            ax_main.scatter(bins['pred'], bins['true'], s=sizes, marker=ds['marker'],
                            facecolors=ds['color'], edgecolors=ds['color'],
                            linewidths=1.3, alpha=0.85, zorder=4)

        # Logistic recalibration 拟合线
        x_fit, y_fit, _, _ = logistic_calibration_curve(y_true, p)
        ax_main.plot(x_fit, y_fit, color=ds['color'], lw=1.5, alpha=0.8, zorder=5)

        # Bootstrap CI
        x_ci, ci_low, ci_up = bootstrap_logistic_cal_ci(y_true, p, n_bootstrap=n_bootstrap)
        ax_main.fill_between(x_ci, ci_low, ci_up, color=ds['color'], alpha=0.15, zorder=1)

        sig = '***' if hl_pval < 0.001 else ('**' if hl_pval < 0.01 else ('*' if hl_pval < 0.05 else ''))
        stats_lines.append(
            f"{ds['label']}: Slope={slope:.2f}, Int={intercept:.2f}, Brier={brier:.3f}"
        )
        stats_lines.append(
            f"  H-L: χ²={hl_stat:.2f}, p={'<0.001' if hl_pval < 0.001 else f'{hl_pval:.3f}'}{sig}, E/O={eo:.2f}"
        )

    # 对角线
    ax_main.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--', zorder=2)

    # 美化主图
    ax_main.set_ylabel('Observed proportion')
    ax_main.set_xlim([-0.02, 1.02])
    ax_main.set_ylim([-0.02, 1.02])
    ax_main.set_aspect('equal')
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    sci_legend(ax_main, loc='upper left')

    stats_text = '\n'.join(stats_lines)
    ax_main.text(0.98, 0.04, stats_text, transform=ax_main.transAxes,
                 fontsize=7.5, va='bottom', ha='right', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           alpha=0.88, edgecolor='#cccccc'))

    # Spike histogram
    spike_bins_arr = np.linspace(0, 1, 41)
    bin_w = spike_bins_arr[1] - spike_bins_arr[0]
    for ds_p, col, offset in [(prob_orig, COLOR_ORIG, 0.002), (prob_cal, COLOR_CAL, -0.002)]:
        counts_pos, _ = np.histogram(ds_p[y_true == 1], bins=spike_bins_arr)
        counts_neg, _ = np.histogram(ds_p[y_true == 0], bins=spike_bins_arr)
        centers = (spike_bins_arr[:-1] + spike_bins_arr[1:]) / 2 + offset
        ax_spike.bar(centers, counts_pos, width=bin_w * 0.42, color=col, alpha=0.6, edgecolor='none')
        neg_scale = max(counts_pos.max(), 1) / max(counts_neg.max(), 1) * 0.8
        ax_spike.bar(centers, -counts_neg * neg_scale, width=bin_w * 0.42, color=col, alpha=0.25, edgecolor='none')

    ax_spike.axhline(y=0, color='#888888', linewidth=0.6)
    ax_spike.set_xlabel('Predicted probability')
    ax_spike.set_xlim([-0.02, 1.02])
    ax_spike.spines['top'].set_visible(False)
    ax_spike.spines['right'].set_visible(False)
    ax_spike.set_yticks([])
    ax_spike.text(0.01, 0.92, 'Events', transform=ax_spike.transAxes,
                  fontsize=8.5, color='#555555', va='top', fontstyle='italic')
    ax_spike.text(0.01, 0.08, 'Non-events', transform=ax_spike.transAxes,
                  fontsize=8.5, color='#555555', va='bottom', fontstyle='italic')

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path} / .pdf")
    plt.close()

# 外部验证集 before vs after
plot_calibration_before_after_overlay(
    y_ext, prob_ext, prob_ext_cal, label_set="External",
    save_path=f'figures/Calibration_BeforeAfter_External_{TIMESTAMP}.png',
    n_bootstrap=500
)

# 内部验证集 before vs after
plot_calibration_before_after_overlay(
    y_int, prob_int, prob_int_cal, label_set="Internal (OOF)",
    save_path=f'figures/Calibration_BeforeAfter_Internal_{TIMESTAMP}.png',
    n_bootstrap=500
)

print("\n✅ All calibration plots generated!")

# ============================================================================
# 3. DCA 决策曲线分析（使用 OOF 数据 + 截距校准对比）
# ============================================================================
print("\n" + "="*80)
print("[3/3] 绘制决策曲线分析（原始 + 校准后概率对比）")
print("="*80)

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
    """计算高风险患者数量"""
    return np.sum(y_pred_prob >= threshold) / n_total * 1000  # 每1000人

# 设置阈值范围
threshs = np.arange(0, 1.01, 0.01)

# 获取最优阈值（从模型包中）
optimal_threshold = full_package.get('optimal_threshold', 0.5)

# ============================================================================
# DCA 图1: 原始概率 DCA 
# ============================================================================
print("\n  [DCA 1/3] 原始概率 DCA... ")

# 计算 Treat All 基线 — 使用各自数据集的患病率 (而非混合)
prevalence_int = np.sum(y_int == 1) / len(y_int)
prevalence_ext = np.sum(y_ext == 1) / len(y_ext)
all_nb_int = prevalence_int - (1 - prevalence_int) * (threshs / (1 - threshs + 1e-10))
all_nb_ext = prevalence_ext - (1 - prevalence_ext) * (threshs / (1 - threshs + 1e-10))

# ============================================================================
# DCA 图2: 校准后概率 DCA (内外部对比)
# ============================================================================
print("\n  [DCA 2/3] 校准后概率 DCA...")
fig = plt.figure(figsize=(3.5, 5))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5], hspace=0.3)
ax_main = fig.add_subplot(gs[0])
ax_num = fig.add_subplot(gs[1], sharex=ax_main)

dca_data_cal = {}

for y, p, label, col in [(y_int, prob_int_cal, 'Internal (calibrated)', C_INT),
                          (y_ext, prob_ext_cal, 'External (calibrated)', C_EXT)]:
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

    dca_data_cal[label] = {
        'net_benefit': nbs_mean, 'ci_lower': nbs_lower,
        'ci_upper': nbs_upper, 'n_high_risk': n_high_risk
    }
    print(f"  ✓ {label} 完成")

ax_main.plot(threshs, all_nb_int, color=C_INT, ls='--', lw=1.0,
             alpha=0.45, label=f'Treat all Int ({prevalence_int:.1%})')
ax_main.plot(threshs, all_nb_ext, color=C_EXT, ls='--', lw=1.0,
             alpha=0.45, label=f'Treat all Ext ({prevalence_ext:.1%})')
ax_main.axhline(y=0, color='black', lw=0.8, ls='-.', label='Treat none', alpha=0.7)
ax_main.axvline(clinical_threshold_cal_int, color=C_INT, ls=':', lw=1.0,
                alpha=0.5, label=f'Threshold Int = {clinical_threshold_cal_int:.3f}')
ax_main.axvline(clinical_threshold_cal_ext, color=C_EXT, ls=':', lw=1.0,
                alpha=0.5, label=f'Threshold Ext = {clinical_threshold_cal_ext:.3f}')
ax_main.set_xlim([0, 0.5])
y_max_candidates_cal = [all_nb_int.max(), all_nb_ext.max()]
for k in dca_data_cal:
    y_max_candidates_cal.append(max(dca_data_cal[k]['net_benefit']))
y_max_cal = max(y_max_candidates_cal)
ax_main.set_ylim([-0.05, y_max_cal + 0.05])
ax_main.set_ylabel('Net benefit')
ax_main.tick_params(axis='x', labelbottom=False)          # 主面板隐藏x标签
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
sci_legend(ax_main, loc='upper right', fontsize=5.5,
           handlelength=1.8, handletextpad=0.5,
           labelspacing=0.35, borderpad=0.4)

ax_num.set_ylabel('High risk\nper 1000', fontsize=7)       # 精简为两行, 缩小字号
ax_num.set_xlabel('Threshold probability')
ax_num.set_ylim([0, 1000])
ax_num.spines['top'].set_visible(False)
ax_num.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/DCA_Analysis_Calibrated.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/DCA_Analysis_Calibrated.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/DCA_Analysis_Calibrated.png / .pdf")
plt.close()

# ============================================================================
# DCA 图3a: 内部校准前后对比
# ============================================================================
print("\n  [DCA 3a] 内部校准前后对比...")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

panel_configs_int = [
    (y_int, prob_int,     'Internal - Original',   C_INT, axes[0], optimal_threshold),
    (y_int, prob_int_cal, 'Internal - Calibrated',  C_INT, axes[1], clinical_threshold_cal_int),
]

for y_true_p, y_pred_p, title, color, ax, thresh_used in panel_configs_int:
    threshs_dca = np.arange(0, 1.01, 0.01)

    nbs_model = []
    for t in threshs_dca:
        if t >= 1.0:
            nbs_model.append(0)
            continue
        tp = np.sum((y_pred_p >= t) & (y_true_p == 1))
        fp = np.sum((y_pred_p >= t) & (y_true_p == 0))
        nb = (tp / len(y_true_p)) - (fp / len(y_true_p)) * (t / (1 - t))
        nbs_model.append(nb)

    ax.plot(threshs_dca, nbs_model, color=color, lw=1.5, label='Model')

    prev_local = np.sum(y_true_p == 1) / len(y_true_p)
    all_nb_local = prev_local - (1 - prev_local) * (threshs_dca / (1 - threshs_dca + 1e-10))
    ax.plot(threshs_dca, all_nb_local, color='#999999', ls='--', lw=1.2,
            alpha=0.7, label='Treat All')
    ax.axhline(y=0, color='black', lw=0.8, ls='-.', label='Treat none')

    idx_opt = np.argmin(np.abs(threshs_dca - thresh_used))
    nb_opt = nbs_model[idx_opt]
    ax.plot(thresh_used, nb_opt, 'r*', markersize=12,
            label=f'Threshold ({thresh_used:.3f}): NB={nb_opt:.4f}')

    ax.set_xlim([0, 0.5])
    ax.set_ylim([-0.05, max(max(nbs_model), all_nb_local.max()) + 0.05])
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(f'{title}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/DCA_Internal_BeforeAfter.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/DCA_Internal_BeforeAfter.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/DCA_Internal_BeforeAfter.png / .pdf")
plt.close()

# ============================================================================
# DCA 图3b: 外部校准前后对比
# ============================================================================
print("\n  [DCA 3b] 外部校准前后对比...")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

panel_configs_ext = [
    (y_ext, prob_ext,     'External - Original',   C_EXT, axes[0], optimal_threshold),
    (y_ext, prob_ext_cal, 'External - Calibrated',  C_EXT, axes[1], clinical_threshold_cal_ext),
]

for y_true_p, y_pred_p, title, color, ax, thresh_used in panel_configs_ext:
    threshs_dca = np.arange(0, 1.01, 0.01)

    nbs_model = []
    for t in threshs_dca:
        if t >= 1.0:
            nbs_model.append(0)
            continue
        tp = np.sum((y_pred_p >= t) & (y_true_p == 1))
        fp = np.sum((y_pred_p >= t) & (y_true_p == 0))
        nb = (tp / len(y_true_p)) - (fp / len(y_true_p)) * (t / (1 - t))
        nbs_model.append(nb)

    ax.plot(threshs_dca, nbs_model, color=color, lw=1.5, label='Model')

    prev_local = np.sum(y_true_p == 1) / len(y_true_p)
    all_nb_local = prev_local - (1 - prev_local) * (threshs_dca / (1 - threshs_dca + 1e-10))
    ax.plot(threshs_dca, all_nb_local, color='#999999', ls='--', lw=1.2,
            alpha=0.7, label='Treat All')
    ax.axhline(y=0, color='black', lw=0.8, ls='-.', label='Treat none')

    idx_opt = np.argmin(np.abs(threshs_dca - thresh_used))
    nb_opt = nbs_model[idx_opt]
    ax.plot(thresh_used, nb_opt, 'r*', markersize=12,
            label=f'Threshold ({thresh_used:.3f}): NB={nb_opt:.4f}')

    ax.set_xlim([0, 0.5])
    ax.set_ylim([-0.05, max(max(nbs_model), all_nb_local.max()) + 0.05])
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(f'{title}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/DCA_External_BeforeAfter.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/DCA_External_BeforeAfter.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/DCA_External_BeforeAfter.png / .pdf")
plt.close()

# ============================================================================
# DCA 图3c: 内部+外部校准后对比
# ============================================================================
print("\n  [DCA 3c] 内部+外部校准后对比...")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

panel_configs_cal = [
    (y_int, prob_int_cal, 'Internal - Calibrated', C_INT, axes[0], clinical_threshold_cal_int),
    (y_ext, prob_ext_cal, 'External - Calibrated', C_EXT, axes[1], clinical_threshold_cal_ext),
]

# ---------- 第一轮: 计算所有面板数据 + 收集全局 y 范围 ----------
panel_results = []
global_ymax = -np.inf

for y_true_p, y_pred_p, title, color, ax, thresh_used in panel_configs_cal:
    threshs_dca = np.arange(0, 1.01, 0.01)

    nbs_model = []
    for t in threshs_dca:
        if t >= 1.0:
            nbs_model.append(0)
            continue
        tp = np.sum((y_pred_p >= t) & (y_true_p == 1))
        fp = np.sum((y_pred_p >= t) & (y_true_p == 0))
        nb = (tp / len(y_true_p)) - (fp / len(y_true_p)) * (t / (1 - t))
        nbs_model.append(nb)

    prev_local = np.sum(y_true_p == 1) / len(y_true_p)
    all_nb_local = prev_local - (1 - prev_local) * (threshs_dca / (1 - threshs_dca + 1e-10))

    idx_opt = np.argmin(np.abs(threshs_dca - thresh_used))
    nb_opt = nbs_model[idx_opt]

    local_ymax = max(max(nbs_model), all_nb_local.max())
    global_ymax = max(global_ymax, local_ymax)

    panel_results.append({
        'threshs_dca': threshs_dca, 'nbs_model': nbs_model,
        'all_nb_local': all_nb_local, 'prev_local': prev_local,
        'thresh_used': thresh_used, 'nb_opt': nb_opt, 'idx_opt': idx_opt,
        'title': title, 'color': color, 'ax': ax,
    })

shared_ylim = [-0.05, global_ymax + 0.05]

# ---------- 第二轮: 统一绘图 ----------
for r in panel_results:
    ax = r['ax']

    ax.plot(r['threshs_dca'], r['nbs_model'], color=r['color'], lw=1.5,
            label='Model')
    ax.plot(r['threshs_dca'], r['all_nb_local'], color='#999999', ls='--', lw=1.2,
            alpha=0.7, label='Treat All')
    ax.axhline(y=0, color='black', lw=0.8, ls='-.', label='Treat none')
    ax.plot(r['thresh_used'], r['nb_opt'], 'r*', markersize=12,
            label=f'Threshold ({r["thresh_used"]:.3f}): NB={r["nb_opt"]:.4f}')

    ax.set_xlim([0, 0.5])
    ax.set_ylim(shared_ylim)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(f'{r["title"]}')

    # spine 粗细与 ROC_CI 一致
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.8)

    sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/DCA_Calibrated_IntExt.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/DCA_Calibrated_IntExt.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/DCA_Calibrated_IntExt.png / .pdf")
plt.close()
print(f"\n" + "="*80)
print("✅ 校准曲线和 DCA 曲线绘制完成（含截距校准前后对比）！")
print("="*80 + "\n")



# ==========================================
# 4. Bootstrap 分析（仅外部验证）
# ==========================================
print("\n[3/9] 执行Bootstrap分析（外部验证）...")
def bootstrap_metrics(y_true, y_prob, threshold, n_bootstrap=1000):
    """Bootstrap计算指标的置信区间（含不平衡数据核心指标）"""
    np.random.seed(42)
    n_samples = len(y_true)
    
    metrics = {
        'AUC': [],
        'AUPRC': [],
        'Sensitivity': [],
        'Specificity': [],
        'PPV': [],
        'NPV': [],
        'F1': [],
        'MCC': [],
        'G_mean': [],
        'Balanced_Acc': [],
    }
    
    for _ in range(n_bootstrap):
        # 重采样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y_true[indices]
        p_boot = y_prob[indices]
        
        # 计算 AUC 和 AUPRC
        try:
            from sklearn.metrics import average_precision_score
            auc_boot = roc_auc_score(y_boot, p_boot)
            metrics['AUC'].append(auc_boot)
            metrics['AUPRC'].append(average_precision_score(y_boot, p_boot))
        except:
            pass
        
        # 预测
        y_pred_boot = (p_boot >= threshold).astype(int)
        
        # 混淆矩阵
        cm = confusion_matrix(y_boot, y_pred_boot)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0
            
            metrics['Sensitivity'].append(sens)
            metrics['Specificity'].append(spec)
            metrics['PPV'].append(ppv)
            metrics['NPV'].append(npv)
            metrics['F1'].append(f1)
            metrics['MCC'].append(matthews_corrcoef(y_boot, y_pred_boot))
            metrics['G_mean'].append(np.sqrt(sens * spec))
            metrics['Balanced_Acc'].append((sens + spec) / 2.0)
    
    # 计算置信区间
    summary = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            summary[metric_name] = {
                'Mean': np.mean(values),
                '95%_CI_Lower': np.percentile(values, 2.5),
                '95%_CI_Upper': np.percentile(values, 97.5)
            }
    
    return pd.DataFrame(summary).T
# ====== 重新计算校准后的 Bootstrap CI ======
print("  重新计算校准后 Bootstrap CI (外部验证)...")
df_bootstrap_cal_ext = bootstrap_metrics(
    y_ext, 
    prob_ext_cal,                  # ← 校准后概率
    clinical_threshold_cal_ext,    # ← 校准后阈值
    n_bootstrap=1000
)
print(f"  ✓ 校准后外部验证 Bootstrap CI 计算完成")

# 同理，如需内部验证的 CI
df_bootstrap_cal_int = bootstrap_metrics(
    y_int, 
    prob_int_cal, 
    clinical_threshold_cal_int, 
    n_bootstrap=1000
)
print(f"  ✓ 内部验证 Bootstrap CI 计算完成")

# ==========================================
# 5. 绘制混淆矩阵（内部+外部）
# ==========================================
print("\n[4/9] 绘制混淆矩阵...")

from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_confusion_matrix(y_true, y_prob, threshold, dataset_name, color=None, 
                          save_path=None, cmap='Blues'):
    """绘制混淆矩阵热图"""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比
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
    print(f"  ✓ 已保存: {save_path}")
    plt.close()
    
    return cm

cm_int = plot_confusion_matrix(y_int, prob_int_cal, clinical_threshold_cal_int, 
                                'Internal (Calibrated)', 
                                save_path='figures/Confusion_Matrix_Internal.png',
                                cmap='Blues')

cm_ext = plot_confusion_matrix(y_ext, prob_ext_cal, clinical_threshold_cal_ext, 
                                'External (Calibrated)', 
                                save_path='figures/Confusion_Matrix_External.png',
                                cmap='Oranges')

# ==========================================
# 6. 绘制并排混淆矩阵对比
# ==========================================
print("\n[5/9] 绘制混淆矩阵对比图...")
from matplotlib.colors import LinearSegmentedColormap
# 定义一次，全局复用
cmap_int = LinearSegmentedColormap.from_list('cmap_int', ['#FFFFFF', C_INT])  # 白→深蓝
cmap_ext = LinearSegmentedColormap.from_list('cmap_ext', ['#FFFFFF', C_EXT])  # 白→红

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for idx, (y_true, y_prob, threshold, dataset_name, color, ax) in enumerate([
    (y_int, prob_int_cal, clinical_threshold_cal_int, 'Internal (Calibrated)', C_INT, axes[0]),
    (y_ext, prob_ext_cal, clinical_threshold_cal_ext, 'External (Calibrated)', C_EXT, axes[1])
]):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    # 使用
    cmap = 'Blues' if idx == 0 else 'Oranges'

    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=True,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['Non-recurrence', 'Recurrence'],
                yticklabels=['Non-recurrence', 'Recurrence'],
                annot_kws={"size": 14},
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_title(f'{dataset_name}', 
                 fontsize=15, pad=10)
    ax.set_ylabel('True Label',fontsize=13, labelpad=10)
    ax.set_xlabel('Predicted Label',fontsize=13, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

# plt.suptitle('Confusion Matrix Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('figures/Confusion_Matrix_Comparison.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: figures/Confusion_Matrix_Comparison.png")
plt.close()




fig, axes = plt.subplots(1, 2, figsize=(16, 7)) #稍微增加高度以容纳标题

# 定义标签列表，匹配图片格式
labels = ['Non-recurrence', 'Recurrence']

for idx, (y_true, y_prob, threshold, dataset_name, color, ax) in enumerate([
    (y_int, prob_int_cal, clinical_threshold_cal_int, 'Internal (Calibrated)', C_INT, axes[0]),
    (y_ext, prob_ext_cal, clinical_threshold_cal_ext, 'External (Calibrated)', C_EXT, axes[1])
]):
    # 1. 预测与混淆矩阵计算
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算各项指标 (用于标题)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    mcc_val = matthews_corrcoef(y_true, y_pred)
    g_mean_val = np.sqrt(sensitivity * specificity)
    
    # 2. 准备注释文本 (数值 + 总百分比)
    # 图片显示的是占总样本的百分比 (例如 338/701 = 48.2%)
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # 3. 确定配色 (Internal用蓝, External用橙)
    cmap = cmap_int if idx == 0 else cmap_ext
    
    # 4. 绘图
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=True,
                square=True, linewidths=2, linecolor='white',
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={"size": 14}, # 增大字体
                ax=ax, cbar_kws={'label': 'Count'})
    
    # 5. 设置复杂标题 (匹配图片格式)
    # title_str = (f"{dataset_name}\n"
    #              f"Threshold: {clinical_threshold:.3f} | Accuracy: {accuracy:.3f}\n"
    #              f"Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}\n"
    #              f"PPV: {ppv:.3f} | NPV: {npv:.3f}\n"
    #              f"MCC: {mcc_val:.3f} | G-mean: {g_mean_val:.3f}")
    
    title_str = (f"{dataset_name}")
    
    ax.set_title(title_str, pad=10)
    
    # 6. 轴标签设置
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('figures/Confusion_Matrix_Comparison2.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: figures/Confusion_Matrix_Comparison2.png")
plt.close()

# ==========================================
# 7. 混淆矩阵指标详细对比表
# ==========================================
print("\n[6/9] 生成混淆矩阵指标详细对比表...")

metrics_comparison = []

for dataset_name, y_true, y_p, thresh in [
    ('Internal (Calibrated)', y_int, prob_int_cal, clinical_threshold_cal_int), 
    ('External (Calibrated)', y_ext, prob_ext_cal, clinical_threshold_cal_ext)
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
    
    # 从 Bootstrap 获取置信区间（仅外部验证）
    if dataset_name == 'External (Calibrated)':
        sens_ci = f"{df_bootstrap_cal_ext.loc['Sensitivity', '95%_CI_Lower']:.4f}-{df_bootstrap_cal_ext.loc['Sensitivity', '95%_CI_Upper']:.4f}"
        spec_ci = f"{df_bootstrap_cal_ext.loc['Specificity', '95%_CI_Lower']:.4f}-{df_bootstrap_cal_ext.loc['Specificity', '95%_CI_Upper']:.4f}"
        ppv_ci  = f"{df_bootstrap_cal_ext.loc['PPV', '95%_CI_Lower']:.4f}-{df_bootstrap_cal_ext.loc['PPV', '95%_CI_Upper']:.4f}"
        npv_ci  = f"{df_bootstrap_cal_ext.loc['NPV', '95%_CI_Lower']:.4f}-{df_bootstrap_cal_ext.loc['NPV', '95%_CI_Upper']:.4f}"
        f1_ci   = f"{df_bootstrap_cal_ext.loc['F1', '95%_CI_Lower']:.4f}-{df_bootstrap_cal_ext.loc['F1', '95%_CI_Upper']:.4f}" if 'F1' in df_bootstrap_cal_ext.index else 'N/A'
        mcc_ci  = f"{df_bootstrap_cal_ext.loc['MCC', '95%_CI_Lower']:.4f}-{df_bootstrap_cal_ext.loc['MCC', '95%_CI_Upper']:.4f}" if 'MCC' in df_bootstrap_cal_ext.index else 'N/A'
        gmean_ci  = f"{df_bootstrap_cal_ext.loc['G_mean', '95%_CI_Lower']:.4f}-{df_bootstrap_cal_ext.loc['G_mean', '95%_CI_Upper']:.4f}" if 'G_mean' in df_bootstrap_cal_ext.index else 'N/A'
        balacc_ci = f"{df_bootstrap_cal_ext.loc['Balanced_Acc', '95%_CI_Lower']:.4f}-{df_bootstrap_cal_ext.loc['Balanced_Acc', '95%_CI_Upper']:.4f}" if 'Balanced_Acc' in df_bootstrap_cal_ext.index else 'N/A'
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
print(f"  ✓ 已保存: figures/Confusion_Matrix_Metrics_Comparison.csv")

# 打印到控制台
print("\n" + "="*80)
print("混淆矩阵指标详细对比")
print("="*80)
print(df_metrics_comparison.to_string(index=False))
print("="*80 + "\n")

# ==========================================
# 8. 混淆矩阵指标可视化对比（雷达图）
# ==========================================
print("[7/9] 绘制混淆矩阵指标雷达图对比...")
from math import pi
# 准备雷达图数据
categories = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score', 'MCC', 'G-mean', 'Bal.Acc']
N = len(categories)

# 计算每个数据集的指标
values_int = []
values_ext = []

for y_true, y_p, thresh, values_list in [
    (y_int, prob_int_cal, clinical_threshold_cal_int, values_int), 
    (y_ext, prob_ext_cal, clinical_threshold_cal_ext, values_ext)
]:
    y_pred = (y_p >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    g_mean = np.sqrt(sens * spec)
    balanced_acc = (sens + spec) / 2.0
    
    # 【注意】MCC范围是[-1,1]，归一化到[0,1]供雷达图显示
    mcc_normalized = (mcc + 1) / 2.0
    
    values_list.extend([sens, spec, ppv, npv, f1, mcc_normalized, g_mean, balanced_acc])

# 闭合雷达图
values_int += values_int[:1]
values_ext += values_ext[:1]

# 计算角度
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# 绘制雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

ax.plot(angles, values_int, 'o-', linewidth=2.5, label='Internal', 
        color=COLOR_INTERNAL, markersize=8)
ax.fill(angles, values_int, alpha=0.15, color=COLOR_INTERNAL)

ax.plot(angles, values_ext, 's-', linewidth=2.5, label='External', 
        color=COLOR_EXTERNAL, markersize=8)
ax.fill(angles, values_ext, alpha=0.15, color=COLOR_EXTERNAL)

# 设置刻度和标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(True, linestyle='--', alpha=0.6)

plt.title(f'Performance Comparison (Calibrated)', fontsize=14, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('figures/Confusion_Matrix_Radar_Comparison.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: figures/Confusion_Matrix_Radar_Comparison.png")
plt.close()

# ==========================================
# 9. 绘制指标柱状图对比
# ==========================================
print("\n[8/9] 绘制指标柱状图对比...")

# 准备数据
metrics_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score', 'MCC', 'G-mean', 'Bal.Acc']
internal_values = []
external_values = []

for y_true, y_p, thresh, values_list in [
    (y_int, prob_int_cal, clinical_threshold_cal_int, internal_values), 
    (y_ext, prob_ext_cal, clinical_threshold_cal_ext, external_values)
]:
    y_pred = (y_p >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    g_mean = np.sqrt(sens * spec)
    balanced_acc = (sens + spec) / 2.0
    
    values_list.extend([sens, spec, ppv, npv, f1, mcc, g_mean, balanced_acc])

# 绘制柱状图
x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, internal_values, width, label='Internal', 
               color=COLOR_INTERNAL, alpha=0.8)
bars2 = ax.bar(x + width/2, external_values, width, label='External', 
               color=COLOR_EXTERNAL, alpha=0.8)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title(f'Performance Comparison (Calibrated)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, rotation=0)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/Confusion_Matrix_Metrics_Bar_Comparison.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: figures/Confusion_Matrix_Metrics_Bar_Comparison.png")
plt.close()

# ==========================================
# 10. 总结
# ==========================================
print(f"\n{'='*60}")
print("✅ 混淆矩阵分析完成！")
print(f"{'='*60}")
print("\n生成的文件:")
print("  📊 figures/Confusion_Matrix_Internal.png")
print("  📊 figures/Confusion_Matrix_External.png")
print("  📊 figures/Confusion_Matrix_Comparison.png")
print("  📊 figures/Confusion_Matrix_Radar_Comparison.png")
print("  📊 figures/Confusion_Matrix_Metrics_Bar_Comparison.png")
print("  📄 figures/Confusion_Matrix_Metrics_Comparison.csv")
print(f"\n{'='*60}\n")


# ==========================================
# 4. 多模型 ROC 对比与 DeLong 检验
# ==========================================
print(f"{'='*60}")
print("📊 开始多模型性能对比分析...")
print(f"{'='*60}\n")
# ------------------------------------------
# 4.1 DeLong 检验函数定义
# ------------------------------------------
def delong_roc_test(y_true, prob1, prob2):

    from scipy import stats
    
    # 确保是 numpy 数组
    y_true = np.asarray(y_true)
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)
    
    def compute_midrank(x):
        """计算中秩"""
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            for k in range(i, j):
                T[k] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T + 1
        return T2
    
    def compute_ground_truth_statistics(y_true):
        """计算真实标签的统计信息"""
        assert np.array_equal(np.unique(y_true), [0, 1])
        order = (-y_true).argsort()
        label_1_count = int(y_true.sum())
        return order, label_1_count
    
    def fast_delong(predictions_sorted_transposed, label_1_count):
        """快速 DeLong 算法"""
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        aucs = np.zeros(k)
        score = np.zeros(k)
        for j in range(k):
            midrank = compute_midrank(predictions_sorted_transposed[j, :])
            aucs[j] = (midrank[:m].sum() - m * (m + 1) / 2) / (m * n)
            score[j] = aucs[j]

        # 计算协方差
        v01 = np.zeros((k, n))
        v10 = np.zeros((k, m))
        for j in range(k):
            midrank = compute_midrank(predictions_sorted_transposed[j, :])
            v10[j, :] = (midrank[:m] - np.arange(1, m + 1)) / n
            v01[j, :] = 1 - (midrank[m:] - np.arange(m + 1, m + n + 1)) / m

        sx = np.cov(v10) if k > 1 else np.var(v10) / m
        sy = np.cov(v01) if k > 1 else np.var(v01) / n
        delongcov = sx / m + sy / n

        return aucs, delongcov
    
    order, label_1_count = compute_ground_truth_statistics(y_true)
    predictions_sorted_transposed = np.vstack([prob1, prob2])[:, order]
    
    aucs, delongcov = fast_delong(predictions_sorted_transposed, label_1_count)
    
    auc1, auc2 = aucs[0], aucs[1]
    
    # 计算 z-统计量和 p 值
    if isinstance(delongcov, np.ndarray):
        var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    else:
        var = 2 * delongcov
    
    z = (auc1 - auc2) / np.sqrt(var + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return p_value, z, auc1, auc2
# ------------------------------------------
# 4.2 加载 DeLong 数据包
# ------------------------------------------
print("[1/6] 加载 DeLong 数据包...")
# ========== 配置数据包路径 ==========
DATA_full_package_PATH = f"{DATA_PATH}/DeLong_Data_Package_latest.pkl"  

if not os.path.exists(DATA_full_package_PATH):
    raise FileNotFoundError(f"❌ 未找到数据包: {DATA_full_package_PATH}")

# 加载数据包
delong_data = joblib.load(DATA_full_package_PATH)

print(f"  ✓ 数据包加载成功: {DATA_full_package_PATH}")
print(f"  ✓ 时间戳: {delong_data.get('timestamp', 'N/A')}")

# ------------------------------------------
# 4.3 解析数据包内容
# ------------------------------------------
print(f"\n[2/6] 解析数据包内容...")

# 提取真实标签
y_true_int = np.asarray(delong_data['y_internal'])
y_true_ext = np.asarray(delong_data['y_external'])

# 获取模型列表
MODELS = delong_data['model_names']
print(f"  ✓ 检测到 {len(MODELS)} 个模型: {', '.join(MODELS)}")
print(f"  ✓ 内部验证样本数: {len(y_true_int)}")
print(f"  ✓ 外部验证样本数: {len(y_true_ext)}")

# 准备颜色列表
# COLORS_LIST = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
MODEL_COLORS = {model: COLORS_LIST[i] for i, model in enumerate(MODELS)}

# 存储所有模型的预测结果
model_preds_int = {}  # 内部验证预测
model_preds_ext = {}  # 外部验证预测
auc_scores_int = {}   # 内部验证 AUC
auc_scores_ext = {}   # 外部验证 AUC
ap_scores_int = {}    # 内部验证 AUPRC
ap_scores_ext = {}    # 外部验证 AUPRC

# 解析每个模型的预测数据
from sklearn.metrics import average_precision_score
print(f"\n  各模型 AUC / AUPRC 分数:")
print(f"  {'-'*80}")
print(f"  {'模型名称':20s} | {'Int AUC':10s} | {'Ext AUC':10s} | {'Int AUPRC':10s} | {'Ext AUPRC':10s}")
print(f"  {'-'*80}")

for model_name in MODELS:
    pred_data = delong_data['model_predictions'].get(model_name, None)
    
    if pred_data is None:
        print(f"  ⚠️ 未找到 {model_name} 的预测数据")
        continue
    
    # 内部验证预测
    if 'internal_probs' in pred_data:
        model_preds_int[model_name] = np.asarray(pred_data['internal_probs'])
        auc_scores_int[model_name] = roc_auc_score(y_true_int, model_preds_int[model_name])
        ap_scores_int[model_name] = average_precision_score(y_true_int, model_preds_int[model_name])
    
    # 外部验证预测
    if 'external_probs' in pred_data:
        model_preds_ext[model_name] = np.asarray(pred_data['external_probs'])
        auc_scores_ext[model_name] = roc_auc_score(y_true_ext, model_preds_ext[model_name])
        ap_scores_ext[model_name] = average_precision_score(y_true_ext, model_preds_ext[model_name])
    
    # 打印 AUC + AUPRC
    int_auc = auc_scores_int.get(model_name, None)
    ext_auc = auc_scores_ext.get(model_name, None)
    int_ap = ap_scores_int.get(model_name, None)
    ext_ap = ap_scores_ext.get(model_name, None)
    int_str = f"{int_auc:.4f}" if int_auc else "N/A"
    ext_str = f"{ext_auc:.4f}" if ext_auc else "N/A"
    int_ap_str = f"{int_ap:.4f}" if int_ap else "N/A"
    ext_ap_str = f"{ext_ap:.4f}" if ext_ap else "N/A"
    print(f"  {model_name:20s} | {int_str:10s} | {ext_str:10s} | {int_ap_str:10s} | {ext_ap_str:10s}")

print(f"  {'-'*55}")

# ------------------------------------------
# 4.4 执行 DeLong 检验（内部验证）
# ------------------------------------------
print(f"\n[3/6] 执行 DeLong 检验 (内部验证)...")

print(f"\n  DeLong 检验结果 (vs {best_model}):")
print(f"  {'-'*65}")
print(f"  {'模型':20s} | {'AUC':8s} | {'P-value':10s} | {'Z-stat':8s} | Sig")
print(f"  {'-'*65}")

delong_results_int = []
for model_name in MODELS:
    if model_name != best_model_name and model_name in model_preds_int:  # ← 修改1
        p_val, z_stat, auc1, auc2 = delong_roc_test(
            y_true_int, 
            model_preds_int[best_model_name],  
            model_preds_int[model_name]
        )
        
        sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'NS'
        
        delong_results_int.append({
            'Comparison': f"{best_model_name} vs {model_name}", 
            'Model': model_name,
            'AUC': auc2,
            'Best_Model': best_model_name,  
            'Best_Model_AUC': auc1,
            'AUC_Diff': auc1 - auc2,
            'Z_Statistic': z_stat,
            'P_Value': p_val,
            'Significant': sig_marker
        })
        
        print(f"  {model_name:20s} | {auc2:.4f}  | {p_val:.6f}  | {z_stat:7.4f}  | {sig_marker}")

print(f"  {'-'*65}")
print(f"  注: *** p<0.001, ** p<0.01, * p<0.05, NS: not significant")

# ------------------------------------------
# 4.5 执行 DeLong 检验（外部验证）
# ------------------------------------------
print(f"\n[4/6] 执行 DeLong 检验 (外部验证)...")

print(f"\n  DeLong 检验结果 (vs {best_model}):")
print(f"  {'-'*65}")
print(f"  {'模型':20s} | {'AUC':8s} | {'P-value':10s} | {'Z-stat':8s} | Sig")
print(f"  {'-'*65}")

delong_results_ext = []
for model_name in MODELS:
    if model_name != best_model_name and model_name in model_preds_ext:
        p_val, z_stat, auc1, auc2 = delong_roc_test(
            y_true_ext, 
            model_preds_ext[best_model_name], 
            model_preds_ext[model_name]
        )
        
        sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'NS'
        
        delong_results_ext.append({
            'Comparison': f"{best_model_name} vs {model_name}",
            'Model': model_name,
            'AUC': auc2,
            'Best_Model': best_model_name,
            'Best_Model_AUC': auc1,
            'AUC_Diff': auc1 - auc2,
            'Z_Statistic': z_stat,
            'P_Value': p_val,
            'Significant': sig_marker
        })
        
        print(f"  {model_name:20s} | {auc2:.4f}  | {p_val:.6f}  | {z_stat:7.4f}  | {sig_marker}")

print(f"  {'-'*65}")

# ------------------------------------------
# 4.6 保存结果
# ------------------------------------------
print(f"\n[5/6] 保存检验结果...")

os.makedirs('figures', exist_ok=True)

# 内部验证 DeLong 结果
df_delong_int = pd.DataFrame(delong_results_int)
df_delong_int = df_delong_int.sort_values('AUC', ascending=False)
df_delong_int.to_csv('figures/DeLong_Test_Results_Internal.csv', index=False)
print(f"  ✓ 已保存: figures/DeLong_Test_Results_Internal.csv")

# 外部验证 DeLong 结果
df_delong_ext = pd.DataFrame(delong_results_ext)
df_delong_ext = df_delong_ext.sort_values('AUC', ascending=False)
df_delong_ext.to_csv('figures/DeLong_Test_Results_External.csv', index=False)
print(f"  ✓ 已保存: figures/DeLong_Test_Results_External.csv")

# AUC 汇总表
summary_data = []
for model_name in MODELS:
    summary_data.append({
        'Model': model_name,
        'AUC_Internal': auc_scores_int.get(model_name, np.nan),
        'AUC_External': auc_scores_ext.get(model_name, np.nan),
        'AUC_Gap': auc_scores_int.get(model_name, 0) - auc_scores_ext.get(model_name, 0)
    })

df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values('AUC_Internal', ascending=False)
df_summary.to_csv('figures/Model_AUC_Summary.csv', index=False)
print(f"  ✓ 已保存: figures/Model_AUC_Summary.csv")

# ------------------------------------------
# 4.7 打印汇总信息
# ------------------------------------------
print(f"\n[6/6] 分析完成汇总")
print(f"{'='*60}")
print(f"📊 模型性能排名 (按内部验证 AUC):")
print(f"{'-'*60}")
for i, row in df_summary.iterrows():
    gap_indicator = "↓" if row['AUC_Gap'] > 0.05 else "→"
    print(f"  {row['Model']:20s} | Int: {row['AUC_Internal']:.4f} | Ext: {row['AUC_External']:.4f} | Gap: {row['AUC_Gap']:+.4f} {gap_indicator}")
print(f"{'-'*60}")
print(f"✅ DeLong 检验完成!")
print(f"{'='*60}")

# ------------------------------------------
# 4.7 绘制多模型 ROC 对比图 
# ------------------------------------------
print(f"\n[7/6] 绘制多模型 ROC 对比图...")
print(f"  ℹ️  内部验证使用 OOF 预测数据（Out-of-Fold Cross-Validation）")
print(f"  ℹ️  外部验证使用独立外部验证集数据")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ===== 内部验证 ROC (基于 OOF 数据) =====

for i, model_name in enumerate(MODELS):
    if model_name in model_preds_int:
        fpr, tpr, _ = roc_curve(y_true_int, model_preds_int[model_name])
        auc_score = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, color=MODEL_COLORS[model_name], lw=2, alpha=0.7,
                    label=f'{model_name} (AUC = {auc_score:.4f})')

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
axes[0].set_xlabel('1 - Specificity (FPR)')
axes[0].set_ylabel('Sensitivity (TPR)')
axes[0].set_title(f'Internal Validation ROC Comparison (OOF)', 
                 fontsize=13, pad=15)
axes[0].legend(loc='lower right', fontsize=9, frameon=True, edgecolor='#999999', fancybox=False)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].set_xlim([-0.02, 1.02])
axes[0].set_ylim([-0.02, 1.02])

# ===== 外部验证 ROC =====
for i, model_name in enumerate(MODELS):
    if model_name in model_preds_ext:
        fpr, tpr, _ = roc_curve(y_true_ext, model_preds_ext[model_name])
        auc_score = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color=MODEL_COLORS[model_name], lw=2, alpha=0.7,
                    label=f'{model_name} (AUC = {auc_score:.4f})')

axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
axes[1].set_xlabel('1 - Specificity (FPR)')
axes[1].set_ylabel('Sensitivity (TPR)')
axes[1].set_title(f'External Validation ROC Comparison', 
                 fontsize=13, pad=15)
axes[1].legend(loc='lower right', fontsize=9, frameon=True, edgecolor='#999999', fancybox=False)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].set_xlim([-0.02, 1.02])
axes[1].set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('figures/ROC_MultiModel_Comparison_OOF.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: figures/ROC_MultiModel_Comparison_OOF.png")
plt.close()

# ------------------------------------------
# 4.8 绘制 AUC 对比柱状图（带 DeLong 显著性标记）
# ------------------------------------------
print(f"\n[8/6] 绘制 AUC 对比柱状图...")


fig, axes = plt.subplots(1, 2, figsize=(16, 8)) 

models_sorted_int = sorted(MODELS, key=lambda x: auc_scores_int.get(x, 0), reverse=True)
models_sorted_ext = sorted(MODELS, key=lambda x: auc_scores_ext.get(x, 0), reverse=True)

# ===== 内部验证柱状图 (基于 OOF 数据) =====
x_int = np.arange(len(models_sorted_int))
aucs_int = [auc_scores_int.get(m, 0) for m in models_sorted_int]
colors_int = [MODEL_COLORS[m] for m in models_sorted_int]

bars_int = axes[0].bar(x_int, aucs_int, color=colors_int, edgecolor='none', linewidth=1.5, alpha=0.8)

# 添加数值标签和显著性标记
for i, (model_name, auc_val) in enumerate(zip(models_sorted_int, aucs_int)):
    # 1. 绘制 AUC 数值
    axes[0].text(i, auc_val + 0.01, f'{auc_val:.3f}', 
                 ha='center', va='bottom', fontsize=9)
    
    # 2. 查找 DeLong 检验结果（基于 OOF 数据）
    model_result = [r for r in delong_results_int if r['Model'] == model_name]
    
    if model_result:
        sig = model_result[0]['Significant']
        # 根据显著性绘制不同颜色
        if sig == 'NS':
            axes[0].text(i, auc_val + 0.04, 'ns', 
                        ha='center', va='bottom', color='black', fontsize=10)
        else:
            axes[0].text(i, auc_val + 0.04, sig, 
                        ha='center', va='bottom', color='red')
        

    elif model_name == best_model_name: 
        axes[0].text(i, auc_val + 0.04, '(Ref)', 
                    ha='center', va='bottom', color='blue', fontsize=10)

axes[0].set_xticks(x_int)
axes[0].set_xticklabels(models_sorted_int, rotation=45, ha='right', fontsize=10)
axes[0].set_ylabel('AUC')


axes[0].set_title(f'Internal Validation AUC Comparison (OOF)', 
                 fontsize=13, pad=15)

axes[0].set_ylim([0.5, 1.15]) 
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].grid(axis='y', linestyle='--', alpha=0.3)

# ===== 外部验证柱状图 =====
x_ext = np.arange(len(models_sorted_ext))
aucs_ext = [auc_scores_ext.get(m, 0) for m in models_sorted_ext]
colors_ext = [MODEL_COLORS[m] for m in models_sorted_ext]

bars_ext = axes[1].bar(x_ext, aucs_ext, color=colors_ext, edgecolor='none', linewidth=1.5, alpha=0.8)

# 添加数值标签和显著性标记
for i, (model_name, auc_val) in enumerate(zip(models_sorted_ext, aucs_ext)):
    axes[1].text(i, auc_val + 0.01, f'{auc_val:.3f}', 
                 ha='center', va='bottom', fontsize=9)
    
    model_result = [r for r in delong_results_ext if r['Model'] == model_name]
    
    if model_result:
        sig = model_result[0]['Significant']
        if sig == 'NS':
            axes[1].text(i, auc_val + 0.04, 'ns', 
                        ha='center', va='bottom', color='black', fontsize=10)
        else:
            axes[1].text(i, auc_val + 0.04, sig, 
                        ha='center', va='bottom', color='red')
        

    elif model_name == best_model_name:
        axes[1].text(i, auc_val + 0.04, '(Ref)', 
                    ha='center', va='bottom', color='blue', fontsize=10)

axes[1].set_xticks(x_ext)
axes[1].set_xticklabels(models_sorted_ext, rotation=45, ha='right', fontsize=10)
axes[1].set_ylabel('AUC')

axes[1].set_title(f'External Validation AUC Comparison', 
                 fontsize=13, pad=15)

axes[1].set_ylim([0.5, 1.00])
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].grid(axis='y', linestyle='--', alpha=0.3)

# 图例说明
fig.text(0.5, 0.02, '*** p<0.001, ** p<0.01, * p<0.05, ns: not significant (DeLong test)', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('figures/AUC_BarChart_with_DeLong_OOF.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: figures/AUC_BarChart_with_DeLong_OOF.png")
plt.close()


# ------------------------------------------
# 4.9 绘制内外部验证 AUC 对比散点图（泛化能力分析）
# ------------------------------------------
print(f"\n[9/6] 绘制泛化能力分析图...")
print("[泛化能力分析图] 绘制内部 vs 外部 AUC 散点图...")

fig, ax = plt.subplots(figsize=(10, 10))

# ✅ 动态计算坐标轴范围（基于实际数据）
all_internal = [auc_scores_int[m] for m in MODELS if m in auc_scores_int]
all_external = [auc_scores_ext[m] for m in MODELS if m in auc_scores_ext]

# 找到最小最大值，留出一点边距
min_auc = min(min(all_internal), min(all_external)) - 0.02
max_auc = max(max(all_internal), max(all_external)) + 0.02
min_auc = max(0.5, min_auc)  # 下限不低于0.5
max_auc = min(1.0, max_auc)  # 上限不超过1.0

print(f"  📊 AUC范围: [{min_auc:.3f}, {max_auc:.3f}]")

# 绘制对角线（完美泛化线）
ax.plot([min_auc, max_auc], [min_auc, max_auc], 'k--', alpha=0.5, lw=2,
        label='Perfect Generalization', zorder=2)

# ✅ 添加泛化容忍区间（±0.02, ±0.05）
tolerance_005 = 0.02
tolerance_010 = 0.05

ax.fill_between([min_auc, max_auc],
                [min_auc - tolerance_005, max_auc - tolerance_005],
                [min_auc + tolerance_005, max_auc + tolerance_005],
                color='green', alpha=0.12, label='Excellent (±2%)', zorder=1)

ax.fill_between([min_auc, max_auc],
                [min_auc - tolerance_010, max_auc - tolerance_010],
                [min_auc - tolerance_005, max_auc - tolerance_005],
                color='yellow', alpha=0.12, label='Good (2%-5%)', zorder=1)

ax.fill_between([min_auc, max_auc],
                [min_auc + tolerance_005, max_auc + tolerance_005],
                [min_auc + tolerance_010, max_auc + tolerance_010],
                color='yellow', alpha=0.12, zorder=1)

# ✅ 绘制各模型点（用 label 替代文字标注，避免重叠）
scatter_data = []
for model_name in MODELS:
    if model_name in auc_scores_int and model_name in auc_scores_ext:
        int_auc = auc_scores_int[model_name]
        ext_auc = auc_scores_ext[model_name]
        is_best = (model_name == best_model_name)

        if is_best:
            size, marker, edgewidth, zorder, alpha = 400, '*', 2.5, 10, 1.0
        else:
            size, marker, edgewidth, zorder, alpha = 150, 'o', 1.5, 5, 0.85

        # ✅ label 中包含模型名和 AUC 数值
        lbl = f"{model_name} (Int:{int_auc:.3f} / Ext:{ext_auc:.3f})"
        ax.scatter(int_auc, ext_auc, c=[MODEL_COLORS[model_name]],
                   s=size, marker=marker, edgecolors='black', linewidth=edgewidth,
                   alpha=alpha, zorder=zorder, label=lbl)

        scatter_data.append((model_name, int_auc, ext_auc, is_best))

# 添加统计信息框（右上角）
gap = abs(auc_scores_int[best_model_name] - auc_scores_ext[best_model_name])
stats_text = (
    f"Best Model: {best_model_name}\n"
    f"Internal AUC: {auc_scores_int[best_model_name]:.4f}\n"
    f"External AUC: {auc_scores_ext[best_model_name]:.4f}\n"
    f"Gap: {gap:.4f}"
)

ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  alpha=0.9, edgecolor='none', linewidth=1.5))

# 图形美化
ax.set_xlabel('Internal Validation AUC (CV)')
ax.set_ylabel('External Validation AUC')
ax.set_title('Generalization Analysis',
             fontsize=14, pad=15)

# ✅ 使用动态范围
ax.set_xlim([min_auc - 0.005, max_auc + 0.005])
ax.set_ylim([min_auc - 0.005, max_auc + 0.005])

ax.grid(alpha=0.15, linewidth=0.5)

# ✅ 统一图例：包含对角线、区域和所有模型点
ax.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.95,
          edgecolor='none', #title='Models & Generalization Quality',
          title_fontsize=9, ncol=1, markerscale=0.8)

ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figures/Generalization_Analysis.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: figures/Generalization_Analysis.png")
plt.close()

# ------------------------------------------
# 4.11 生成综合性能汇总表
# ------------------------------------------
print(f"\n[11/6] 生成综合性能汇总表...")

summary_data = []
for model_name in MODELS:
    row = {'Model': model_name}
    
    # AUC 分数
    row['Internal_AUC'] = auc_scores_int.get(model_name, np.nan)
    row['External_AUC'] = auc_scores_ext.get(model_name, np.nan)
    row['AUC_Gap'] = row['Internal_AUC'] - row['External_AUC'] if not np.isnan(row['Internal_AUC']) else np.nan
    
    # 内部验证 DeLong 结果
    model_delong_int = [r for r in delong_results_int if r['Model'] == model_name]
    if model_delong_int:
        row['P_Value_Int'] = model_delong_int[0]['P_Value']
        row['Sig_Int'] = model_delong_int[0]['Significant']
    elif model_name == best_model:
        row['P_Value_Int'] = np.nan
        row['Sig_Int'] = 'Ref'
    else:
        row['P_Value_Int'] = np.nan
        row['Sig_Int'] = 'N/A'
    
    # 外部验证 DeLong 结果
    model_delong_ext = [r for r in delong_results_ext if r['Model'] == model_name]
    if model_delong_ext:
        row['P_Value_Ext'] = model_delong_ext[0]['P_Value']
        row['Sig_Ext'] = model_delong_ext[0]['Significant']
    elif model_name == best_model:
        row['P_Value_Ext'] = np.nan
        row['Sig_Ext'] = 'Ref'
    else:
        row['P_Value_Ext'] = np.nan
        row['Sig_Ext'] = 'N/A'
    
    # 泛化评估
    if not np.isnan(row['AUC_Gap']):
        if abs(row['AUC_Gap']) < 0.03:
            row['Generalization'] = 'Excellent'
        elif abs(row['AUC_Gap']) < 0.05:
            row['Generalization'] = 'Good'
        elif abs(row['AUC_Gap']) < 0.10:
            row['Generalization'] = 'Moderate'
        else:
            row['Generalization'] = 'Poor'
    else:
        row['Generalization'] = 'N/A'
    
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values('Internal_AUC', ascending=False)

# 格式化输出
df_summary_display = df_summary.copy()
df_summary_display['Internal_AUC'] = df_summary_display['Internal_AUC'].apply(lambda x: f'{x:.4f}' if not np.isnan(x) else 'N/A')
df_summary_display['External_AUC'] = df_summary_display['External_AUC'].apply(lambda x: f'{x:.4f}' if not np.isnan(x) else 'N/A')
df_summary_display['AUC_Gap'] = df_summary_display['AUC_Gap'].apply(lambda x: f'{x:+.4f}' if not np.isnan(x) else 'N/A')
df_summary_display['P_Value_Int'] = df_summary_display['P_Value_Int'].apply(lambda x: f'{x:.4f}' if not np.isnan(x) else '-')
df_summary_display['P_Value_Ext'] = df_summary_display['P_Value_Ext'].apply(lambda x: f'{x:.4f}' if not np.isnan(x) else '-')

df_summary_display.to_csv('figures/Model_Performance_Summary.csv', index=False)
print(f"  ✓ 已保存: figures/Model_Performance_Summary.csv")

# 打印到控制台
print("\n" + "="*100)
print("📊 模型性能综合汇总")
print("="*100)
print(df_summary_display.to_string(index=False))
print("="*100)
print("="*100 + "\n")

print(f"{'='*60}")
print("✅ 多模型性能对比分析完成！")
print(f"{'='*60}")

# ============================================================================
# SHAP 分析 
# ============================================================================
print(f"{'='*60}")
print("🔍 开始 SHAP 分析...")
print(f"{'='*60}\n")

# ============================================================================
# 🎨 SHAP 专用样式
# ============================================================================
# 保存当前 rcParams 以便 SHAP 结束后恢复
_saved_rcParams = mpl.rcParams.copy()

# SHAP 密集图专用较小字号
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
mpl.rcParams.update(SHAP_STYLE)

# ---------- 统一配色 ----------
# 主色板 (来自 Nature / Lancet 风格)
C_BLUE    = '#2166AC'
C_RED     = '#B2182B'
C_GREEN   = '#1B7837'
C_ORANGE  = '#E08214'
C_GRAY    = '#636363'
C_LIGHT   = '#D9D9D9'

# 正/负 SHAP 专用色
C_POS     = '#D6604D'   # 正向 (风险↑)
C_NEG     = '#4393C3'   # 负向 (保护↓)

DPI = 600
os.makedirs('figures/SHAP_Dependence', exist_ok=True)


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


# ------------------------------------------
# 步骤 0: 加载SHAP专用数据包
# ------------------------------------------
print("[0/9] 加载SHAP分析数据包...")

import glob
shap_files = glob.glob(f"{DATA_PATH}/SHAP_Analysis_Data_{TIMESTAMP}.pkl")
if not shap_files:
    raise FileNotFoundError("❌ 未找到SHAP数据包！请先运行建模代码。")

latest_shap_file = max(shap_files, key=os.path.getctime)
print(f"  📦 加载: {os.path.basename(latest_shap_file)}")

shap_package = joblib.load(latest_shap_file)

X_train_transformed = shap_package['X_train_transformed']
X_ext_transformed = shap_package['X_ext_transformed']
feature_names = shap_package['feature_names']
# ✅ 映射为 SCI 展示标签
feature_names = rename_feature_list(feature_names)
classifier = shap_package['best_model_step']
optimal_threshold = shap_package['optimal_threshold']
assert shap_package['optimal_threshold'] == optimal_threshold, \
    f"阈值不一致! shap={shap_package['optimal_threshold']}, main={optimal_threshold}"

print(f"  ✓ 训练集: {X_train_transformed.shape}")
print(f"  ✓ 外部验证集: {X_ext_transformed.shape}")
print(f"  ✓ 特征数量: {len(feature_names)}")
print(f"  ✓ 分类器类型: {type(classifier).__name__}")

y_train = full_package['y_train']
y_ext = full_package['y_external']

# ------------------------------------------
# 步骤 1: 准备 SHAP 分析数据
# ------------------------------------------
print("\n[1/9] 准备 SHAP 分析数据...")

X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
X_ext_df = pd.DataFrame(X_ext_transformed, columns=feature_names)

print(f"  ✓ 训练集特征矩阵: {X_train_df.shape}")
print(f"  ✓ 外部验证集特征矩阵: {X_ext_df.shape}")
print(f"  ✓ 特征数量: {len(feature_names)}")

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
print(f"  ✓ 实际用于SHAP的模型: {model_type}")

if model_type in ['LinearSVC', 'SVC']:
    print(f"  ℹ️ 使用 LinearExplainer (SVM模式)...")
    try:
        explainer = shap.LinearExplainer(
            actual_classifier, X_sample,
            feature_perturbation="interventional"
        )
        print("  ✓ LinearExplainer 创建完成")
    except Exception as e:
        print(f"  ⚠️ LinearExplainer 失败 ({e})，转为 KernelExplainer...")
        if hasattr(actual_classifier, 'decision_function'):
            from scipy.special import expit
            predict_fn = lambda x: expit(actual_classifier.decision_function(x))
            print("  ✓ 使用 decision_function + sigmoid")
        else:
            raise ValueError("SVM模型没有decision_function")
        background_data = X_sample.sample(min(100, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(predict_fn, background_data)

elif model_type in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
    print(f"  ℹ️ 使用 TreeExplainer...")
    explainer = shap.TreeExplainer(actual_classifier)
    print("  ✓ TreeExplainer 创建完成")

else:
    print(f"  ℹ️ 使用 KernelExplainer (通用模式)...")
    print("  ⏳ 初始化中（需要几分钟）...")
    background_data = X_sample.sample(min(200, len(X_sample)), random_state=42)

    if hasattr(actual_classifier, 'predict_proba'):
        predict_fn = lambda x: actual_classifier.predict_proba(x)[:, 1]
        print("  ✓ 使用 predict_proba")
    elif hasattr(actual_classifier, 'decision_function'):
        from scipy.special import expit
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
        base_values=shap_values.base_values[:, 1] if shap_values.base_values.ndim > 1 else shap_values.base_values,
        data=shap_values.data,
        feature_names=shap_values.feature_names
    )

print(f"  ✓ 最终 SHAP 值矩阵形状: {shap_values_pos.values.shape}")

if np.abs(shap_values_pos.values).max() < 1e-6:
    print("  ⚠️⚠️⚠️ 警告：所有SHAP值接近0！模型可能有问题！")
else:
    print("  ✓ SHAP值计算成功！")

# ============================================================================
# 📊 步骤 4 – 14: 出图
# ============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

C_BAR     = COLORS_LIST[1]   
C_ANNOT   = '#636363'        # 灰色 — 数值标注文字

# ── 准备数据 ──────────────────────────────────────────────────
pred_probs = classifier.predict_proba(X_sample)[:, 1]
high_risk_idx = np.argmax(pred_probs)
low_risk_idx  = np.argmin(pred_probs)
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
# Panel A — SHAP Summary Beeswarm
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

# colorbar 美化
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
# Panel B — Feature Importance 
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

# 数值标注 (统一 4 位小数)
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
    a.tick_params(labelsize=8)
    for txt in a.texts:
        txt.set_fontsize(8)


if len(all_ax_c) > 2:
    for label in all_ax_c[2].get_xticklabels()[1:]:
        label.set_visible(False)

plt.tight_layout()
plt.savefig('figures/Fig3C_Waterfall_HighRisk.png', dpi=DPI, bbox_inches='tight')
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
    a.tick_params(labelsize=8)
    for txt in a.texts:
        txt.set_fontsize(8)

if len(all_ax_d) > 2:
    for label in all_ax_d[2].get_xticklabels()[1:]:
        label.set_visible(False)

plt.tight_layout()
plt.savefig('figures/Fig3D_Waterfall_LowRisk.png', dpi=DPI, bbox_inches='tight')
print("  ✓ Fig3D_Waterfall_LowRisk.png/.pdf")
plt.close()

# ------------------------------------------
# 步骤 4: SHAP Summary Plot (蜂群图)
# ------------------------------------------
print("\n[4/14] 绘制 SHAP Summary Plot (蜂群图)...")

fig, ax = plt.subplots(figsize=(6, 3))

shap.summary_plot(
    shap_values_pos.values,
    X_sample,
    feature_names=feature_names,
    show=False,
    max_display=20,
    color_bar=True,
    plot_size=None,       # 由外部 fig 控制
)

# 获取当前 axes 并美化
ax = plt.gca()
sci_ax(ax, xlabel='SHAP value', ylabel='')
ax.set_title('SHAP Summary', fontsize=9,  pad=8)

# 调整 colorbar 字号
for child in fig.get_children():
    if isinstance(child, mpl.colorbar.Colorbar) or hasattr(child, 'set_label'):
        pass
# colorbar 文字可通过 fig.axes 获取
if len(fig.axes) > 1:
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=6)
    cbar_ax.set_ylabel('Feature value', fontsize=7)

plt.tight_layout()
plt.savefig('figures/SHAP_Summary_Beeswarm.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Summary_Beeswarm.png")
plt.close()

# ------------------------------------------
# 步骤 5: SHAP Feature Importance (自定义条形图)
# ------------------------------------------
print("\n[5/14] 绘制 SHAP Feature Importance...")

mean_abs_shap = np.abs(shap_values_pos.values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': mean_abs_shap
}).sort_values('SHAP_Importance', ascending=False)

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# 自定义蓝→紫→红渐变，与蜂群图一致
cmap = LinearSegmentedColormap.from_list(
    'blue_purple_red', ["#008AFB", "#9D28B0", "#FF0554"], N=256
)
# cmap = shap.plots.colors.red_blue

fig, ax = plt.subplots(figsize=(6, 3))
top_n = min(8, len(importance_df))
importance_top = importance_df.head(top_n).iloc[::-1]  # 最重要的在上

# 归一化 → 映射颜色
norm = plt.Normalize(
    vmin=importance_top['SHAP_Importance'].min(),
    vmax=importance_top['SHAP_Importance'].max()
)
colors = [cmap(norm(v)) for v in importance_top['SHAP_Importance']]

# 条形高度随重要性缩放（0.35 ~ 0.75）
heights = 0.35 + 0.4 * norm(importance_top['SHAP_Importance'].values)

bars = ax.barh(
    range(len(importance_top)),
    importance_top['SHAP_Importance'],
    color=colors,
    edgecolor='white', linewidth=0.3,
    height=heights
)

ax.set_yticks(range(len(importance_top)))
ax.set_yticklabels(importance_top['Feature'], fontsize=7)
sci_ax(ax, xlabel='Mean |SHAP value|', title='Feature Importance')

for i, (bar, val) in enumerate(zip(bars, importance_top['SHAP_Importance'])):
    ax.text(val + importance_top['SHAP_Importance'].max() * 0.02, i,
            f'{val:.4f}', va='center', fontsize=6, color='#555555')

# 添加colorbar与蜂群图呼应
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
cbar.set_label('Importance', fontsize=7)
cbar.ax.tick_params(labelsize=6)
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig('figures/SHAP_Importance_Bar.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Importance_Bar.png")
plt.close()

importance_df.to_csv('figures/SHAP_Importance.csv', index=False)
print("  ✓ 已保存: figures/SHAP_Importance.csv")

print("\n  Top 10 最重要特征:")
for idx, row in importance_df.head(10).iterrows():
    print(f"    {idx+1:2d}. {row['Feature']:30s} {row['SHAP_Importance']:.6f}")

# ------------------------------------------
# 步骤 6: SHAP Bar Plot (官方条形图)
# ------------------------------------------
print("\n[6/14] 绘制 SHAP Bar Plot...")

fig, ax = plt.subplots(figsize=(4.5, 3))

shap.summary_plot(
    shap_values_pos.values,
    X_sample,
    feature_names=feature_names,
    plot_type='bar',
    show=False,
    max_display=20,
    plot_size=None,
)

ax = plt.gca()
sci_ax(ax, xlabel='Mean |SHAP value|', title='Feature Importance (SHAP)')

plt.tight_layout()
plt.savefig('figures/SHAP_Bar_Official.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Bar_Official.png")
plt.close()

# ------------------------------------------
# 步骤 7: SHAP Waterfall Plot (瀑布图)
# ------------------------------------------
print("\n[7/14] 绘制 SHAP Waterfall Plot (高风险&低风险样本)...")

pred_probs = classifier.predict_proba(X_sample)[:, 1]
high_risk_idx = np.argmax(pred_probs)
low_risk_idx = np.argmin(pred_probs)
high_risk_prob = pred_probs[high_risk_idx]
low_risk_prob = pred_probs[low_risk_idx]

print(f"  ✓ 高风险样本: #{high_risk_idx}, 预测概率: {high_risk_prob:.4f}")
print(f"  ✓ 低风险样本: #{low_risk_idx}, 预测概率: {low_risk_prob:.4f}")

# 高风险样本
fig = plt.figure(figsize=(5, 3))
shap.waterfall_plot(shap_values_pos[high_risk_idx], show=False, max_display=15)

all_axes = fig.get_axes()
ax = all_axes[0]

ax.set_title(f'High-risk sample (Pred Prob = {high_risk_prob:.3f})',
             fontsize=9,  pad=10)
despine(ax, top=True, right=True)

# 统一所有 axes 的字号
for a in all_axes:
    a.tick_params(labelsize=8)
    for txt in a.texts:
        txt.set_fontsize(8)

# 修复顶部 x 轴重复标签
if len(all_axes) > 2:
    top_labels = all_axes[2].get_xticklabels()
    if len(top_labels) > 1:
        for label in top_labels[1:]:
            label.set_visible(False)

plt.tight_layout()
plt.savefig('figures/SHAP_Waterfall_HighRisk.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 低风险样本
fig = plt.figure(figsize=(5, 3))
shap.waterfall_plot(shap_values_pos[low_risk_idx], show=False, max_display=15)

all_axes = fig.get_axes()
ax = all_axes[0]

ax.set_title(f'Low-risk sample (Pred Prob = {low_risk_prob:.3f})',
             fontsize=9,  pad=10)
despine(ax, top=True, right=True)

# 统一所有 axes 的字号
for a in all_axes:
    a.tick_params(labelsize=8)
    for txt in a.texts:
        txt.set_fontsize(8)

# 修复顶部 x 轴重复标签
if len(all_axes) > 2:
    top_labels = all_axes[2].get_xticklabels()
    if len(top_labels) > 1:
        for label in top_labels[1:]:
            label.set_visible(False)

plt.tight_layout()
plt.savefig('figures/SHAP_Waterfall_LowRisk.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ------------------------------------------
# 步骤 8: SHAP Dependence Plots
# ------------------------------------------
print("\n[8/14] 绘制 SHAP Dependence Plots...")

top_features = importance_df['Feature'].head(5).tolist()
print(f"  ✓ 选择特征: {', '.join(top_features)}")

for idx, feat in enumerate(top_features):
    feat_idx = list(feature_names).index(feat)
    try:
        fig, ax = plt.subplots(figsize=(4, 3.2))
        shap.dependence_plot(
            feat_idx,
            shap_values_pos.values,
            X_sample,
            feature_names=feature_names,
            interaction_index=None,
            ax=ax,
            show=False,
            dot_size=8,
            alpha=0.9
        )
        sci_ax(ax, xlabel=feat, ylabel='SHAP value', title=f'{feat}')

        plt.tight_layout()
        safe_name = feat.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        save_path = f'figures/SHAP_Dependence/SHAP_Dependence_{safe_name}.png'
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  ✓ 已保存: {save_path}")
        plt.close()
    except Exception as e:
        print(f"  ⚠️ 特征 {feat} 依赖图绘制失败: {e}")
        plt.close()

print("  ✓ 所有 Dependence Plots 已保存至 figures/SHAP_Dependence/")

# ------------------------------------------
# 步骤 9: SHAP Heatmap
# ------------------------------------------
print("\n[9/14] 绘制 SHAP Heatmap...")

n_samples_heatmap = min(50, len(X_sample))
sorted_indices = np.argsort(pred_probs)[::-1][:n_samples_heatmap]

shap_subset = shap.Explanation(
    values=shap_values_pos.values[sorted_indices],
    base_values=(shap_values_pos.base_values
                 if isinstance(shap_values_pos.base_values, (int, float))
                 else shap_values_pos.base_values[sorted_indices]),
    data=X_sample.iloc[sorted_indices].values,
    feature_names=feature_names
)

shap.plots.heatmap(shap_subset, show=False, max_display=15)

fig = plt.gcf()
for a in fig.axes:
    # 统一 tick 字号
    a.tick_params(labelsize=7)
    # 统一 xlabel / ylabel 字号
    if a.get_ylabel():
        a.yaxis.label.set_text(
            'SHAP value' if 'SHAP value' in a.get_ylabel() else a.get_ylabel()
        )
        a.yaxis.label.set_size(8)
    if a.get_xlabel():
        a.xaxis.label.set_size(8)
    # 统一 title 字号（f(x) 上方的线图标题）
    if a.get_title():
        a.title.set_size(8)

plt.savefig('figures/SHAP_Heatmap.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Heatmap.png")
plt.close()

# ------------------------------------------
# 步骤 10: SHAP Decision Plot
# ------------------------------------------
print("\n[10/14] 绘制 SHAP Decision Plot...")

high_risk_indices = np.argsort(pred_probs)[-3:][::-1]
mid_risk_indices = np.argsort(np.abs(pred_probs - 0.5))[:3]
low_risk_indices = np.argsort(pred_probs)[:3]
selected_indices = np.concatenate([high_risk_indices, mid_risk_indices, low_risk_indices])

if isinstance(shap_values_pos.base_values, np.ndarray):
    expected_value = (shap_values_pos.base_values[0]
                      if len(shap_values_pos.base_values.shape) > 0
                      else float(shap_values_pos.base_values))
else:
    expected_value = shap_values_pos.base_values

shap.decision_plot(
    expected_value,
    shap_values_pos.values[selected_indices],
    X_sample.iloc[selected_indices],
    feature_names=feature_names,
    feature_order='importance',
    show=False,
    highlight=[0, 1, 2]
)

fig = plt.gcf()
for a in fig.axes:
    a.tick_params(labelsize=7)
    if a.get_ylabel():
        a.yaxis.label.set_size(8)
    if a.get_xlabel():
        a.xaxis.label.set_size(8)
    if a.get_title():
        a.title.set_size(9)

plt.savefig('figures/SHAP_Decision_Plot.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Decision_Plot.png")
plt.close()

# ------------------------------------------
# 步骤 11: SHAP 值分布箱线图
# ------------------------------------------
print("\n[11/14] 绘制 SHAP 值分布箱线图...")

fig, ax = plt.subplots(figsize=(5, 4))

top10_features = importance_df['Feature'].head(10).tolist()
top10_indices = [list(feature_names).index(f) for f in top10_features]
top10_shap_values = [shap_values_pos.values[:, i] for i in top10_indices]

bp = ax.boxplot(
    top10_shap_values,
    labels=top10_features,
    vert=False,
    patch_artist=True,
    showmeans=True,
    widths=0.55,
    meanprops=dict(marker='D', markerfacecolor=C_RED, markeredgecolor='white',
                   markersize=4, markeredgewidth=0.5),
    medianprops=dict(color='white', linewidth=1),
    whiskerprops=dict(linewidth=0.6),
    capprops=dict(linewidth=0.6),
    flierprops=dict(marker='o', markersize=2, alpha=0.4,
                    markerfacecolor=C_GRAY, markeredgecolor='none')
)

for patch in bp['boxes']:
    patch.set_facecolor(C_BLUE)
    patch.set_alpha(0.6)
    patch.set_edgecolor('white')
    patch.set_linewidth(0.5)

ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
sci_ax(ax, xlabel='SHAP value', title='SHAP Value Distribution')

plt.tight_layout()
plt.savefig('figures/SHAP_Value_Distribution.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Value_Distribution.png")
plt.close()

# ------------------------------------------
# 步骤 12: SHAP 正负影响统计分析
# ------------------------------------------
print("\n[12/14] 绘制 SHAP 正负影响分析...")

positive_impact_pct, negative_impact_pct = [], []
mean_positive_shap, mean_negative_shap = [], []

for i, feat in enumerate(feature_names):
    shap_vals = shap_values_pos.values[:, i]
    pos_mask = shap_vals > 0
    neg_mask = shap_vals < 0

    positive_impact_pct.append((pos_mask.sum() / len(shap_vals)) * 100)
    negative_impact_pct.append((neg_mask.sum() / len(shap_vals)) * 100)
    mean_positive_shap.append(shap_vals[pos_mask].mean() if pos_mask.any() else 0)
    mean_negative_shap.append(shap_vals[neg_mask].mean() if neg_mask.any() else 0)

impact_df = pd.DataFrame({
    'Feature': feature_names,
    'Positive_%': positive_impact_pct,
    'Negative_%': negative_impact_pct,
    'Mean_Positive_SHAP': mean_positive_shap,
    'Mean_Negative_SHAP': mean_negative_shap
})

top15_impact = (impact_df.merge(importance_df, on='Feature')
                .sort_values('SHAP_Importance', ascending=False)
                .head(15).iloc[::-1])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

y_pos = range(len(top15_impact))

# 左图：正负影响比例
axes[0].barh(y_pos, top15_impact['Positive_%'],
             color=C_POS, alpha=0.75, height=0.6, label='Positive')
axes[0].barh(y_pos, -top15_impact['Negative_%'],
             color=C_NEG, alpha=0.75, height=0.6, label='Negative')
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(top15_impact['Feature'], fontsize=6.5)
axes[0].axvline(0, color='black', linewidth=0.5)
sci_ax(axes[0], xlabel='Samples (%)', title='Impact direction')
axes[0].legend(fontsize=6, loc='lower right')

# 右图：平均SHAP值
axes[1].barh(y_pos, top15_impact['Mean_Positive_SHAP'],
             color=C_POS, alpha=0.75, height=0.6, label='Mean positive')
axes[1].barh(y_pos, top15_impact['Mean_Negative_SHAP'],
             color=C_NEG, alpha=0.75, height=0.6, label='Mean negative')
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels([])   # 共享标签，右图不重复
axes[1].axvline(0, color='black', linewidth=0.5)
sci_ax(axes[1], xlabel='Mean SHAP value', title='Average SHAP by direction')
axes[1].legend(fontsize=6, loc='lower right')

plt.tight_layout(w_pad=1.5)
plt.savefig('figures/SHAP_Positive_Negative_Impact.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Positive_Negative_Impact.png")
plt.close()

impact_df.to_csv('figures/SHAP_Impact_Statistics.csv', index=False)
print("  ✓ 已保存: figures/SHAP_Impact_Statistics.csv")

# ------------------------------------------
# 步骤 13: SHAP 累积贡献图
# ------------------------------------------
print("\n[13/14] 绘制 SHAP 累积贡献图...")

sorted_importance = importance_df.sort_values('SHAP_Importance', ascending=False)
cumulative_importance = (np.cumsum(sorted_importance['SHAP_Importance'])
                         / sorted_importance['SHAP_Importance'].sum() * 100)

fig, ax = plt.subplots(figsize=(4, 3.2))

# 填充 + 曲线
ax.fill_between(range(1, len(cumulative_importance) + 1), cumulative_importance,
                alpha=0.12, color=C_BLUE)
ax.plot(range(1, len(cumulative_importance) + 1), cumulative_importance,
        color=C_BLUE, linewidth=1.5, zorder=3)

# 阈值线
ax.axhline(80, color=C_POS, linestyle='--', linewidth=0.6, alpha=0.7)
ax.axhline(90, color=C_ORANGE, linestyle='--', linewidth=0.6, alpha=0.7)

# 关键点
n_80 = int(np.argmax(cumulative_importance >= 80)) + 1
n_90 = int(np.argmax(cumulative_importance >= 90)) + 1
ax.scatter([n_80], [80], s=30, c=C_POS, zorder=5,
           edgecolors='white', linewidths=0.6)
ax.scatter([n_90], [90], s=30, c=C_ORANGE, zorder=5,
           edgecolors='white', linewidths=0.6)

# 标注
offset_x = max(1.5, len(cumulative_importance) * 0.08)
ax.annotate(f'n = {n_80} (80 %)', xy=(n_80, 80),
            xytext=(n_80 + offset_x, 72),
            fontsize=6.5, color=C_POS,
            arrowprops=dict(arrowstyle='->', color=C_POS, lw=0.8))
ax.annotate(f'n = {n_90} (90 %)', xy=(n_90, 90),
            xytext=(n_90 + offset_x, 95),
            fontsize=6.5, color=C_ORANGE,
            arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=0.8))

sci_ax(ax, xlabel='Number of features',
       ylabel='Cumulative importance (%)',
       title='Cumulative Feature Importance')
ax.set_ylim(0, 105)
ax.set_xlim(0, len(cumulative_importance) + 1)

plt.tight_layout()
plt.savefig('figures/SHAP_Cumulative_Importance.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Cumulative_Importance.png")
print(f"  ℹ️ {n_80} 个特征贡献了 80% 的预测能力")
print(f"  ℹ️ {n_90} 个特征贡献了 90% 的预测能力")
plt.close()

# ------------------------------------------
# 步骤 14: SHAP 与原始特征相关性分析
# ------------------------------------------
print("\n[14/14] 绘制 SHAP 与原始特征相关性分析...")

from scipy.stats import pearsonr

correlations, p_values = [], []
for i, feat in enumerate(feature_names):
    feat_values = X_sample.iloc[:, i].values
    shap_vals = shap_values_pos.values[:, i]
    mask = ~(np.isnan(feat_values) | np.isnan(shap_vals))
    if mask.sum() > 10:
        corr, p = pearsonr(feat_values[mask], shap_vals[mask])
        correlations.append(corr)
        p_values.append(p)
    else:
        correlations.append(np.nan)
        p_values.append(np.nan)

corr_df = pd.DataFrame({
    'Feature': feature_names,
    'Correlation': correlations,
    'P_Value': p_values,
    'Abs_Correlation': np.abs(correlations)
}).sort_values('Abs_Correlation', ascending=False)

top20_corr = corr_df.head(20).iloc[::-1]

fig, ax = plt.subplots(figsize=(4.5, 5))
colors_corr = [C_NEG if c < 0 else C_POS for c in top20_corr['Correlation']]
bars = ax.barh(range(len(top20_corr)), top20_corr['Correlation'],
               color=colors_corr, alpha=0.7, height=0.6)

ax.set_yticks(range(len(top20_corr)))
ax.set_yticklabels(top20_corr['Feature'], fontsize=6.5)
ax.axvline(0, color='black', linewidth=0.5)
sci_ax(ax, xlabel='Pearson r (feature value ↔ SHAP value)',
       title='SHAP–Feature Correlation')

# 标注
for i, (bar, (_, row)) in enumerate(zip(bars, top20_corr.iterrows())):
    corr_val = row['Correlation']
    p_val = row['P_Value']
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    label = f'{corr_val:.2f}{sig}'
    x_pos = corr_val + (0.02 if corr_val > 0 else -0.02)
    ha = 'left' if corr_val > 0 else 'right'
    ax.text(x_pos, i, label, va='center', ha=ha, fontsize=5.5, color=C_GRAY)

plt.tight_layout()
plt.savefig('figures/SHAP_Feature_Correlation.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/SHAP_Feature_Correlation.png")
plt.close()

corr_df.to_csv('figures/SHAP_Feature_Correlation.csv', index=False)
print("  ✓ 已保存: figures/SHAP_Feature_Correlation.csv")

# ------------------------------------------
# 步骤 15: SHAP Force Plot (HTML)
# ------------------------------------------
print("\n[额外] 生成 SHAP Force Plot (HTML)...")

try:
    shap.initjs()

    if hasattr(shap_values_pos, 'base_values'):
        base_value = shap_values_pos.base_values[high_risk_idx]
    elif isinstance(explainer.expected_value, np.ndarray):
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    force_plot = shap.force_plot(
        base_value,
        shap_values_pos.values[high_risk_idx],
        X_sample.iloc[high_risk_idx],
        show=False
    )
    shap.save_html('figures/SHAP_Force_Plot_HighRisk.html', force_plot)
    print("  ✓ 已保存: figures/SHAP_Force_Plot_HighRisk.html")

    force_plot_all = shap.force_plot(
        base_value if isinstance(base_value, (int, float)) else shap_values_pos.base_values[:100],
        shap_values_pos.values[:100],
        X_sample.iloc[:100],
        show=False
    )
    shap.save_html('figures/SHAP_Force_Plot_All.html', force_plot_all)
    print("  ✓ 已保存: figures/SHAP_Force_Plot_All.html")

except Exception as e:
    print(f"  ⚠️ Force Plot 生成失败: {e}")

try:
    shap.force_plot(
        shap_values_pos.base_values[high_risk_idx],
        shap_values_pos.values[high_risk_idx],
        X_sample.iloc[high_risk_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )

    fig = plt.gcf()
    import re
    for text_obj in fig.findobj(mpl.text.Text):
        text_obj.set_fontsize(10)
        # 将长小数截断为2位
        s = text_obj.get_text()
        s = re.sub(r'(\d+\.\d{2})\d+', r'\1', s)
        text_obj.set_text(s)

    plt.savefig('figures/SHAP_Force_Plot_HighRisk.png', dpi=DPI, bbox_inches='tight')
    print("  ✓ 已保存: figures/SHAP_Force_Plot_HighRisk.png")
    plt.close()

except Exception as e:
    print(f"  ⚠️ Force Plot 生成失败: {e}")

# ------------------------------------------
# 步骤 16: 综合摘要报告
# ------------------------------------------
print(f"\n{'='*60}")
print("📊 SHAP 分析摘要报告")
print(f"{'='*60}")

print(f"\n【数据信息】")
print(f"  - 分析样本数: {len(X_sample)}")
print(f"  - 特征数量: {len(feature_names)}")
print(f"  - 模型类型: {type(classifier).__name__}")
print(f"  - 解释器类型: {type(explainer).__name__}")

print(f"\n【特征重要性】")
print(f"  - {n_80} 个特征贡献了 80% 的预测能力")
print(f"  - {n_90} 个特征贡献了 90% 的预测能力")

print(f"\n【Top 10 最重要特征】")
for idx, row in importance_df.head(10).iterrows():
    pos_pct = impact_df[impact_df['Feature'] == row['Feature']]['Positive_%'].values[0]
    print(f"  {idx+1:2d}. {row['Feature']:35s} | SHAP: {row['SHAP_Importance']:.6f} | Positive: {pos_pct:.1f}%")

print(f"\n【高风险样本分析】")
print(f"  - 样本索引: {high_risk_idx}")
print(f"  - 预测概率: {high_risk_prob:.4f}")
print(f"  - Top 3 驱动因素:")
sample_shap = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Value': shap_values_pos.values[high_risk_idx]
}).sort_values('SHAP_Value', ascending=False)
for i, row in sample_shap.head(3).iterrows():
    print(f"    {row['Feature']:30s} SHAP = {row['SHAP_Value']:+.4f}")

print(f"\n【低风险样本分析】")
print(f"  - 样本索引: {low_risk_idx}")
print(f"  - 预测概率: {low_risk_prob:.4f}")
print(f"  - Top 3 保护因素:")
sample_shap_low = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Value': shap_values_pos.values[low_risk_idx]
}).sort_values('SHAP_Value', ascending=True)
for i, row in sample_shap_low.head(3).iterrows():
    print(f"    {row['Feature']:30s} SHAP = {row['SHAP_Value']:+.4f}")

print(f"\n【特征相关性发现】")
top_pos_corr = corr_df[corr_df['Correlation'] > 0].head(3)
top_neg_corr = corr_df[corr_df['Correlation'] < 0].head(3)
print(f"  正相关 Top 3:")
for _, row in top_pos_corr.iterrows():
    print(f"    {row['Feature']:30s} r = {row['Correlation']:+.3f} (p = {row['P_Value']:.4f})")
print(f"  负相关 Top 3:")
for _, row in top_neg_corr.iterrows():
    print(f"    {row['Feature']:30s} r = {row['Correlation']:+.3f} (p = {row['P_Value']:.4f})")

print(f"\n【输出文件】")
output_files = [
    "SHAP_Summary_Beeswarm.png",
    "SHAP_Importance_Bar.png",
    "SHAP_Bar_Official.png",
    "SHAP_Waterfall_HighRisk.png",
    "SHAP_Waterfall_LowRisk.png",
    "SHAP_Dependence/SHAP_Dependence_*.png",
    "SHAP_Heatmap.png",
    "SHAP_Decision_Plot.png",
    "SHAP_Value_Distribution.png",
    "SHAP_Positive_Negative_Impact.png",
    "SHAP_Cumulative_Importance.png",
    "SHAP_Feature_Correlation.png",
    "SHAP_Importance.csv",
    "SHAP_Impact_Statistics.csv",
    "SHAP_Feature_Correlation.csv",
    "SHAP_Force_Plot_*.html"
]
for file in output_files:
    print(f"  ✓ {file}")

print(f"\n{'='*60}")
print("✅ SHAP 分析完成！")
print(f"{'='*60}\n")

# ------------------------------------------
# 保存完整分析结果
# ------------------------------------------
shap_results = {
    'shap_values': shap_values_pos,
    'importance_df': importance_df,
    'impact_df': impact_df,
    'corr_df': corr_df,
    'X_sample': X_sample,
    'feature_names': feature_names,
    'high_risk_idx': high_risk_idx,
    'low_risk_idx': low_risk_idx,
    'pred_probs': pred_probs,
    'n_80_threshold': n_80,
    'n_90_threshold': n_90
}

joblib.dump(shap_results, 'figures/SHAP_Analysis_Results.pkl')
print("💾 SHAP 分析结果已保存至: figures/SHAP_Analysis_Results.pkl\n")

# ============================================================================
# 恢复主样式 (SHAP 专用样式仅在 SHAP 区段内生效)
# ============================================================================
mpl.rcParams.update(_saved_rcParams)
set_sci_style()  # 重新应用主 SCI 样式


# ------------------------------------------
# 1. Bootstrap AUC 分布图（外部验证稳定性）
# ------------------------------------------
print("\n[额外-1] 绘制 Bootstrap AUC 分布图...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1.1 直方图 + 核密度估计
axes[0].hist(df_bootstrap_raw['AUC'], bins=50, alpha=0.7, 
             color=COLOR_EXTERNAL, edgecolor='none', density=True)

# 添加核密度估计曲线
from scipy.stats import gaussian_kde
kde = gaussian_kde(df_bootstrap_raw['AUC'])
x_range = np.linspace(df_bootstrap_raw['AUC'].min(), 
                      df_bootstrap_raw['AUC'].max(), 100)
axes[0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

# 添加均值和置信区间
mean_auc = df_bootstrap_summary.loc['AUC', 'Mean']
ci_lower = df_bootstrap_summary.loc['AUC', '95%_CI_Lower']
ci_upper = df_bootstrap_summary.loc['AUC', '95%_CI_Upper']

axes[0].axvline(mean_auc, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_auc:.4f}')
axes[0].axvline(ci_lower, color='orange', linestyle=':', linewidth=2, 
                label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
axes[0].axvline(ci_upper, color='orange', linestyle=':', linewidth=2)

axes[0].set_xlabel('AUC')
axes[0].set_ylabel('Density')
axes[0].set_title('Bootstrap AUC distribution', 
                  fontsize=13)
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)

# 1.2 箱线图 + 小提琴图
parts = axes[1].violinplot([df_bootstrap_raw['AUC']], 
                           positions=[0], widths=0.7, 
                           showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor(COLOR_EXTERNAL)
    pc.set_alpha(0.7)

# 叠加箱线图
bp = axes[1].boxplot([df_bootstrap_raw['AUC']], positions=[0], 
                     widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor='white', alpha=0.8),
                     medianprops=dict(color='#E64B35', linewidth=1.5))

axes[1].set_ylabel('AUC')
axes[1].set_title('Bootstrap AUC variability')
axes[1].set_xticks([0])
axes[1].set_xticklabels([f'n={len(df_bootstrap_raw)}'])
axes[1].grid(axis='y', alpha=0.3)

stats_text = f"Mean: {mean_auc:.4f}\nMedian: {df_bootstrap_summary.loc['AUC', 'Median']:.4f}\nStd: {df_bootstrap_summary.loc['AUC', 'Std']:.4f}"
axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='grey', alpha=0.85, linewidth=1.2))

plt.tight_layout()
plt.savefig('figures/Bootstrap_AUC_Distribution.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/Bootstrap_AUC_Distribution.png")
plt.close()


# ==============================================================
# 3. 内部验证 Bootstrap 重采样 + Harrell Optimism Correction
# ==============================================================
print("\n" + "="*70)
print("  内部验证 Bootstrap 重采样 & Harrell Optimism Correction")
print("="*70)
import time
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
from scipy import interpolate

# ------------------------------------------
# 颜色映射（基于 COLORS_LIST）
# ------------------------------------------
C_INT      = COLORS_LIST[3]   # #3C5488 深蓝 — 内部验证
C_EXT      = COLORS_LIST[0]   # #E64B35 红色 — 外部验证
C_CORRECT  = COLORS_LIST[2]   # #00A087 绿色 — Optimism-Corrected
C_OOF_LINE = COLORS_LIST[1]   # #4DBBD5 青色 — OOF 原始曲线
C_EXT_LINE = COLORS_LIST[7]   # #DC0000 深红 — 外部原始曲线
C_APPARENT = COLORS_LIST[5]   # #8491B4 灰蓝 — Apparent
C_BRIER    = COLORS_LIST[4]   # #F39B7F 珊瑚橙 — Brier / 第三指标

# ------------------------------------------
# 3.1 准备数据
# ------------------------------------------
print("\n[Step 1] 准备开发集数据...")

if hasattr(best_model, 'named_steps') or hasattr(best_model, 'steps'):
    X_dev = X_train.copy()
    model_is_pipeline = True
    print("  ✓ 检测到 Pipeline 模型，使用原始特征")
else:
    X_dev = preprocessor.transform(X_train)
    model_is_pipeline = False
    print("  ✓ 检测到独立分类器，已完成特征预处理")

y_dev = y_train.copy()
print(f"  ✓ 开发集规模: {X_dev.shape[0]} 样本, "
      f"{X_dev.shape[1] if hasattr(X_dev, 'shape') and len(X_dev.shape) > 1 else 'N/A'} 特征")
print(f"  ✓ 正例比例: {y_dev.mean():.4f} ({y_dev.sum()}/{len(y_dev)})")

# ------------------------------------------
# 3.2 Apparent Performance
# ------------------------------------------
print("\n[Step 2] 计算 Apparent Performance...")

prob_apparent = best_model.predict_proba(X_dev)[:, 1]
apparent_auc = roc_auc_score(y_dev, prob_apparent)
apparent_brier = brier_score_loss(y_dev, prob_apparent)
apparent_ap = average_precision_score(y_dev, prob_apparent)

print(f"  ✓ Apparent AUC:   {apparent_auc:.4f}")
print(f"  ✓ Apparent AP:    {apparent_ap:.4f}")
print(f"  ✓ Apparent Brier: {apparent_brier:.4f}")

# ------------------------------------------
# 3.3 OOF Performance（5-fold CV × 20 repeats）
# ------------------------------------------
print("\n[Step 3] 计算 OOF Performance...")

oof_auc = roc_auc_score(y_int, prob_int)
oof_brier = brier_score_loss(y_int, prob_int)
oof_ap = average_precision_score(y_int, prob_int)

print(f"  ✓ OOF AUC:   {oof_auc:.4f}")
print(f"  ✓ OOF AP:    {oof_ap:.4f}")
print(f"  ✓ OOF Brier: {oof_brier:.4f}")

# ------------------------------------------
# 3.4 Harrell Bootstrap Optimism Correction (1000 iterations)
#     同时收集 ROC + PR 曲线数据
# ------------------------------------------
print("\n[Step 4] Harrell Bootstrap Optimism Correction (1000 次)...")
print("  ⏳ 每次迭代需要 clone + fit，请耐心等待...")

N_BOOT_INTERNAL = 1000
np.random.seed(42)

mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

# ---- 收集容器 ----
optimism_auc_list = []
optimism_brier_list = []
optimism_ap_list = []
auc_boot_list = []
auc_orig_list = []

boot_tprs_int = []     # ROC（bootstrap 模型 → 原始数据）
boot_precs_int = []    # PR （bootstrap 模型 → 原始数据）
boot_tprs_oof = []     # OOF bootstrap 重采样 ROC
boot_precs_oof = []    # OOF bootstrap 重采样 PR
boot_auc_oof = []
boot_ap_oof = []

start_time = time.time()
success_count = 0
fail_count = 0

for i in range(N_BOOT_INTERNAL):
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        eta = elapsed / (i + 1) * (N_BOOT_INTERNAL - i - 1)
        print(f"    进度: {i+1}/{N_BOOT_INTERNAL} | "
              f"已用时: {elapsed:.0f}s | 预计剩余: {eta:.0f}s")

    try:
        # ========== Part A: Harrell Optimism ==========
        idx_boot = np.random.choice(len(y_dev), size=len(y_dev), replace=True)

        if hasattr(X_dev, 'iloc'):
            X_boot = X_dev.iloc[idx_boot].reset_index(drop=True)
        else:
            X_boot = X_dev[idx_boot]
        y_boot = y_dev[idx_boot]

        if len(np.unique(y_boot)) < 2:
            fail_count += 1
            continue

        model_boot = clone(best_model)
        model_boot.fit(X_boot, y_boot)

        # 在 bootstrap 样本上预测
        prob_on_boot = model_boot.predict_proba(X_boot)[:, 1]
        auc_boot = roc_auc_score(y_boot, prob_on_boot)
        brier_boot = brier_score_loss(y_boot, prob_on_boot)
        ap_boot = average_precision_score(y_boot, prob_on_boot)

        # 在原始开发集上预测
        prob_on_orig = model_boot.predict_proba(X_dev)[:, 1]
        auc_orig = roc_auc_score(y_dev, prob_on_orig)
        brier_orig = brier_score_loss(y_dev, prob_on_orig)
        ap_orig = average_precision_score(y_dev, prob_on_orig)

        # Optimism
        optimism_auc_list.append(auc_boot - auc_orig)
        optimism_brier_list.append(brier_boot - brier_orig)
        optimism_ap_list.append(ap_boot - ap_orig)
        auc_boot_list.append(auc_boot)
        auc_orig_list.append(auc_orig)

        # ROC（bootstrap 模型 → 原始数据）
        fpr_i, tpr_i, _ = roc_curve(y_dev, prob_on_orig)
        interp_tpr = np.interp(mean_fpr, fpr_i, tpr_i)
        interp_tpr[0] = 0.0
        boot_tprs_int.append(interp_tpr)

        # PR（bootstrap 模型 → 原始数据）
        prec_i, rec_i, _ = precision_recall_curve(y_dev, prob_on_orig)
        sorted_idx = np.argsort(rec_i)
        interp_prec = np.interp(mean_recall, rec_i[sorted_idx], prec_i[sorted_idx])
        boot_precs_int.append(interp_prec)

        # ========== Part B: OOF Bootstrap 重采样 ==========
        idx_oof = np.random.choice(len(y_int), size=len(y_int), replace=True)
        y_oof_b = y_int[idx_oof]
        p_oof_b = prob_int[idx_oof]

        if len(np.unique(y_oof_b)) >= 2:
            fpr_o, tpr_o, _ = roc_curve(y_oof_b, p_oof_b)
            interp_tpr_o = np.interp(mean_fpr, fpr_o, tpr_o)
            interp_tpr_o[0] = 0.0
            boot_tprs_oof.append(interp_tpr_o)
            boot_auc_oof.append(roc_auc_score(y_oof_b, p_oof_b))

            prec_o, rec_o, _ = precision_recall_curve(y_oof_b, p_oof_b)
            sorted_idx_o = np.argsort(rec_o)
            interp_prec_o = np.interp(mean_recall, rec_o[sorted_idx_o], prec_o[sorted_idx_o])
            boot_precs_oof.append(interp_prec_o)
            boot_ap_oof.append(average_precision_score(y_oof_b, p_oof_b))

        success_count += 1

    except Exception as e:
        fail_count += 1
        if fail_count <= 3:
            print(f"    ⚠️ 第 {i+1} 次迭代失败: {e}")
        continue

elapsed_total = time.time() - start_time
print(f"\n  ✓ Bootstrap 完成: {success_count} 成功, {fail_count} 失败")
print(f"  ✓ 总耗时: {elapsed_total:.1f} 秒")

# ------------------------------------------
# 3.5 Optimism-Corrected 指标
# ------------------------------------------
print("\n[Step 5] 计算 Optimism-Corrected 指标...")

mean_optimism_auc = np.mean(optimism_auc_list)
std_optimism_auc = np.std(optimism_auc_list)
corrected_auc = apparent_auc - mean_optimism_auc

mean_optimism_ap = np.mean(optimism_ap_list)
std_optimism_ap = np.std(optimism_ap_list)
corrected_ap = apparent_ap - mean_optimism_ap

mean_optimism_brier = np.mean(optimism_brier_list)
corrected_brier = apparent_brier - mean_optimism_brier

ext_auc = roc_auc_score(y_ext, prob_ext)
ext_brier = brier_score_loss(y_ext, prob_ext)
ext_ap = average_precision_score(y_ext, prob_ext)

print(f"\n  {'='*65}")
print(f"  {'指标':<30} {'AUC':>10} {'AP':>10} {'Brier':>10}")
print(f"  {'-'*65}")
print(f"  {'Apparent (训练集):':<30} {apparent_auc:>10.4f} {apparent_ap:>10.4f} {apparent_brier:>10.4f}")
print(f"  {'Mean Optimism:':<30} {mean_optimism_auc:>10.4f} {mean_optimism_ap:>10.4f} {mean_optimism_brier:>10.4f}")
print(f"  {'Optimism-Corrected:':<30} {corrected_auc:>10.4f} {corrected_ap:>10.4f} {corrected_brier:>10.4f}")
print(f"  {'OOF (5×20 CV):':<30} {oof_auc:>10.4f} {oof_ap:>10.4f} {oof_brier:>10.4f}")
print(f"  {'External Validation:':<30} {ext_auc:>10.4f} {ext_ap:>10.4f} {ext_brier:>10.4f}")
print(f"  {'='*65}")
print(f"\n  📊 AUC Optimism: {mean_optimism_auc:.4f} ± {std_optimism_auc:.4f}")
print(f"  📊 AP  Optimism: {mean_optimism_ap:.4f} ± {std_optimism_ap:.4f}")
print(f"  📊 |Corrected AUC - OOF|      = {abs(corrected_auc - oof_auc):.4f}")
print(f"  📊 |Corrected AUC - External|  = {abs(corrected_auc - ext_auc):.4f}")

if mean_optimism_auc < 0.02:
    print("  ✅ AUC Optimism < 0.02 → 过拟合程度极低")
elif mean_optimism_auc < 0.05:
    print("  ✅ AUC Optimism < 0.05 → 过拟合程度可控")
else:
    print("  ⚠️ AUC Optimism ≥ 0.05 → 存在一定过拟合风险")

# ------------------------------------------
# 3.6 保存
# ------------------------------------------
print("\n[Step 6] 保存内部 Bootstrap 数据...")

df_internal_boot = pd.DataFrame({
    'AUC_boot': auc_boot_list,
    'AUC_orig': auc_orig_list,
    'Optimism_AUC': optimism_auc_list,
    'Optimism_AP': optimism_ap_list,
    'Optimism_Brier': optimism_brier_list
})
df_internal_boot.to_csv(f"{DATA_PATH}/Internal_Bootstrap_Optimism_{TIMESTAMP}.csv", index=False)

df_optimism_summary = pd.DataFrame({
    'Metric': ['AUC', 'AP', 'Brier Score'],
    'Apparent': [apparent_auc, apparent_ap, apparent_brier],
    'Mean_Optimism': [mean_optimism_auc, mean_optimism_ap, mean_optimism_brier],
    'Optimism_Corrected': [corrected_auc, corrected_ap, corrected_brier],
    'OOF_CV': [oof_auc, oof_ap, oof_brier],
    'External': [ext_auc, ext_ap, ext_brier]
})
df_optimism_summary.to_csv(f"{DATA_PATH}/Optimism_Correction_Summary_{TIMESTAMP}.csv", index=False)
print("  ✓ 已保存")


# ==============================================================
# 4. 外部 Bootstrap 曲线数据
# ==============================================================
print("\n[Step 7] 计算外部 Bootstrap 曲线数据...")

tprs_external = []
precs_external = []
aucs_external = []
aps_external = []

for boot_pred in bootstrap_predictions:
    y_true_b = boot_pred['true_labels']
    y_pred_b = boot_pred['pred_probs']

    if len(np.unique(y_true_b)) < 2:
        continue

    fpr_e, tpr_e, _ = roc_curve(y_true_b, y_pred_b)
    interp_tpr = np.interp(mean_fpr, fpr_e, tpr_e)
    interp_tpr[0] = 0.0
    tprs_external.append(interp_tpr)
    aucs_external.append(roc_auc_score(y_true_b, y_pred_b))

    prec_e, rec_e, _ = precision_recall_curve(y_true_b, y_pred_b)
    sorted_idx = np.argsort(rec_e)
    interp_prec = np.interp(mean_recall, rec_e[sorted_idx], prec_e[sorted_idx])
    precs_external.append(interp_prec)
    aps_external.append(average_precision_score(y_true_b, y_pred_b))

tprs_external = np.array(tprs_external)
precs_external = np.array(precs_external)

mean_tpr_ext = tprs_external.mean(axis=0)
std_tpr_ext = tprs_external.std(axis=0)
upper_tpr_ext = np.minimum(mean_tpr_ext + 1.96 * std_tpr_ext, 1)
lower_tpr_ext = np.maximum(mean_tpr_ext - 1.96 * std_tpr_ext, 0)
mean_auc_ext = np.mean(aucs_external)
std_auc_ext = np.std(aucs_external)

mean_prec_ext = precs_external.mean(axis=0)
std_prec_ext = precs_external.std(axis=0)
upper_prec_ext = np.minimum(mean_prec_ext + 1.96 * std_prec_ext, 1)
lower_prec_ext = np.maximum(mean_prec_ext - 1.96 * std_prec_ext, 0)
mean_ap_ext_boot = np.mean(aps_external)
std_ap_ext_boot = np.std(aps_external)

print(f"  ✓ 外部 ROC: Mean AUC = {mean_auc_ext:.4f} ± {std_auc_ext:.4f}")
print(f"  ✓ 外部 PR:  Mean AP  = {mean_ap_ext_boot:.4f} ± {std_ap_ext_boot:.4f}")


# ==============================================================
# 5. CI 带统计量
# ==============================================================

# 内部 Harrell Bootstrap
boot_tprs_int = np.array(boot_tprs_int)
boot_precs_int = np.array(boot_precs_int)

mean_tpr_int = boot_tprs_int.mean(axis=0)
std_tpr_int = boot_tprs_int.std(axis=0)
upper_tpr_int = np.minimum(mean_tpr_int + 1.96 * std_tpr_int, 1)
lower_tpr_int = np.maximum(mean_tpr_int - 1.96 * std_tpr_int, 0)
mean_auc_int_boot = np.mean(auc_orig_list)
std_auc_int_boot = np.std(auc_orig_list)

mean_prec_int = boot_precs_int.mean(axis=0)
std_prec_int = boot_precs_int.std(axis=0)
upper_prec_int = np.minimum(mean_prec_int + 1.96 * std_prec_int, 1)
lower_prec_int = np.maximum(mean_prec_int - 1.96 * std_prec_int, 0)

# OOF Bootstrap 重采样
boot_tprs_oof = np.array(boot_tprs_oof)
boot_precs_oof = np.array(boot_precs_oof)

mean_tpr_oof = boot_tprs_oof.mean(axis=0)
std_tpr_oof = boot_tprs_oof.std(axis=0)
upper_tpr_oof = np.minimum(mean_tpr_oof + 1.96 * std_tpr_oof, 1)
lower_tpr_oof = np.maximum(mean_tpr_oof - 1.96 * std_tpr_oof, 0)
mean_auc_oof_boot = np.mean(boot_auc_oof)
std_auc_oof_boot = np.std(boot_auc_oof)

mean_prec_oof = boot_precs_oof.mean(axis=0)
std_prec_oof = boot_precs_oof.std(axis=0)
upper_prec_oof = np.minimum(mean_prec_oof + 1.96 * std_prec_oof, 1)
lower_prec_oof = np.maximum(mean_prec_oof - 1.96 * std_prec_oof, 0)
mean_ap_oof_boot = np.mean(boot_ap_oof)
std_ap_oof_boot = np.std(boot_ap_oof)

# 原始曲线
fpr_oof_orig, tpr_oof_orig, _ = roc_curve(y_int, prob_int)
fpr_ext_orig, tpr_ext_orig, _ = roc_curve(y_ext, prob_ext)
prec_oof_orig, rec_oof_orig, _ = precision_recall_curve(y_int, prob_int)
prec_ext_orig, rec_ext_orig, _ = precision_recall_curve(y_ext, prob_ext)

prevalence_int = y_int.mean()
prevalence_ext = y_ext.mean()


# ==============================================================
# 6. 可视化
# ==============================================================
print("\n" + "="*70)
print("  可视化：SCI 顶刊风格 ROC & PR CI 带 + Optimism 分析")
print("="*70)

FIG_SINGLE = (3.5, 3.5)     # 单栏正方形
FIG_WIDE   = (7.2, 3.5)     # 双栏宽图
FIG_PANEL  = (7.2, 7.0)     # 2×2 面板

# ==================== 图1：ROC — 仅内部 ====================
fig, ax = plt.subplots(figsize=FIG_SINGLE)

ax.fill_between(mean_fpr, lower_tpr_int, upper_tpr_int,
                color=C_INT, alpha=0.15, label='95% CI')
ax.plot(mean_fpr, mean_tpr_int, color=C_INT, lw=1.5,
        label=f'Bootstrap mean (AUC = {mean_auc_int_boot:.3f})')
ax.plot(fpr_oof_orig, tpr_oof_orig, color=C_OOF_LINE, lw=1.2, ls='--',
        label=f'OOF (AUC = {oof_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_title('Internal Validation', fontsize=10)
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
sci_legend(ax, loc='lower right')

plt.tight_layout()
plt.savefig('figures/ROC_CI_Internal.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_CI_Internal.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/ROC_CI_Internal.png / .pdf")
plt.close()

# ==================== 图2：ROC — 仅外部 ====================
fig, ax = plt.subplots(figsize=FIG_SINGLE)

ax.fill_between(mean_fpr, lower_tpr_ext, upper_tpr_ext,
                color=C_EXT, alpha=0.15, label='95% CI')
ax.plot(mean_fpr, mean_tpr_ext, color=C_EXT, lw=1.5,
        label=f'Bootstrap mean (AUC = {mean_auc_ext:.3f})')
ax.plot(fpr_ext_orig, tpr_ext_orig, color=C_EXT_LINE, lw=1.2, ls='--',
        label=f'Original (AUC = {ext_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_title('External Validation', fontsize=10)
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
sci_legend(ax, loc='lower right')

plt.tight_layout()
plt.savefig('figures/ROC_CI_External.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_CI_External.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/ROC_CI_External.png / .pdf")
plt.close()

# ==================== 图3：ROC — 合并 ====================
fig, ax = plt.subplots(figsize=FIG_SINGLE)

ax.fill_between(mean_fpr, lower_tpr_int, upper_tpr_int, color=C_INT, alpha=0.12)
ax.plot(mean_fpr, mean_tpr_int, color=C_INT, lw=1.5,
        label=f'Internal (AUC = {mean_auc_int_boot:.3f})')
ax.fill_between(mean_fpr, lower_tpr_ext, upper_tpr_ext, color=C_EXT, alpha=0.12)
ax.plot(mean_fpr, mean_tpr_ext, color=C_EXT, lw=1.5,
        label=f'External (AUC = {mean_auc_ext:.3f})')
ax.plot(fpr_oof_orig, tpr_oof_orig, color=C_OOF_LINE, lw=1.0, ls='--',
        label=f'Internal OOF (AUC = {oof_auc:.3f})')
ax.plot(fpr_ext_orig, tpr_ext_orig, color=C_EXT_LINE, lw=1.0, ls='--',
        label=f'External original (AUC = {ext_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')

ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
sci_legend(ax, loc='lower right')

plt.tight_layout()
plt.savefig('figures/ROC_CI_Combined.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_CI_Combined.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/ROC_CI_Combined.png / .pdf")
plt.close()

# ==================== 图4：PR — 仅内部 ====================
fig, ax = plt.subplots(figsize=(3.5, 3.5))

ax.fill_between(mean_recall, lower_prec_int, upper_prec_int,
                color=C_INT, alpha=0.15, label='95% CI')
ax.plot(mean_recall, mean_prec_int, color=C_INT, lw=1.5,
        label=f'Bootstrap mean (AP = {corrected_ap:.3f})')
ax.plot(rec_oof_orig, prec_oof_orig, color=C_OOF_LINE, lw=1.2, ls='--',
        label=f'OOF (AP = {oof_ap:.3f})')
ax.axhline(y=prevalence_int, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_int:.3f}')
ax.set_title('Internal Validation', fontsize=10)
ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision (PPV)')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
sci_legend(ax, loc='lower left')

plt.tight_layout()
plt.savefig('figures/PR_CI_Internal.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/PR_CI_Internal.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/PR_CI_Internal.png / .pdf")
plt.close()

# ==================== 图5：PR — 仅外部 ====================
fig, ax = plt.subplots(figsize=(3.5, 3.5))

ax.fill_between(mean_recall, lower_prec_ext, upper_prec_ext,
                color=C_EXT, alpha=0.15, label='95% CI')
ax.plot(mean_recall, mean_prec_ext, color=C_EXT, lw=1.5,
        label=f'Bootstrap mean (AP = {mean_ap_ext_boot:.3f})')
ax.plot(rec_ext_orig, prec_ext_orig, color=C_EXT_LINE, lw=1.2, ls='--',
        label=f'Original (AP = {ext_ap:.3f})')
ax.axhline(y=prevalence_ext, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_ext:.3f}')
ax.set_title('External Validation', fontsize=10)
ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision (PPV)')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
sci_legend(ax, loc='lower left')

plt.tight_layout()
plt.savefig('figures/PR_CI_External.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/PR_CI_External.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/PR_CI_External.png / .pdf")
plt.close()

# ==================== 图6：PR — 合并 ====================
fig, ax = plt.subplots(figsize=FIG_SINGLE)

ax.fill_between(mean_recall, lower_prec_int, upper_prec_int, color=C_INT, alpha=0.12)
ax.plot(mean_recall, mean_prec_int, color=C_INT, lw=1.5,
        label=f'Internal (AP = {corrected_ap:.3f})')
ax.fill_between(mean_recall, lower_prec_ext, upper_prec_ext, color=C_EXT, alpha=0.12)
ax.plot(mean_recall, mean_prec_ext, color=C_EXT, lw=1.5,
        label=f'External (AP = {mean_ap_ext_boot:.3f})')
ax.plot(rec_oof_orig, prec_oof_orig, color=C_OOF_LINE, lw=1.0, ls='--',
        label=f'Internal OOF (AP = {oof_ap:.3f})')
ax.plot(rec_ext_orig, prec_ext_orig, color=C_EXT_LINE, lw=1.0, ls='--',
        label=f'External original (AP = {ext_ap:.3f})')
ax.axhline(y=prevalence_int, color=C_INT, ls=':', lw=0.8, alpha=0.5,
           label=f'Int. prevalence = {prevalence_int:.3f}')
ax.axhline(y=prevalence_ext, color=C_EXT, ls=':', lw=0.8, alpha=0.5,
           label=f'Ext. prevalence = {prevalence_ext:.3f}')

ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision (PPV)')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/PR_CI_Combined.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/PR_CI_Combined.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/PR_CI_Combined.png / .pdf")
plt.close()

# ==================== 图7：Optimism 分布（三面板）====================
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

for ax, data, metric, color in zip(
    axes,
    [optimism_auc_list, optimism_ap_list, optimism_brier_list],
    ['AUC', 'AP', 'Brier'],
    [C_INT, C_CORRECT, C_BRIER]
):
    mean_val = np.mean(data)
    ax.hist(data, bins=35, color=color, alpha=0.7, edgecolor='white', lw=0.3)
    ax.axvline(mean_val, color=C_EXT, lw=1.2, ls='--',
               label=f'Mean = {mean_val:.4f}')
    ax.axvline(0, color='#333333', lw=0.6, alpha=0.5)
    ax.set_xlabel(f'Optimism ({metric})')
    ax.set_ylabel('Frequency')
    sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/Optimism_Distribution.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/Optimism_Distribution.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/Optimism_Distribution.png / .pdf")
plt.close()

# ==================== 图8：AUC + AP 对比柱状图 ====================
fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE)

bar_colors = [C_APPARENT, C_CORRECT, C_INT, C_EXT]
labels = ['Apparent', 'Optimism-\ncorrected', 'OOF\n(5×20 CV)', 'External']

for ax, metric_name, values, optimism_val in zip(
    axes,
    ['AUC', 'AP'],
    [
        [apparent_auc, corrected_auc, oof_auc, ext_auc],
        [apparent_ap, corrected_ap, oof_ap, ext_ap]
    ],
    [mean_optimism_auc, mean_optimism_ap]
):
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom')

    # Optimism 箭头
    ax.annotate('', xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle='->', color=C_EXT, lw=1.5, ls='--'))
    mid_y = (values[0] + values[1]) / 2
    ax.text(0.5, mid_y + 0.003, f'Optimism\n= {optimism_val:.4f}',
            ha='center', va='bottom', color=C_EXT)

    y_min = min(values) - 0.05
    y_max = max(values) + 0.04
    ax.set_ylim([y_min, y_max])
    ax.set_ylabel(metric_name)
    ax.axhline(y=values[1], color=C_CORRECT, alpha=0.3, ls=':', lw=0.8)

plt.tight_layout()
plt.savefig('figures/AUC_AP_Comparison_Optimism.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/AUC_AP_Comparison_Optimism.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/AUC_AP_Comparison_Optimism.png / .pdf")
plt.close()

# ==================== 图9：AUC + AP 一致性点线图 ====================
fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE)

methods = ['Apparent', 'Optimism-\ncorrected', 'OOF CV', 'External']
dot_colors = [C_APPARENT, C_CORRECT, C_INT, C_EXT]

for ax, metric_name, values, std_corr, std_ext in zip(
    axes,
    ['AUC', 'AP'],
    [
        [apparent_auc, corrected_auc, oof_auc, ext_auc],
        [apparent_ap, corrected_ap, oof_ap, ext_ap]
    ],
    [std_optimism_auc, std_optimism_ap],
    [std_auc_ext, std_ap_ext_boot]
):
    # CI 计算
    ci_lo = [values[0], values[1] - 1.96 * std_corr,
             values[2], values[3] - 1.96 * std_ext]
    ci_hi = [values[0], values[1] + 1.96 * std_corr,
             values[2], values[3] + 1.96 * std_ext]

    # OOF CI
    if df_cv_detail is not None and metric_name in df_cv_detail.columns:
        cv_std = df_cv_detail[metric_name].std()
        ci_lo[2] = values[2] - 1.96 * cv_std
        ci_hi[2] = values[2] + 1.96 * cv_std
    else:
        oof_std = std_auc_oof_boot if metric_name == 'AUC' else np.std(boot_ap_oof)
        ci_lo[2] = values[2] - 1.96 * oof_std
        ci_hi[2] = values[2] + 1.96 * oof_std

    for j, (val, lo, hi, c) in enumerate(zip(values, ci_lo, ci_hi, dot_colors)):
        ax.errorbar(j, val, yerr=[[val - lo], [hi - val]], fmt='o', markersize=6,
                    color=c, ecolor=c, elinewidth=1.5, capsize=4, capthick=1.2,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)
        ax.text(j, hi + 0.006, f'{val:.3f}', ha='center', fontweight='bold', color=c)

    ax.plot(range(4), values, color='#AAAAAA', ls='--', lw=0.8, zorder=1)

    # 一致性范围高亮
    ax.axhspan(min(values[1:]), max(values[1:]), alpha=0.05, color=C_CORRECT)

    ax.set_xticks(range(4))
    ax.set_xticklabels(methods)
    ax.set_ylabel(metric_name)

    consistency = max(values[1:]) - min(values[1:])
    ax.text(0.97, 0.03,
            f'Δ range = {consistency:.4f}',
            transform=ax.transAxes, ha='right', va='bottom',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7',
                      edgecolor='#E0E0E0', alpha=0.9))

plt.tight_layout()
plt.savefig('figures/AUC_AP_Consistency_Overview.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/AUC_AP_Consistency_Overview.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/AUC_AP_Consistency_Overview.png / .pdf")
plt.close()

# ==================== 图10：2×2 ROC + PR 总览面板 ====================
fig, axes = plt.subplots(2, 2, figsize=FIG_PANEL)

# --- (0,0) ROC Internal ---
ax = axes[0, 0]
ax.fill_between(mean_fpr, lower_tpr_int, upper_tpr_int, color=C_INT, alpha=0.15)
ax.plot(mean_fpr, mean_tpr_int, color=C_INT, lw=1.5,
        label=f'Mean (AUC = {mean_auc_int_boot:.3f})')
ax.plot(fpr_oof_orig, tpr_oof_orig, color=C_OOF_LINE, lw=1.0, ls='--',
        label=f'OOF (AUC = {oof_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='lower right')

# --- (0,1) ROC External ---
ax = axes[0, 1]
ax.fill_between(mean_fpr, lower_tpr_ext, upper_tpr_ext, color=C_EXT, alpha=0.15)
ax.plot(mean_fpr, mean_tpr_ext, color=C_EXT, lw=1.5,
        label=f'Mean (AUC = {mean_auc_ext:.3f})')
ax.plot(fpr_ext_orig, tpr_ext_orig, color=C_EXT_LINE, lw=1.0, ls='--',
        label=f'Original (AUC = {ext_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='lower right')

# --- (1,0) PR Internal ---
ax = axes[1, 0]
ax.fill_between(mean_recall, lower_prec_int, upper_prec_int, color=C_INT, alpha=0.15)
ax.plot(mean_recall, mean_prec_int, color=C_INT, lw=1.5,
        label=f'Mean (AP = {corrected_ap:.3f})')
ax.plot(rec_oof_orig, prec_oof_orig, color=C_OOF_LINE, lw=1.0, ls='--',
        label=f'OOF (AP = {oof_ap:.3f})')
ax.axhline(y=prevalence_int, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_int:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='upper right')

# --- (1,1) PR External ---
ax = axes[1, 1]
ax.fill_between(mean_recall, lower_prec_ext, upper_prec_ext, color=C_EXT, alpha=0.15)
ax.plot(mean_recall, mean_prec_ext, color=C_EXT, lw=1.5,
        label=f'Mean (AP = {mean_ap_ext_boot:.3f})')
ax.plot(rec_ext_orig, prec_ext_orig, color=C_EXT_LINE, lw=1.0, ls='--',
        label=f'Original (AP = {ext_ap:.3f})')
ax.axhline(y=prevalence_ext, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_ext:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/ROC_PR_Panel_Overview.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_PR_Panel_Overview.pdf', bbox_inches='tight')
print("  ✓ 已保存: figures/ROC_PR_Panel_Overview.png / .pdf")
plt.close()


# ==============================================================
# 总结
# ==============================================================
print("\n" + "="*70)
print("  ✅ 全部完成！共生成 10 张图（均有 PNG + PDF）：")
print("  ─────────────────────────────────────────────")
print("  ROC 系列：")
print("    1.  figures/ROC_CI_Internal.png")
print("    2.  figures/ROC_CI_External.png")
print("    3.  figures/ROC_CI_Combined.png")
print("  PR 系列：")
print("    4.  figures/PR_CI_Internal.png")
print("    5.  figures/PR_CI_External.png")
print("    6.  figures/PR_CI_Combined.png")
print("  Optimism 分析：")
print("    7.  figures/Optimism_Distribution.png")
print("    8.  figures/AUC_AP_Comparison_Optimism.png")
print("    9.  figures/AUC_AP_Consistency_Overview.png")
print("  综合面板：")
print("    10. figures/ROC_PR_Panel_Overview.png")
print("  ─────────────────────────────────────────────")
print("  CSV 输出：")
print(f"    • {DATA_PATH}/Internal_Bootstrap_Optimism_{TIMESTAMP}.csv")
print(f"    • {DATA_PATH}/Optimism_Correction_Summary_{TIMESTAMP}.csv")
print("="*70)

# ==============================================================
# 补充计算：OOF bootstrap resample 的平均曲线
# ==============================================================

# OOF resample 的平均 ROC
mean_tpr_oof_smooth = boot_tprs_oof.mean(axis=0)
# OOF resample 的平均 PR
mean_prec_oof_smooth = boot_precs_oof.mean(axis=0)

# Apparent 曲线（用于方案 B）
fpr_apparent, tpr_apparent, _ = roc_curve(y_dev, prob_apparent)
prec_apparent, rec_apparent, _ = precision_recall_curve(y_dev, prob_apparent)


# ==============================================================
# 图10-D：2×2 面板
# ==============================================================
print("\n绘制 2×2 面板（方案 D：Harrell mean + OOF resample mean）...")

fig, axes = plt.subplots(2, 2, figsize=FIG_PANEL)

# --- (0,0) ROC Internal ---
ax = axes[0, 0]
ax.fill_between(mean_fpr, lower_tpr_int, upper_tpr_int, color=C_INT, alpha=0.15)
ax.plot(mean_fpr, mean_tpr_int, color=C_INT, lw=1.5,
        label=f'Harrell corrected (AUC = {mean_auc_int_boot:.3f})')
ax.plot(mean_fpr, mean_tpr_oof_smooth, color=C_OOF_LINE, lw=1.2, ls='--',
        label=f'OOF resampled (AUC = {mean_auc_oof_boot:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='lower right')

# --- (0,1) ROC External ---
ax = axes[0, 1]
ax.fill_between(mean_fpr, lower_tpr_ext, upper_tpr_ext, color=C_EXT, alpha=0.15)
ax.plot(mean_fpr, mean_tpr_ext, color=C_EXT, lw=1.5,
        label=f'Bootstrap mean (AUC = {mean_auc_ext:.3f})')
ax.plot(fpr_ext_orig, tpr_ext_orig, color=C_EXT_LINE, lw=1.2, ls='--',
        label=f'Original (AUC = {ext_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='lower right')

# --- (1,0) PR Internal ---
ax = axes[1, 0]
ax.fill_between(mean_recall, lower_prec_int, upper_prec_int, color=C_INT, alpha=0.15)
ax.plot(mean_recall, mean_prec_int, color=C_INT, lw=1.5,
        label=f'Harrell corrected (AP = {corrected_ap:.3f})')
ax.plot(mean_recall, mean_prec_oof_smooth, color=C_OOF_LINE, lw=1.2, ls='--',
        label=f'OOF resampled (AP = {mean_ap_oof_boot:.3f})')
ax.axhline(y=prevalence_int, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_int:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='upper right')

# --- (1,1) PR External ---
ax = axes[1, 1]
ax.fill_between(mean_recall, lower_prec_ext, upper_prec_ext, color=C_EXT, alpha=0.15)
ax.plot(mean_recall, mean_prec_ext, color=C_EXT, lw=1.5,
        label=f'Bootstrap mean (AP = {mean_ap_ext_boot:.3f})')
ax.plot(rec_ext_orig, prec_ext_orig, color=C_EXT_LINE, lw=1.2, ls='--',
        label=f'Original (AP = {ext_ap:.3f})')
ax.axhline(y=prevalence_ext, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_ext:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/ROC_PR_Panel_Overview_D.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_PR_Panel_Overview_D.pdf', bbox_inches='tight')
print("  ✓ 方案 D: figures/ROC_PR_Panel_Overview_D.png / .pdf")
plt.close()


# ==============================================================
# 图10-B：2×2 面板 
# ==============================================================
print("绘制 2×2 面板（方案 B：Apparent + Harrell mean，直观展示 optimism）...")

fig, axes = plt.subplots(2, 2, figsize=FIG_PANEL)

# --- (0,0) ROC Internal ---
ax = axes[0, 0]
ax.fill_between(mean_fpr, lower_tpr_int, upper_tpr_int, color=C_INT, alpha=0.15)
ax.plot(mean_fpr, mean_tpr_int, color=C_INT, lw=1.5,
        label=f'Corrected (AUC = {mean_auc_int_boot:.3f})')
ax.plot(fpr_apparent, tpr_apparent, color=C_APPARENT, lw=1.2, ls='--',
        label=f'Apparent (AUC = {apparent_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='lower right')

# --- (0,1) ROC External ---
ax = axes[0, 1]
ax.fill_between(mean_fpr, lower_tpr_ext, upper_tpr_ext, color=C_EXT, alpha=0.15)
ax.plot(mean_fpr, mean_tpr_ext, color=C_EXT, lw=1.5,
        label=f'Bootstrap mean (AUC = {mean_auc_ext:.3f})')
ax.plot(fpr_ext_orig, tpr_ext_orig, color=C_EXT_LINE, lw=1.2, ls='--',
        label=f'Original (AUC = {ext_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='lower right')

# --- (1,0) PR Internal ---
ax = axes[1, 0]
ax.fill_between(mean_recall, lower_prec_int, upper_prec_int, color=C_INT, alpha=0.15)
ax.plot(mean_recall, mean_prec_int, color=C_INT, lw=1.5,
        label=f'Corrected (AP = {corrected_ap:.3f})')
ax.plot(rec_apparent, prec_apparent, color=C_APPARENT, lw=1.2, ls='--',
        label=f'Apparent (AP = {apparent_ap:.3f})')
ax.axhline(y=prevalence_int, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_int:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='upper right')

# --- (1,1) PR External ---
ax = axes[1, 1]
ax.fill_between(mean_recall, lower_prec_ext, upper_prec_ext, color=C_EXT, alpha=0.15)
ax.plot(mean_recall, mean_prec_ext, color=C_EXT, lw=1.5,
        label=f'Bootstrap mean (AP = {mean_ap_ext_boot:.3f})')
ax.plot(rec_ext_orig, prec_ext_orig, color=C_EXT_LINE, lw=1.2, ls='--',
        label=f'Original (AP = {ext_ap:.3f})')
ax.axhline(y=prevalence_ext, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_ext:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/ROC_PR_Panel_Overview_B.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_PR_Panel_Overview_B.pdf', bbox_inches='tight')
print("  ✓ 方案 B: figures/ROC_PR_Panel_Overview_B.png / .pdf")
plt.close()


# ==============================================================
# 图10-Full：2×2 面板
# ==============================================================
print("绘制 2×2 面板（完整版：Apparent + Corrected + OOF resample）...")

fig, axes = plt.subplots(2, 2, figsize=FIG_PANEL)

# --- (0,0) ROC Internal ---
ax = axes[0, 0]
ax.fill_between(mean_fpr, lower_tpr_int, upper_tpr_int, color=C_INT, alpha=0.12)
ax.plot(mean_fpr, mean_tpr_int, color=C_INT, lw=1.5,
        label=f'Corrected (AUC = {mean_auc_int_boot:.3f})')
ax.plot(fpr_apparent, tpr_apparent, color=C_APPARENT, lw=1.0, ls=':',
        label=f'Apparent (AUC = {apparent_auc:.3f})')
ax.plot(mean_fpr, mean_tpr_oof_smooth, color=C_OOF_LINE, lw=1.0, ls='--',
        label=f'OOF (AUC = {mean_auc_oof_boot:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='lower right')

# --- (0,1) ROC External ---
ax = axes[0, 1]
ax.fill_between(mean_fpr, lower_tpr_ext, upper_tpr_ext, color=C_EXT, alpha=0.15)
ax.plot(mean_fpr, mean_tpr_ext, color=C_EXT, lw=1.5,
        label=f'Bootstrap mean (AUC = {mean_auc_ext:.3f})')
ax.plot(fpr_ext_orig, tpr_ext_orig, color=C_EXT_LINE, lw=1.2, ls='--',
        label=f'Original (AUC = {ext_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='lower right')

# --- (1,0) PR Internal ---
ax = axes[1, 0]
ax.fill_between(mean_recall, lower_prec_int, upper_prec_int, color=C_INT, alpha=0.12)
ax.plot(mean_recall, mean_prec_int, color=C_INT, lw=1.5,
        label=f'Corrected (AP = {corrected_ap:.3f})')
ax.plot(rec_apparent, prec_apparent, color=C_APPARENT, lw=1.0, ls=':',
        label=f'Apparent (AP = {apparent_ap:.3f})')
ax.plot(mean_recall, mean_prec_oof_smooth, color=C_OOF_LINE, lw=1.0, ls='--',
        label=f'OOF (AP = {mean_ap_oof_boot:.3f})')
ax.axhline(y=prevalence_int, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_int:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='upper right')

# --- (1,1) PR External ---
ax = axes[1, 1]
ax.fill_between(mean_recall, lower_prec_ext, upper_prec_ext, color=C_EXT, alpha=0.15)
ax.plot(mean_recall, mean_prec_ext, color=C_EXT, lw=1.5,
        label=f'Bootstrap mean (AP = {mean_ap_ext_boot:.3f})')
ax.plot(rec_ext_orig, prec_ext_orig, color=C_EXT_LINE, lw=1.2, ls='--',
        label=f'Original (AP = {ext_ap:.3f})')
ax.axhline(y=prevalence_ext, color='#999999', ls=':', lw=0.8,
           label=f'Prevalence = {prevalence_ext:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontweight='bold', fontsize=12)
sci_legend(ax, loc='upper right')

plt.tight_layout()
plt.savefig('figures/ROC_PR_Panel_Overview_Full.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/ROC_PR_Panel_Overview_Full.pdf', bbox_inches='tight')
print("  ✓ 完整版: figures/ROC_PR_Panel_Overview_Full.png / .pdf")
plt.close()

print("\n  ✅ 三个方案面板均已生成，请对比选择：")
print("    • _D.png    → Harrell mean + OOF resample mean（推荐，两条都平滑）")
print("    • _B.png    → Apparent + Harrell mean（直观展示 optimism 差距）")
print("    • _Full.png → 三条线完整版（信息最丰富，适合补充材料）")

# ------------------------------------------
# 4. 校准曲线置信带（基于 Bootstrap）— 校准后版本
# ------------------------------------------
print("\n[额外-4] 绘制校准曲线置信带（校准后 + 前后对比）...")

# 计算所有 Bootstrap 的校准曲线
n_bins = 10
calib_curves = []

for boot_pred in bootstrap_predictions[:100]:  # 使用前100个以加快速度
    y_true_boot = boot_pred['true_labels']
    y_pred_boot = boot_pred['pred_probs']
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_boot, y_pred_boot, n_bins=n_bins, strategy='quantile'
    )
    calib_curves.append((mean_predicted_value, fraction_of_positives))

# 插值到统一网格
pred_grid = np.linspace(0, 1, 100)
interp_fractions = []

for mean_pred, frac_pos in calib_curves:
    if len(mean_pred) > 1:
        f = interpolate.interp1d(mean_pred, frac_pos, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
        interp_fractions.append(f(pred_grid))

# --- 图4a: 校准后版本 ---
fig, ax = plt.subplots(figsize=(10, 10))

if interp_fractions:
    interp_fractions_arr = np.array(interp_fractions)
    mean_frac = np.nanmean(interp_fractions_arr, axis=0)
    std_frac = np.nanstd(interp_fractions_arr, axis=0)
    
    upper = np.minimum(mean_frac + 1.96 * std_frac, 1)
    lower = np.maximum(mean_frac - 1.96 * std_frac, 0)
    
    ax.fill_between(pred_grid, lower, upper, color=COLOR_EXTERNAL, 
                    alpha=0.2, label='95% CI')
    ax.plot(pred_grid, mean_frac, color=COLOR_EXTERNAL, linewidth=2.5,
            label='Mean calibration')

# 校准后外部验证校准曲线
fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
    y_ext, prob_ext_cal, n_bins=n_bins, strategy='quantile'
)
ax.plot(mean_predicted_value_cal, fraction_of_positives_cal, 'o-', 
        color='darkblue', linewidth=2, markersize=8,
        label='External (calibrated)')

ax.plot([0, 1], [0, 1], 'k:', linewidth=2, label='Ideal')

ax.set_xlabel('Predicted probability')
ax.set_ylabel('Observed proportion')
ax.set_title('Calibration (calibrated)', 
            fontsize=13, pad=15)
ax.legend(loc='upper left')
ax.grid(alpha=0.15, linewidth=0.5)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('figures/Calibration_Confidence_Band_Calibrated.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/Calibration_Confidence_Band_Calibrated.png")
plt.close()

# --- 图4b: 校准前后对比版本 ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax_idx, (prob_ver, title_label) in enumerate([
    (prob_ext, 'Before Calibration (Original)'),
    (prob_ext_cal, 'After calibration')
]):
    ax = axes[ax_idx]
    
    if interp_fractions:
        ax.fill_between(pred_grid, lower, upper, color=COLOR_EXTERNAL, 
                        alpha=0.2, label='95% CI')
        ax.plot(pred_grid, mean_frac, color=COLOR_EXTERNAL, linewidth=2.5,
                label='Mean calibration')
    
    frac_pos_ver, mean_pred_ver = calibration_curve(
        y_ext, prob_ver, n_bins=n_bins, strategy='quantile'
    )
    ax.plot(mean_pred_ver, frac_pos_ver, 'o-', 
            color='darkred' if ax_idx == 0 else 'darkblue', linewidth=2, markersize=8,
            label=f'External Calibration')
    
    ax.plot([0, 1], [0, 1], 'k:', linewidth=2, label='Ideal')
    
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed proportion')
    ax.set_title(title_label, pad=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')

plt.suptitle('Calibration: before vs. after')
plt.tight_layout()
plt.savefig('figures/Calibration_Confidence_Band_BeforeAfter.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/Calibration_Confidence_Band_BeforeAfter.png")
plt.close()


# ------------------------------------------
# 5. 阈值-性能曲线（Threshold Performance Curve）
# ------------------------------------------
print("\n[额外-5] 绘制阈值-性能曲线...")

thresholds = np.linspace(0, 1, 100)
metrics_at_thresh = {
    'Sensitivity': [],
    'Specificity': [],
    'PPV': [],
    'NPV': [],
    'F1_Score': [],
    'Accuracy': []
}

for thresh in thresholds:
    y_pred = (prob_ext_cal >= thresh).astype(int)
    
    tn = np.sum((y_ext == 0) & (y_pred == 0))
    tp = np.sum((y_ext == 1) & (y_pred == 1))
    fn = np.sum((y_ext == 1) & (y_pred == 0))
    fp = np.sum((y_ext == 0) & (y_pred == 1))
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    metrics_at_thresh['Sensitivity'].append(sens)
    metrics_at_thresh['Specificity'].append(spec)
    metrics_at_thresh['PPV'].append(ppv)
    metrics_at_thresh['NPV'].append(npv)
    metrics_at_thresh['F1_Score'].append(f1)
    metrics_at_thresh['Accuracy'].append(acc)

fig, ax = plt.subplots(figsize=(12, 8))

colors_thresh = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for (metric, values), color in zip(metrics_at_thresh.items(), colors_thresh):
    ax.plot(thresholds, values, linewidth=2.5, label=metric, color=color)

# 标记校准后最优阈值
ax.axvline(clinical_threshold_cal_ext, color='red', linestyle='--', linewidth=2,
          label=f'Calibrated Threshold: {clinical_threshold_cal_ext:.4f}')

ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Metric Value')
ax.set_title('Performance Metrics vs Classification Threshold\nExternal Validation (calibrated)', 
            fontsize=13, pad=15)
ax.legend(loc='best', fontsize=11, ncol=2)
ax.grid(alpha=0.15, linewidth=0.5)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('figures/Threshold_Performance_Curve.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/Threshold_Performance_Curve.png")
plt.close()


# ------------------------------------------
# 6. 预测概率分布图（按真实标签分层）
# ------------------------------------------
print("\n[额外-6] 绘制预测概率分布图 (校准后)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

datasets_prob = [
    (y_int, prob_int_cal, 'Internal (calibrated)', COLOR_INTERNAL, clinical_threshold_cal_int),
    (y_ext, prob_ext_cal, 'External (calibrated)', COLOR_EXTERNAL, clinical_threshold_cal_ext)
]

# 定义 SCI 高级配色
col_non_rec = '#4A7BB7'  # 沉稳科研蓝
col_rec = '#C44E52'      # 经典砖红色
col_thresh = '#595959'   # 碳灰色（用于阈值线）

for idx, (y_true, y_pred, title, color, thresh_cal) in enumerate(datasets_prob):
    # ====================
    # 1. 直方图 (Histogram)
    # ====================
    ax1 = axes[idx, 0]
    
    # 调整 alpha，将 edgecolor 改为 white，增加柱子层次感
    ax1.hist(y_pred[y_true == 0], bins=30, alpha=0.75, color=col_non_rec, 
             label='Non-recurrence', edgecolor='white', linewidth=0.8, density=True)
    ax1.hist(y_pred[y_true == 1], bins=30, alpha=0.75, color=col_rec, 
             label='Recurrence', edgecolor='white', linewidth=0.8, density=True)
    
    # 阈值线：使用校准后阈值
    ax1.axvline(thresh_cal, color=col_thresh, linestyle='--', linewidth=1.5,
               label=f'Threshold: {thresh_cal:.3f}')
    
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title}\nProbability Distribution')
    
    # 图例去边框，网格线极度弱化
    ax1.legend(frameon=True, edgecolor='#999999', fontsize=10)
    ax1.grid(alpha=0.2, linestyle=':')
    
    # 顶刊必备：去掉右侧和顶部的边框线
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ====================
    # 2. 箱线图 (Boxplot)
    # ====================
    ax2 = axes[idx, 1]
    
    # 自定义 boxplot 的各个部件样式
    boxprops = dict(linewidth=1.2, alpha=0.8)
    medianprops = dict(linewidth=1.5, color='black') # 中位数线用纯黑，对比鲜明
    whiskerprops = dict(linewidth=1.2, color='#333333')
    capprops = dict(linewidth=1.2, color='#333333')
    flierprops = dict(marker='o', markersize=4, alpha=0.4, markeredgecolor='none')

    bp = ax2.boxplot([y_pred[y_true == 0], y_pred[y_true == 1]], 
                     labels=['Non-recurrence', 'Recurrence'],
                     patch_artist=True, widths=0.5,
                     boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops,
                     flierprops=flierprops)
    
    # 给箱子填充颜色
    colors = [col_non_rec, col_rec]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#333333') # 箱体边框用深灰
        
    # 给异常值点 (fliers) 也配上对应的颜色
    for flier, color in zip(bp['fliers'], colors):
        flier.set_markerfacecolor(color)
    
    ax2.axhline(thresh_cal, color=col_thresh, linestyle='--', linewidth=1.5,
               label=f'Threshold: {thresh_cal:.3f}')
               
    ax2.set_ylabel('Predicted probability')
    ax2.set_title(f'{title}\nProbability by Outcome')
    
    ax2.legend(frameon=True, edgecolor='#999999', fontsize=10, loc='best')
    ax2.grid(alpha=0.2, linestyle=':', axis='y')
    
    # 同样去掉多余边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/Prediction_Probability_Distribution.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/Prediction_Probability_Distribution.png")
plt.close()



# ------------------------------------------
# 8. 学习曲线（Learning Curve）
# ------------------------------------------
print("\n[额外-8] 绘制学习曲线...")
# 数据已在 Section 2.1 加载 (X_train, y_train, X_ext)
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    cv=5, scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1, random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(train_sizes, train_mean, 'o-', color=COLOR_INTERNAL, linewidth=2.5,
        markersize=8, label='Training')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.2, color=COLOR_INTERNAL)

ax.plot(train_sizes, val_mean, 's-', color=COLOR_EXTERNAL, linewidth=2.5,
        markersize=8, label='Validation (CV)')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.2, color=COLOR_EXTERNAL)

ax.set_xlabel('Training cohort size')
ax.set_ylabel('AUC Score')
ax.set_title(f'Learning Curve', pad=10)
ax.legend(loc='lower right')
ax.grid(alpha=0.15, linewidth=0.5)
ax.set_ylim([0.6, 1.05])

plt.tight_layout()
plt.savefig('figures/Learning_Curve.png', dpi=DPI, bbox_inches='tight')
print("  ✓ 已保存: figures/Learning_Curve.png")
plt.close()


# ------------------------------------------
# 9. 特征相关性热图
# ------------------------------------------
print("\n[额外-9] 绘制特征相关性热图...")

if 'importance_df' in locals():
    top_features = importance_df['Feature'].head(10).tolist()
    X_top_features = X_train_df[top_features]
    
    correlation_matrix = X_top_features.corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 自定义 mask：只显示下三角（含对角线）
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(correlation_matrix, 
                annot=True, fmt='.2f', 
                annot_kws={'size': 13},
                mask=mask,
                cmap='RdBu_r', center=0, 
                square=True, linewidths=1.5, linecolor='white',
                cbar_kws={"shrink": 0.75, "label": "Pearson r"},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation', 
                fontsize=14, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('figures/Feature_Correlation_Heatmap.png', dpi=DPI, bbox_inches='tight')
    print("  ✓ 已保存: figures/Feature_Correlation_Heatmap.png")
    plt.close()



print(f"\n{'='*60}")
print("✅ 所有额外可视化分析完成！")
print(f"{'='*60}\n")


# ============================================================================
# 第三部分：Bootstrap 稳定性分析可视化
# ============================================================================

# (Using global SCI style set at top of script)

# ------------------------------------------
# 3.1 Bootstrap 稳定性分析 - 指标分布可视化
# ------------------------------------------
print("\n[可视化 1/6] 绘制 Bootstrap 指标分布图...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'Bootstrap Stability (n = 1,000)', 
             fontsize=14, y=0.98)

metrics_to_plot = ['AUC', 'Brier', 'AP', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Balanced_Acc']

# 如果没有 Balanced_Acc，添加它
if 'Balanced_Acc' not in df_bootstrap_raw.columns:
    df_bootstrap_raw['Balanced_Acc'] = (df_bootstrap_raw['Sensitivity'] + df_bootstrap_raw['Specificity']) / 2

if 'Balanced_Acc' not in df_bootstrap_summary.index:
    print("正在补全 Balanced_Acc 的统计信息...")
    # 获取刚刚算出来的这一列数据
    ba_data = df_bootstrap_raw['Balanced_Acc']
    
    # 计算统计量
    ba_mean = ba_data.mean()
    ba_lower = np.percentile(ba_data, 2.5)  # 2.5% 分位数
    ba_upper = np.percentile(ba_data, 97.5) # 97.5% 分位数

    df_bootstrap_summary.loc['Balanced_Acc', 'Mean'] = ba_mean
    df_bootstrap_summary.loc['Balanced_Acc', '95%_CI_Lower'] = ba_lower
    df_bootstrap_summary.loc['Balanced_Acc', '95%_CI_Upper'] = ba_upper
for idx, metric in enumerate(metrics_to_plot):
    ax = axes.flatten()[idx]
    
    # 绘制小提琴图
    parts = ax.violinplot([df_bootstrap_raw[metric]], positions=[0], widths=0.7,
                          showmeans=True, showmedians=True)
    
    # 自定义颜色
    for pc in parts['bodies']:
        pc.set_facecolor('#4ECDC4')
        pc.set_alpha(0.7)
    
    # 添加散点（抖动显示）
    y_jitter = df_bootstrap_raw[metric] + np.random.normal(0, 0.01, len(df_bootstrap_raw))
    ax.scatter(np.zeros(len(df_bootstrap_raw)), y_jitter, 
              alpha=0.3, s=10, c='gray', edgecolors='none')
    
    # 标注统计信息
    mean_val = df_bootstrap_summary.loc[metric, 'Mean']
    ci_lower = df_bootstrap_summary.loc[metric, '95%_CI_Lower']
    ci_upper = df_bootstrap_summary.loc[metric, '95%_CI_Upper']
    
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Mean')
    ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='orange', label='95% CI')
    
    # 设置标题和标签
    ax.set_title(f'{metric}\n{mean_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})', 
                fontsize=10)
    ax.set_ylabel('Value')
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.15, linewidth=0.5)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('figures/Bootstrap_Metrics_Distribution.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: Bootstrap_Metrics_Distribution.png")
plt.close()

# ------------------------------------------
# 3.2 Bootstrap ROC 曲线置信带
# ------------------------------------------
print("\n[可视化 2/6] 绘制 Bootstrap ROC 曲线置信带...")

fig, ax = plt.subplots(figsize=(8, 8))

# 计算每次 Bootstrap 的 ROC 曲线
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for boot_pred in bootstrap_predictions:
    y_true = boot_pred['true_labels']
    y_score = boot_pred['pred_probs']
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # 插值到统一的 FPR 网格
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)

# 计算均值和标准差
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)

# 绘制置信区间
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightblue', alpha=0.4,
                label=f'95% CI (Bootstrap n=1000)')

# 绘制平均 ROC 曲线
ax.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
        label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

# 绘制对角线
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Reference')

# 美化
ax.set_xlabel('1 - Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title(f'Bootstrap ROC with 95% CI',
            fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.15, linewidth=0.5)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('figures/Bootstrap_ROC_Confidence_Band.png', dpi=DPI, bbox_inches='tight')
print(f"  ✓ 已保存: Bootstrap_ROC_Confidence_Band.png")
plt.close()


"""
===================================================================================
风险分层 & NRI 可视化脚本
===================================================================================
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, fisher_exact
from scipy.special import expit as _expit
from itertools import combinations

DATA_PATH = 'model_results'
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTAMP = '20260216_1724'

# ============================================================
# 辅助函数
# ============================================================

def get_train_source_label(pkg):
    meta = pkg.get('metadata', {})
    source = meta.get('train_data_source', 'unknown')
    if source == 'OOF':
        return 'Internal'
    elif source == 'resubstitution':
        return 'Internal (Resub)'
    else:
        return 'Internal'


def _sig_mark(p):
    if p is None or np.isnan(p):
        return 'N/A'
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return 'NS'


def _safe_logit(p, eps=1e-7):
    """安全 logit 变换"""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def compute_fixed_risk_groups(y_true, y_prob, low_cut=0.05, high_cut=0.15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    group_masks = [
        y_prob < low_cut,
        (y_prob >= low_cut) & (y_prob < high_cut),
        y_prob >= high_cut,
    ]
    labels = [
        f'Low (<{low_cut:.0%})',
        f'Medium ({low_cut:.0%}\u2013{high_cut:.0%})',
        f'High (\u2265{high_cut:.0%})'
    ]

    stats = {}
    contingency = []

    for g, (mask, label) in enumerate(zip(group_masks, labels)):
        n = int(mask.sum())
        events = int(y_true[mask].sum()) if n > 0 else 0
        non_events = n - events
        event_rate = events / n if n > 0 else 0.0
        mean_prob = float(y_prob[mask].mean()) if n > 0 else 0.0
        stats[g] = {
            'label': label, 'n': n,
            'events': events, 'non_events': non_events,
            'event_rate': event_rate, 'mean_prob': mean_prob,
        }
        contingency.append([events, non_events])

    contingency_table = np.array(contingency)

    try:
        chi2_stat, overall_p, dof, expected = chi2_contingency(contingency_table)
        min_expected = expected.min()
    except Exception:
        chi2_stat, overall_p, dof, min_expected = np.nan, np.nan, np.nan, 0

    pairwise_results = []
    for (i, j) in combinations([0, 1, 2], 2):
        sub_table = contingency_table[[i, j], :]
        if sub_table.sum(axis=1).min() == 0:
            pairwise_results.append({
                'comparison': f'{labels[i]} vs {labels[j]}',
                'group_i': i, 'group_j': j,
                'p_raw': np.nan, 'p_corrected': np.nan,
                'method': 'N/A (empty group)'
            })
            continue
        if sub_table.min() < 5 or sub_table.sum() < 40:
            try:
                _, p_raw = fisher_exact(sub_table)
                method = 'Fisher'
            except Exception:
                _, p_raw, _, _ = chi2_contingency(sub_table, correction=True)
                method = 'Chi2(Yates)'
        else:
            _, p_raw, _, _ = chi2_contingency(sub_table, correction=False)
            method = 'Chi2'
        pairwise_results.append({
            'comparison': f'{labels[i]} vs {labels[j]}',
            'group_i': i, 'group_j': j,
            'p_raw': p_raw,
            'p_corrected': min(p_raw * 3, 1.0),
            'method': method
        })

    return {
        'labels': labels, 'stats': stats,
        'cutpoints': [low_cut, high_cut],
        'contingency_table': contingency_table,
        'overall_chi2': chi2_stat, 'overall_p': overall_p,
        'overall_dof': dof, 'min_expected': min_expected,
        'pairwise': pairwise_results,
    }


def compute_tertile_risk_groups(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    q33 = np.percentile(y_prob, 33.33)
    q67 = np.percentile(y_prob, 66.67)
    group_masks = [
        y_prob < q33,
        (y_prob >= q33) & (y_prob < q67),
        y_prob >= q67,
    ]
    labels = [
        f'Low (<{q33:.3f})',
        f'Medium ({q33:.3f}\u2013{q67:.3f})',
        f'High (\u2265{q67:.3f})'
    ]
    stats = {}
    for g, (mask, label) in enumerate(zip(group_masks, labels)):
        n = int(mask.sum())
        events = int(y_true[mask].sum()) if n > 0 else 0
        stats[g] = {
            'label': label, 'n': n, 'events': events,
            'non_events': n - events,
            'event_rate': events / n if n > 0 else 0.0,
            'mean_prob': float(y_prob[mask].mean()) if n > 0 else 0.0,
        }
    return {'labels': labels, 'stats': stats, 'quantiles': [q33, q67]}


def save_risk_stratification_csv(fixed_result, model_name, ds_name, output_dir):
    rows = []
    for g in range(3):
        s = fixed_result['stats'][g]
        rows.append({
            'Model': model_name, 'Dataset': ds_name,
            'Risk_Group': s['label'], 'N': s['n'],
            'Events': s['events'], 'Non_Events': s['non_events'],
            'Event_Rate': f"{s['event_rate']:.4f}",
            'Event_Rate_Pct': f"{s['event_rate']:.1%}",
            'Mean_Predicted_Prob': f"{s['mean_prob']:.4f}",
        })
    df_groups = pd.DataFrame(rows)

    comparison_rows = []
    comparison_rows.append({
        'Comparison': 'Overall (3-group)', 'Method': 'Chi-square',
        'P_Value_Raw': f"{fixed_result['overall_p']:.6f}" if not np.isnan(fixed_result['overall_p']) else 'N/A',
        'P_Value_Corrected': '\u2014',
        'Significance': _sig_mark(fixed_result['overall_p']),
    })
    for pw in fixed_result['pairwise']:
        comparison_rows.append({
            'Comparison': pw['comparison'], 'Method': pw['method'],
            'P_Value_Raw': f"{pw['p_raw']:.6f}" if not np.isnan(pw['p_raw']) else 'N/A',
            'P_Value_Corrected': f"{pw['p_corrected']:.6f}" if not np.isnan(pw['p_corrected']) else 'N/A',
            'Significance': _sig_mark(pw['p_corrected']),
        })
    df_comparisons = pd.DataFrame(comparison_rows)

    csv_path = f'{output_dir}/Risk_Stratification_Stats_{model_name}_{ds_name}.csv'
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        f.write(f"# Risk Stratification Statistics\n")
        f.write(f"# Model: {model_name}, Dataset: {ds_name}\n")
        f.write(f"# Cutpoints (calibrated prob): {fixed_result['cutpoints']}\n\n")
        f.write("## Group Statistics\n")
        df_groups.to_csv(f, index=False)
        f.write(f"\n## Between-Group Comparisons (Bonferroni corrected)\n")
        df_comparisons.to_csv(f, index=False)
    print(f"  \u2713 风险分层统计表已保存: {csv_path}")
    return df_groups, df_comparisons


# ============================================================
# 单数据集绘图函数
# ============================================================
def plot_risk_stratification(risk_data, model_name, ds_name, output_dir,
                             delta_b, low_cut=0.05, high_cut=0.15):
    """
    绘制单个数据集的风险分层三合一图 (与原版一致)
    """
    y_true = np.asarray(risk_data['y_true'])
    y_prob_orig = np.asarray(risk_data['y_prob'])
    y_prob_cal = _expit(_safe_logit(y_prob_orig) + delta_b)

    print(f"  [{ds_name}] 应用 \u0394b = {delta_b:.4f}, "
          f"Mean prob: {y_prob_orig.mean():.4f} \u2192 {y_prob_cal.mean():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{ds_name} (Calibrated)', fontsize=14)

    _draw_three_subplots(axes, y_true, y_prob_cal, ds_name, low_cut, high_cut)

    plt.tight_layout()
    out_path = f'{output_dir}/Risk_Stratification_{model_name}_{ds_name}.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 风险分层图已保存: {out_path}")

    fixed = compute_fixed_risk_groups(y_true, y_prob_cal, low_cut, high_cut)
    save_risk_stratification_csv(fixed, model_name, ds_name, output_dir)
    return fixed


# ============================================================
# 核心绘图子函数：在给定的 3 个 axes 上绘制三子图 
# ============================================================
def _draw_three_subplots(axes, y_true, y_prob_cal, ds_name,
                         low_cut=0.05, high_cut=0.15,
                         panel_prefix=''):
    """
    在给定的 3 个 axes 上绘制:
      axes[0]: 三分位风险分层柱状图
      axes[1]: 预测概率分布直方图
      axes[2]: 固定切点三级分层

    完全遵循全局 SCI 样式，无冗余设置。
    """
    # --- 确定 panel 标签 ---
    if panel_prefix:
        lbl_1 = f'{panel_prefix}1'
        lbl_2 = f'{panel_prefix}2'
        lbl_3 = f'{panel_prefix}3'
    else:
        lbl_1, lbl_2, lbl_3 = 'A', 'B', 'C'

    # ================================================================
    # 子图 1: 三分位风险分层
    # ================================================================
    ax = axes[0]
    tertile = compute_tertile_risk_groups(y_true, y_prob_cal)

    labels_t     = tertile['labels']
    event_rates  = [tertile['stats'][g]['event_rate'] for g in range(3)]
    mean_probs   = [tertile['stats'][g]['mean_prob']  for g in range(3)]
    ns_t         = [tertile['stats'][g]['n']           for g in range(3)]

    x_pos = np.arange(len(labels_t))
    width = 0.35
    bars1 = ax.bar(x_pos - width / 2, event_rates, width,
                   label='Observed Event Rate', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, mean_probs, width,
                   label='Mean Predicted Prob', color='#1f77b4', alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(labels_t, ns_t)])
    ax.set_ylabel('Proportion')
    ax.set_title(f'{ds_name}: Risk Tertiles', fontsize=10)  # 无加粗
    sci_legend(ax, loc='upper left')
    ax.set_ylim(0, 1)

    for bar, val in zip(bars1, event_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=7)

    # ================================================================
    # 子图 2: 概率分布直方图
    # ================================================================
    ax = axes[1]
    ax.hist(y_prob_cal[y_true == 0], bins=20, alpha=0.6, color='#2ca02c',
            label='Non-event', density=True)
    ax.hist(y_prob_cal[y_true == 1], bins=20, alpha=0.6, color='#d62728',
            label='Event', density=True)

    ax.axvline(low_cut, color='#E08214', linestyle='--', linewidth=1.5,
               label=f'Low cut = {low_cut:.0%}')
    ax.axvline(high_cut, color='#B2182B', linestyle='--', linewidth=1.5,
               label=f'High cut = {high_cut:.0%}')

    ax.axvspan(0, low_cut, alpha=0.06, color='green', zorder=0)
    ax.axvspan(low_cut, high_cut, alpha=0.06, color='orange', zorder=0)
    ax.axvspan(high_cut, 1, alpha=0.06, color='red', zorder=0)

    ax.set_xlabel('Predicted probability (calibrated)')
    ax.set_ylabel('Density')
    ax.set_title(f'{ds_name}: Probability Distribution', fontsize=10)
    sci_legend(ax)

    # ================================================================
    # 子图 3: 固定切点三级分层
    # ================================================================
    ax = axes[2]
    fixed = compute_fixed_risk_groups(y_true, y_prob_cal, low_cut=low_cut, high_cut=high_cut)

    labels_f        = fixed['labels']
    events_list     = [fixed['stats'][g]['events']     for g in range(3)]
    non_events_list = [fixed['stats'][g]['non_events'] for g in range(3)]
    event_rates_f   = [fixed['stats'][g]['event_rate'] for g in range(3)]
    ns_f            = [fixed['stats'][g]['n']           for g in range(3)]

    y_pos = np.arange(3)
    ax.barh(y_pos, non_events_list, color='#aec7e8',
            label='Non-event', edgecolor='white', linewidth=0.5)
    ax.barh(y_pos, events_list, left=non_events_list,
            color='#d62728', label='Event', edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{l}\n(n={n})' for l, n in zip(labels_f, ns_f)])
    ax.set_xlabel('Number of Patients')
    ax.set_title(f'{ds_name}: Risk Groups (cuts: {low_cut:.0%}, {high_cut:.0%})',
                 fontsize=10)
    sci_legend(ax)

    for i, (ne, ev, er) in enumerate(zip(non_events_list, events_list, event_rates_f)):
        total = ne + ev
        ax.text(total + max(total * 0.02, 1), i,
                f'Event rate: {er:.1%}', va='center', fontsize=7, color='#333333')

    p_overall = fixed['overall_p']
    p_text = (f'Overall \u03c7\u00b2 p = '
              f'{"<0.001" if p_overall < 0.001 else f"{p_overall:.4f}"}'
              if not np.isnan(p_overall) else 'Overall p: N/A')
    ax.text(0.98, 0.02, p_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7, fontstyle='italic',
            color='#555555')

    return fixed


# ============================================================
# Combined Panel 绘图函数: Internal + External → 2×3
# ============================================================
def plot_risk_stratification_combined(risk_data_internal, risk_data_external,
                                      model_name, output_dir,
                                      delta_b_int, delta_b_ext,
                                      low_cut=0.05, high_cut=0.15,
                                      model_display_name=None):
    """
    将同一模型的 Internal 和 External 风险分层拼为 2×3 Combined Panel。

    布局:
      上排 (Row 1): Internal Validation  → A1, A2, A3
      下排 (Row 2): External Validation  → B1, B2, B3
    """
    if model_display_name is None:
        model_display_name = model_name.replace('_', ' ')

    # --- 校准 ---
    y_true_int = np.asarray(risk_data_internal['y_true'])
    y_prob_int_orig = np.asarray(risk_data_internal['y_prob'])
    y_prob_int_cal = _expit(_safe_logit(y_prob_int_orig) + delta_b_int)

    y_true_ext = np.asarray(risk_data_external['y_true'])
    y_prob_ext_orig = np.asarray(risk_data_external['y_prob'])
    y_prob_ext_cal = _expit(_safe_logit(y_prob_ext_orig) + delta_b_ext)

    print(f"  [Combined] {model_display_name}")
    print(f"    Internal: Δb = {delta_b_int:.4f}, "
          f"Mean prob: {y_prob_int_orig.mean():.4f} → {y_prob_int_cal.mean():.4f}")
    print(f"    External: Δb = {delta_b_ext:.4f}, "
          f"Mean prob: {y_prob_ext_orig.mean():.4f} → {y_prob_ext_cal.mean():.4f}")

    # --- 创建 2×3 画布 ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    fig.suptitle(f'{model_display_name} — Risk Stratification (Calibrated)',
                 fontsize=12, y=0.98)  # 无加粗，字号缩小

    # --- 上排: Internal (A, B, C) ---
    _draw_three_subplots(axes[0, :], y_true_int, y_prob_int_cal,
                         ds_name='Internal Validation',
                         low_cut=low_cut, high_cut=high_cut,
                         panel_prefix='A')

    # --- 下排: External (D, E, F) ---
    fixed_ext = _draw_three_subplots(axes[1, :], y_true_ext, y_prob_ext_cal,
                                      ds_name='External Validation',
                                      low_cut=low_cut, high_cut=high_cut,
                                      panel_prefix='B')

    # # --- 添加行标签 (左侧大字) ---
    # fig.text(0.01, 0.72, 'Internal\nValidation',
    #          fontsize=10, va='center', ha='left',
    #          rotation=90, color= COLOR_INTERNAL)
    # fig.text(0.01, 0.28, 'External\nValidation',
    #          fontsize=10, va='center', ha='left',
    #          rotation=90, color= COLOR_EXTERNAL)

    # --- 保存 ---
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    out_path = f'{output_dir}/Risk_Stratification_{model_name}_Combined.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 Combined 风险分层图已保存: {out_path}")

    # --- 同时保存 CSV ---
    fixed_int = compute_fixed_risk_groups(y_true_int, y_prob_int_cal, low_cut, high_cut)
    fixed_ext = compute_fixed_risk_groups(y_true_ext, y_prob_ext_cal, low_cut, high_cut)
    save_risk_stratification_csv(fixed_int, model_name, 'Internal', output_dir)
    save_risk_stratification_csv(fixed_ext, model_name, 'External', output_dir)

    return fixed_int, fixed_ext


# ============================================================
#  简化版 Combined Panel (仅固定切点分层, 1×2 布局)
# ============================================================
def plot_risk_stratification_combined_simple(risk_data_internal, risk_data_external,
                                             model_name, output_dir,
                                             delta_b_int, delta_b_ext,
                                             low_cut=0.05, high_cut=0.15,
                                             model_display_name=None):
    """
    简化版: 仅展示固定切点三级分层 (子图3)，Internal + External → 1×2
    适合正文 Figure 使用（如 Figure 4A/B）
    """
    if model_display_name is None:
        model_display_name = model_name.replace('_', ' ')

    y_true_int = np.asarray(risk_data_internal['y_true'])
    y_prob_int_cal = _expit(_safe_logit(np.asarray(risk_data_internal['y_prob'])) + delta_b_int)

    y_true_ext = np.asarray(risk_data_external['y_true'])
    y_prob_ext_cal = _expit(_safe_logit(np.asarray(risk_data_external['y_prob'])) + delta_b_ext)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_display_name} — Risk Group Composition (Calibrated)',
                 fontsize=12)

    for idx, (ax, y_true, y_prob_cal, ds_name) in enumerate([
        (axes[0], y_true_int, y_prob_int_cal, 'Internal Validation'),
        (axes[1], y_true_ext, y_prob_ext_cal, 'External Validation'),
    ]):
        fixed = compute_fixed_risk_groups(y_true, y_prob_cal, low_cut, high_cut)

        labels_f        = fixed['labels']
        events_list     = [fixed['stats'][g]['events']     for g in range(3)]
        non_events_list = [fixed['stats'][g]['non_events'] for g in range(3)]
        event_rates_f   = [fixed['stats'][g]['event_rate'] for g in range(3)]
        ns_f            = [fixed['stats'][g]['n']           for g in range(3)]

        y_pos = np.arange(3)
        ax.barh(y_pos, non_events_list, color='#aec7e8',
                label='Non-event', edgecolor='white', linewidth=0.5)
        ax.barh(y_pos, events_list, left=non_events_list,
                color='#d62728', label='Event', edgecolor='white', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{l}\n(n={n})' for l, n in zip(labels_f, ns_f)])
        ax.set_xlabel('Number of Patients')
        ax.set_title(f'{ds_name}', fontsize=10)  # 无加粗
        sci_legend(ax)

        for i, (ne, ev, er) in enumerate(zip(non_events_list, events_list, event_rates_f)):
            total = ne + ev
            ax.text(total + max(total * 0.02, 1), i,
                    f'Event rate: {er:.1%}', va='center', fontsize=8, color='#333333')

        p_overall = fixed['overall_p']
        p_text = (f'Overall \u03c7\u00b2 p = '
                  f'{"<0.001" if p_overall < 0.001 else f"{p_overall:.4f}"}'
                  if not np.isnan(p_overall) else 'Overall p: N/A')
        ax.text(0.98, 0.02, p_text, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, fontstyle='italic', color='#555555')

    plt.tight_layout()
    out_path = f'{output_dir}/Risk_Stratification_{model_name}_Combined_Simple.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 简化版 Combined 图已保存: {out_path}")

# ============================================================
# NRI 相关函数
# ============================================================
def plot_nri_forest(ax, nri_pkg, train_label):
    """绘制 NRI / IDI 森林图 (与原版一致)"""
    metrics_data = []
    for ds_name, ds_key in [(train_label, 'train'), ('External', 'external')]:
        nri_res = nri_pkg['nri_results'].get(ds_key)
        if nri_res is None:
            continue
        for metric_name in ['cfNRI', 'IDI']:
            if metric_name in nri_res:
                m = nri_res[metric_name]
                metrics_data.append({
                    'Dataset': ds_name, 'Metric': metric_name,
                    'Value': m['value'],
                    'CI_low': m['95CI'][0], 'CI_high': m['95CI'][1],
                    'p_value': m['p_value']
                })
        if 'categorical_NRI' in nri_res:
            m = nri_res['categorical_NRI']
            metrics_data.append({
                'Dataset': ds_name, 'Metric': 'Cat. NRI',
                'Value': m['value'],
                'CI_low': m['95CI'][0], 'CI_high': m['95CI'][1],
                'p_value': m['p_value']
            })

    if metrics_data:
        for i, d in enumerate(metrics_data):
            color = '#1f77b4' if 'Internal' in d['Dataset'] else '#d62728'
            ax.errorbar(d['Value'], i,
                        xerr=[[d['Value'] - d['CI_low']], [d['CI_high'] - d['Value']]],
                        fmt='o', color=color, capsize=5, markersize=8)
            sig = ('***' if d['p_value'] < 0.001 else
                   '**'  if d['p_value'] < 0.01  else
                   '*'   if d['p_value'] < 0.05  else 'ns')
            ax.text(d['CI_high'] + 0.02, i,
                    f"p={d['p_value']:.3f} {sig}", va='center', fontsize=8)
        ax.set_yticks(range(len(metrics_data)))
        ax.set_yticklabels([f"{d['Dataset']}\n{d['Metric']}" for d in metrics_data], fontsize=9)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                   markersize=8, label=train_label),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                   markersize=8, label='External')
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
    else:
        ax.text(0.5, 0.5, 'NRI data unavailable', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_title('NRI / IDI Forest Plot')


def plot_nri_reclassification_detail(nri_pkg, train_label, output_prefix, output_dir):
    """绘制 NRI 重分类细节图 (与原版一致)"""
    for ds_name, ds_key in [(f'{train_label.replace(" ", "_")}', 'train'),
                            ('External', 'external')]:
        nri_res = nri_pkg['nri_results'].get(ds_key)
        if nri_res is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle(f'{ds_name} Cohort', fontsize=14, y=1.02)

        # cfNRI 分解
        ax = axes[0]
        cf = nri_res['cfNRI']
        components = ['Events\nComponent', 'Non-events\nComponent', 'Total\ncfNRI']
        values = [cf['events_component'], cf['non_events_component'], cf['value']]
        colors = ['#d62728' if v < 0 else '#2ca02c' for v in values]
        bars = ax.bar(components, values, color=colors, alpha=0.8, edgecolor='none')
        for bar, val in zip(bars, values):
            val_range = max(abs(v) for v in values) if values else 1
            offset = val_range * 0.06
            y_pos = bar.get_height() + offset if val >= 0 else bar.get_height() - offset
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:+.4f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=10)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('NRI Component Value')
        ax.set_title(f'cfNRI Decomposition (Total={cf["value"]:.4f}, p={cf["p_value"]:.4f})', fontsize=10)
        ax.grid(axis='y', alpha=0.15)

        # IDI 分解
        ax = axes[1]
        idi = nri_res['IDI']
        idi_components = ['Events\n(\u0394 mean prob)', 'Non-events\n(\u0394 mean prob)', 'Total\nIDI']
        idi_values = [idi['events_component'], idi['non_events_component'], idi['value']]
        idi_colors = ['#d62728' if v < 0 else '#2ca02c' for v in idi_values]
        bars = ax.bar(idi_components, idi_values, color=idi_colors, alpha=0.8, edgecolor='none')
        for bar, val in zip(bars, idi_values):
            val_range = max(abs(v) for v in idi_values) if idi_values else 1
            offset = val_range * 0.06
            y_pos = bar.get_height() + offset if val >= 0 else bar.get_height() - offset
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:+.4f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=10)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('IDI Component Value')
        ax.set_title(f'IDI Decomposition (Total={idi["value"]:.4f}, p={idi["p_value"]:.4f})', fontsize=10)
        ax.grid(axis='y', alpha=0.15)

        # 重分类流向
        ax = axes[2]
        detail = cf['detail']
        categories = ['Events\nUpward', 'Events\nDownward',
                      'Non-events\nUpward', 'Non-events\nDownward']
        counts = [detail['event_up'], detail['event_down'],
                  detail['nonevent_up'], detail['nonevent_down']]
        bar_colors = ['#2ca02c', '#d62728', '#d62728', '#2ca02c']
        bars = ax.bar(categories, counts, color=bar_colors, alpha=0.8, edgecolor='none')
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    f'{val}', ha='center', fontsize=10)
        ax.set_ylabel('Number of Patients')
        ax.set_title('Reclassification Direction (Green=Correct, Red=Incorrect)', fontsize=10)
        ax.grid(axis='y', alpha=0.15)

        for a in axes:
            ymin, ymax = a.get_ylim()
            margin = (ymax - ymin) * 0.18
            a.set_ylim(bottom=ymin - margin, top=ymax + margin)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = f'{output_dir}/NRI_Reclassification_Detail_{output_prefix}_{ds_name}.png'
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  \u2713 重分类细节图已保存: {out_path}")


def plot_reclassification_heatmap(nri_pkg, train_label, new_model_label,
                                  old_model_label, output_prefix, output_dir):
    """绘制分类NRI重分类表热图 (与原版一致)"""
    for ds_name, ds_key in [(f'{train_label.replace(" ", "_")}', 'train'),
                            ('External', 'external')]:
        nri_res = nri_pkg['nri_results'].get(ds_key)
        if nri_res is None or 'categorical_NRI' not in nri_res:
            print(f"  \u26a0\ufe0f {ds_name} 无分类NRI数据，跳过")
            continue

        cat_nri = nri_res['categorical_NRI']
        reclass_events = cat_nri.get('reclassification_table_events')
        reclass_non_events = cat_nri.get('reclassification_table_non_events')
        if reclass_events is None:
            continue

        n_cats = reclass_events.shape[0]
        cat_labels = ['Low Risk', 'High Risk'] if n_cats == 2 else [f'Cat {i}' for i in range(n_cats)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{ds_name} Cohort')

        for idx, (table, group_name) in enumerate([
            (reclass_events, 'Recurrence'), (reclass_non_events, 'Non-recurrence')
        ]):
            ax = axes[idx]
            im = ax.imshow(table, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(n_cats))
            ax.set_yticks(range(n_cats))
            ax.set_xticklabels(cat_labels, fontsize=9)
            ax.set_yticklabels(cat_labels, fontsize=9)
            ax.set_xlabel(new_model_label, fontsize=10)
            ax.set_ylabel(old_model_label, fontsize=10)
            ax.set_title(group_name)
            for i in range(n_cats):
                for j in range(n_cats):
                    val = table[i, j]
                    text_color = 'white' if val > table.max() * 0.6 else 'black'
                    ax.text(j, i, str(val), ha='center', va='center', fontsize=12, color=text_color)
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            ax.tick_params(axis='both', which='both', length=0)
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        out_path = f'{output_dir}/Reclassification_Table_{output_prefix}_{ds_name}.png'
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  \u2713 重分类表热图已保存: {out_path}")


###########################################################################
#                                                                         #
#  Part 1: 术前 vs 术中 模型对比可视化                                      #
#                                                                         #
###########################################################################

print("\n" + "#" * 80)
print("#  Part 1: 术前 vs 术中 模型对比可视化")
print("#" * 80)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n" + "=" * 70)
print("\U0001f4e6 加载数据包 (术前 vs 术中)")
print("=" * 70)

preop_files = sorted(glob.glob(f"{DATA_PATH}/Model_Package_{TIMESTAMP}.pkl"))
preop_pkg = joblib.load(preop_files[-1]) if preop_files else None
if preop_pkg:
    print(f"  \u2713 已加载术前模型: {preop_files[-1]}")
else:
    print("  \u26a0\ufe0f 未找到术前模型数据包")

sens_files = sorted(glob.glob(f'{DATA_PATH}/Sensitivity_Model_Package_latest.pkl'))
sens_pkg = joblib.load(sens_files[-1]) if sens_files else None
if sens_pkg:
    print(f"  \u2713 已加载术中模型: {sens_files[-1]}")
else:
    print("  \u26a0\ufe0f 未找到术中模型数据包")

nri_files = sorted(glob.glob(f'{DATA_PATH}/NRI_Comparison_Data_latest.pkl'))
nri_pkg_intraop = joblib.load(nri_files[-1]) if nri_files else None
if nri_pkg_intraop:
    print(f"  \u2713 已加载NRI数据包: {nri_files[-1]}")
    train_source = nri_pkg_intraop.get('metadata', {}).get('train_data_source', 'unknown')
    print(f"    训练集数据源: {train_source}")
else:
    print("  \u26a0\ufe0f 未找到NRI数据包")

print()


# ============================================================
# 2. 风险分层图 — 术前模型 (单独 + Combined) 
# ============================================================
print("=" * 70)
print("\U0001f4ca [Step 2] 风险分层图 — 术前模型")
print("=" * 70)

if preop_pkg:
    risk_strat = preop_pkg.get('risk_stratification', {})

    # --- 2a. 仍生成单独的 Internal / External 图 ---
    for ds_name, ds_key, db in [
        ('Internal', 'train',    delta_b_int),
        ('External', 'external', delta_b_ext),
    ]:
        risk_data = risk_strat.get(ds_key)
        if risk_data is None:
            print(f"  \u26a0\ufe0f 术前模型缺少 {ds_name} 风险分层数据")
            continue
        plot_risk_stratification(risk_data, 'Preoperative_Model', ds_name, OUTPUT_DIR,
                                 delta_b=db)

    # --- 2b. [NEW] 生成 Combined Panel (2×3) ---
    risk_int = risk_strat.get('train')
    risk_ext = risk_strat.get('external')
    if risk_int is not None and risk_ext is not None:
        print("\n  --- 生成 Combined Panel ---")
        plot_risk_stratification_combined(
            risk_int, risk_ext,
            model_name='Preoperative_Model',
            output_dir=OUTPUT_DIR,
            delta_b_int=delta_b_int,
            delta_b_ext=delta_b_ext,
            model_display_name='Preoperative Model'
        )
        # 同时生成简化版 (仅固定切点, 1×2, 适合正文 Fig 4)
        plot_risk_stratification_combined_simple(
            risk_int, risk_ext,
            model_name='Preoperative_Model',
            output_dir=OUTPUT_DIR,
            delta_b_int=delta_b_int,
            delta_b_ext=delta_b_ext,
            model_display_name='Preoperative Model'
        )
else:
    print("  \u26a0\ufe0f 术前模型数据包不可用，跳过")

print()


# ============================================================
# 3. 风险分层图 — 术中模型 (单独 + Combined) 
# ============================================================
print("=" * 70)
print("\U0001f4ca [Step 3] 风险分层图 — 术中模型")
print("=" * 70)

if sens_pkg:
    risk_strat_intra = sens_pkg.get('risk_stratification', {})

    # --- 3a. 单独图 ---
    for ds_name, ds_key, db in [
        ('Internal', 'train',    delta_b_int),
        ('External', 'external', delta_b_ext),
    ]:
        risk_data = risk_strat_intra.get(ds_key)
        if risk_data is None:
            continue
        plot_risk_stratification(risk_data, 'PreIntraop_Model', ds_name, OUTPUT_DIR,
                                 delta_b=db)

    # --- 3b. [NEW] Combined Panel ---
    risk_int = risk_strat_intra.get('train')
    risk_ext = risk_strat_intra.get('external')
    if risk_int is not None and risk_ext is not None:
        print("\n  --- 生成 Combined Panel ---")
        plot_risk_stratification_combined(
            risk_int, risk_ext,
            model_name='PreIntraop_Model',
            output_dir=OUTPUT_DIR,
            delta_b_int=delta_b_int,
            delta_b_ext=delta_b_ext,
            model_display_name='Pre+Intraoperative Model'
        )
else:
    print("  \u26a0\ufe0f 术中模型数据包不可用，跳过")

print()


# ============================================================
# 4-7: NRI 对比可视化
# ============================================================
print("=" * 70)
print("\U0001f4ca [Step 4] NRI 对比可视化 — 术前 vs 术中")
print("=" * 70)

if nri_pkg_intraop:
    train_label = get_train_source_label(nri_pkg_intraop)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Preoperative vs. Pre+Intraoperative', fontsize=14)

    # 4-1: NRI/IDI 森林图
    ax = axes[0, 0]
    ax.set_xlabel('Improvement (Pre+Intraop - Preop)')
    plot_nri_forest(ax, nri_pkg_intraop, train_label)

    # 4-2: 概率散点图
    ax = axes[0, 1]
    prob_old = nri_pkg_intraop['preop_model']['prob_external']
    prob_new = nri_pkg_intraop['intraop_model']['prob_external']
    y_ext = nri_pkg_intraop['y_external']
    scatter_colors = ['#2ca02c' if y == 0 else '#d62728' for y in y_ext]
    ax.scatter(prob_old, prob_new, c=scatter_colors, alpha=0.6, s=30,
               edgecolors='white', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Preoperative Model Probability')
    ax.set_ylabel('Pre+Intraop Model Probability')
    ax.set_title('Probability Scatter (External Validation)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markersize=8, label='Event (Recurrence)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=8, label='Non-event')
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.annotate('Pre+Intraop \u2191\npredicts higher',
                xy=(0.15, 0.85), fontsize=8, color='gray', ha='center', va='center')
    ax.annotate('Preop \u2191\npredicts higher',
                xy=(0.85, 0.15), fontsize=8, color='gray', ha='center', va='center')

    # 4-3 / 4-4: 风险分层对比 (外部)
    y_ext_nri = nri_pkg_intraop['y_external']
    for idx, (model_key, model_label) in enumerate([
        ('preop_model', 'Preoperative'),
        ('intraop_model', 'Pre+Intraoperative')
    ]):
        ax = axes[1, idx]
        prob_orig = nri_pkg_intraop[model_key]['prob_external']
        db_this, prob_cal_this, _ = intercept_only_recalibration(
            y_ext_nri, prob_orig, method='mle')
        tertile_cal = compute_tertile_risk_groups(y_ext_nri, prob_cal_this)
        labels = tertile_cal['labels']
        event_rates = [tertile_cal['stats'][g]['event_rate'] for g in range(3)]
        ns = [tertile_cal['stats'][g]['n'] for g in range(3)]
        bar_colors = ['#2ca02c', '#ff7f0e', '#d62728']
        bars = ax.bar(labels, event_rates, color=bar_colors, alpha=0.8)
        for bar, val, n in zip(bars, event_rates, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.1%}\n(n={n})', ha='center', fontsize=9)
        auc_key = model_key.replace('_model', '')
        auc_val = nri_pkg_intraop['auc_comparison']['external'][auc_key]
        ax.set_title(f'{model_label} (Calibrated)\n'
                     f'External AUC = {auc_val:.3f}, \u0394b = {db_this:.3f}')
        ax.set_ylabel('Event Rate')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Risk Tertile')

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/NRI_Intraop_Model_Comparison.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 NRI 对比图已保存: {out_path}")
else:
    print("  \u26a0\ufe0f 未找到 NRI 数据包")

print()

# Step 5: NRI 重分类细节图
print("=" * 70)
print("\U0001f4ca [Step 5] NRI 重分类细节图 — 术中模型")
print("=" * 70)
if nri_pkg_intraop:
    train_label = get_train_source_label(nri_pkg_intraop)
    plot_nri_reclassification_detail(nri_pkg_intraop, train_label,
                                     output_prefix='Intraop', output_dir=OUTPUT_DIR)
print()

# Step 6: AUC 对比汇总
print("=" * 70)
print("\U0001f4ca [Step 6] AUC 对比汇总图 — 术中模型")
print("=" * 70)
if nri_pkg_intraop:
    auc_data = nri_pkg_intraop['auc_comparison']
    train_label = get_train_source_label(nri_pkg_intraop)
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = [f'Internal Validation ({auc_data["train"].get("data_source", "OOF")})', 'External']
    auc_preop = [auc_data['train']['preop'], auc_data['external']['preop']]
    auc_intraop = [auc_data['train']['intraop'], auc_data['external']['intraop']]
    x = np.arange(len(datasets))
    width = 0.3
    bars1 = ax.bar(x - width / 2, auc_preop, width, label='Preoperative', color='#80C764', alpha=0.85)
    bars2 = ax.bar(x + width / 2, auc_intraop, width, label='Pre+Intraoperative', color='#E38D80', alpha=0.85)
    for bar, val in zip(bars1, auc_preop):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', fontsize=10)
    for bar, val in zip(bars2, auc_intraop):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', fontsize=10)
    for i in range(len(datasets)):
        delta = auc_intraop[i] - auc_preop[i]
        y_max = max(auc_preop[i], auc_intraop[i])
        ax.annotate(f'\u0394={delta:+.3f}', xy=(x[i], y_max + 0.025),
                    fontsize=9, ha='center', color='#333333', fontstyle='italic')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel('AUC')
    ax.set_title('Pre- vs. Pre+Intraoperative', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.15)
    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/AUC_Comparison_Intraop.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 AUC 对比图已保存: {out_path}")

print()

# Step 7: 分类NRI重分类表热图
print("=" * 70)
print("\U0001f4ca [Step 7] 分类NRI 重分类表热图 — 术中模型")
print("=" * 70)
if nri_pkg_intraop:
    train_label = get_train_source_label(nri_pkg_intraop)
    plot_reclassification_heatmap(nri_pkg_intraop, train_label,
                                  new_model_label='Pre+Intraoperative',
                                  old_model_label='Preoperative',
                                  output_prefix='Intraop', output_dir=OUTPUT_DIR)

print()
print("=" * 70)
print("\U0001f389 Part 1 完成!")
print("=" * 70)


###########################################################################
#                                                                         #
#  Part 2: 术前 vs 裂孔聚类 模型对比可视化                                  #
#                                                                         #
###########################################################################

print("\n" + "#" * 80)
print("#  Part 2: 术前 vs 裂孔聚类 模型对比可视化")
print("#" * 80)

# 1. 加载数据
print("\n" + "=" * 70)
print("\U0001f4e6 加载数据包 (术前 vs 裂孔聚类)")
print("=" * 70)

if preop_pkg:
    print(f"  \u2713 原术前模型已加载 (复用)")

sens_bc_files = sorted(glob.glob(f'{DATA_PATH}/Sensitivity_BreakCluster_Package_latest.pkl'))
sens_bc_pkg = joblib.load(sens_bc_files[-1]) if sens_bc_files else None
if sens_bc_pkg:
    print(f"  \u2713 已加载裂孔聚类模型: {sens_bc_files[-1]}")

nri_bc_files = sorted(glob.glob(f'{DATA_PATH}/NRI_BreakCluster_Comparison_Data_latest.pkl'))
nri_pkg_cluster = joblib.load(nri_bc_files[-1]) if nri_bc_files else None
if nri_pkg_cluster:
    print(f"  \u2713 已加载NRI对比数据包: {nri_bc_files[-1]}")
    train_source = nri_pkg_cluster.get('metadata', {}).get('train_data_source', 'unknown')
    print(f"    训练集数据源: {train_source}")

print()


# 3. 风险分层图 — 裂孔聚类模型 (单独 + Combined) 
print("=" * 70)
print("\U0001f4ca [Step 3] 风险分层图 — 裂孔聚类模型")
print("=" * 70)

if sens_bc_pkg:
    risk_strat_cluster = sens_bc_pkg.get('risk_stratification', {})

    # --- 单独图 ---
    for ds_name, ds_key, db in [
        ('Internal', 'train',    delta_b_int),
        ('External', 'external', delta_b_ext),
    ]:
        risk_data = risk_strat_cluster.get(ds_key)
        if risk_data is None:
            continue
        plot_risk_stratification(risk_data, 'BreakCluster_Model', ds_name, OUTPUT_DIR,
                                 delta_b=db)

    # --- Combined Panel ---
    risk_int = risk_strat_cluster.get('train')
    risk_ext = risk_strat_cluster.get('external')
    if risk_int is not None and risk_ext is not None:
        print("\n  --- 生成 Combined Panel ---")
        plot_risk_stratification_combined(
            risk_int, risk_ext,
            model_name='BreakCluster_Model',
            output_dir=OUTPUT_DIR,
            delta_b_int=delta_b_int,
            delta_b_ext=delta_b_ext,
            model_display_name='Break Cluster Model'
        )
else:
    print("  \u26a0\ufe0f 裂孔聚类模型数据包不可用，跳过")

print()



# ============================================================
# 4. NRI 对比可视化
# ============================================================
print("=" * 70)
print("📊 [Step 4] NRI 对比可视化 — 原术前 vs 裂孔聚类")
print("=" * 70)

if nri_pkg_cluster:
    train_label = get_train_source_label(nri_pkg_cluster)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Original Preop vs Break Cluster',
                 fontsize=14)

    # ----------------------------------------------------------
    # 图 4-1: NRI / IDI 森林图
    # ----------------------------------------------------------
    ax = axes[0, 0]
    ax.set_xlabel('Improvement (Break Cluster - Original)')
    plot_nri_forest(ax, nri_pkg_cluster, train_label)

    # ----------------------------------------------------------
    # 图 4-2: 概率散点图（外部验证集）
    # ----------------------------------------------------------
    ax = axes[0, 1]
    prob_old = nri_pkg_cluster['preop_model']['prob_external']
    prob_new = nri_pkg_cluster['cluster_model']['prob_external']
    y_ext    = nri_pkg_cluster['y_external']

    scatter_colors = ['#2ca02c' if y == 0 else '#d62728' for y in y_ext]
    ax.scatter(prob_old, prob_new, c=scatter_colors, alpha=0.6, s=30,
               edgecolors='white', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Original Preop Model Probability')
    ax.set_ylabel('Break Cluster Model Probability')
    ax.set_title('Probability Scatter (External Validation)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markersize=8, label='Event (Recurrence)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=8, label='Non-event')
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    ax.annotate('Break Cluster ↑\npredicts higher',
                xy=(0.15, 0.85), fontsize=8, color='gray',
                ha='center', va='center')
    ax.annotate('Original ↑\npredicts higher',
                xy=(0.85, 0.15), fontsize=8, color='gray',
                ha='center', va='center')

    # ----------------------------------------------------------
    # 图 4-3 / 4-4: 两模型风险分层对比（外部验证集）—— 校准后
    # ----------------------------------------------------------
    y_ext_nri = nri_pkg_cluster['y_external']

    for idx, (model_key, model_label, auc_key) in enumerate([
        ('preop_model',   'Original Preop', 'preop'),
        ('cluster_model', 'Break Cluster',  'cluster')
    ]):
        ax = axes[1, idx]

        # 取该模型的原始概率
        prob_orig = nri_pkg_cluster[model_key]['prob_external']

        # 为该模型单独计算截距校准
        db_this, prob_cal_this, _ = intercept_only_recalibration(
            y_ext_nri, prob_orig, method='mle')
        print(f"    {model_label.split(chr(10))[0]} 外部 Δb = {db_this:.4f}")

        # 用校准后概率重新计算三分位
        tertile_cal = compute_tertile_risk_groups(y_ext_nri, prob_cal_this)

        labels      = tertile_cal['labels']
        event_rates = [tertile_cal['stats'][g]['event_rate'] for g in range(3)]
        ns          = [tertile_cal['stats'][g]['n']           for g in range(3)]

        bar_colors = ['#2ca02c', '#ff7f0e', '#d62728']
        bars = ax.bar(labels, event_rates, color=bar_colors, alpha=0.8)
        for bar, val, n in zip(bars, event_rates, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.1%}\n(n={n})', ha='center', fontsize=9)

        auc_val = nri_pkg_cluster['auc_comparison']['external'][auc_key]
        ax.set_title(f'{model_label} (Calibrated)\n'
                     f'External AUC = {auc_val:.3f}, Δb = {db_this:.3f}')
        ax.set_ylabel('Event Rate')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Risk Tertile')

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/NRI_BreakCluster_Model_Comparison.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✓ NRI 对比图已保存: {out_path}")

else:
    print("  ⚠️ 未找到 NRI 数据包，请先运行 4_裂孔聚类_敏感性_生存分析.py")

print()


# ============================================================
# 5. 补充图: NRI 重分类细节图（事件组 / 非事件组）
# ============================================================
print("=" * 70)
print("📊 [Step 5] NRI 重分类细节图 — 裂孔聚类")
print("=" * 70)

if nri_pkg_cluster:
    train_label = get_train_source_label(nri_pkg_cluster)
    plot_nri_reclassification_detail(
        nri_pkg_cluster, train_label,
        output_prefix='BreakCluster', output_dir=OUTPUT_DIR)

print()


# ============================================================
# 6. 补充图: AUC 对比汇总柱状图
# ============================================================
print("=" * 70)
print("📊 [Step 6] AUC 对比汇总图 — 裂孔聚类")
print("=" * 70)

if nri_pkg_cluster:
    auc_data = nri_pkg_cluster['auc_comparison']

    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = [f'Internal Validation ({auc_data["train"].get("data_source", "OOF")})',
                'External']
    auc_preop   = [auc_data['train']['preop'],     auc_data['external']['preop']]
    auc_cluster = [auc_data['train']['cluster'],   auc_data['external']['cluster']]

    x = np.arange(len(datasets))
    width = 0.3

    bars1 = ax.bar(x - width / 2, auc_preop, width,
                   label='Original Preop', color="#80C764", alpha=0.85)
    bars2 = ax.bar(x + width / 2, auc_cluster, width,
                   label='Break Cluster', color="#E38D80", alpha=0.85)

    for bar, val in zip(bars1, auc_preop):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=10)
    for bar, val in zip(bars2, auc_cluster):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=10)

    for i in range(len(datasets)):
        delta = auc_cluster[i] - auc_preop[i]
        y_max = max(auc_preop[i], auc_cluster[i])
        ax.annotate(f'Δ={delta:+.3f}',
                    xy=(x[i], y_max + 0.025),
                    fontsize=9, ha='center', color='#333333', fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel('AUC')
    ax.set_title('Original Preop vs Break Cluster Model',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.15, linewidth=0.5)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/AUC_Comparison_BreakCluster.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✓ AUC 对比图已保存: {out_path}")
else:
    print("  ⚠️ NRI 数据包不可用，跳过")


# ============================================================
# 7. 补充图: 分类NRI重分类表热图
# ============================================================
print()
print("=" * 70)
print("📊 [Step 7] 分类NRI 重分类表热图 — 裂孔聚类")
print("=" * 70)

if nri_pkg_cluster:
    train_label = get_train_source_label(nri_pkg_cluster)
    plot_reclassification_heatmap(
        nri_pkg_cluster, train_label,
        new_model_label='Break Cluster Model',
        old_model_label='Original Preop Model',
        output_prefix='BreakCluster', output_dir=OUTPUT_DIR)

print()
print("=" * 70)
print("🎉 Part 2 完成: 裂孔聚类敏感性分析可视化！")
print("=" * 70)
print(f"\n  📁 所有图片已保存至: {OUTPUT_DIR}/")
print(f"  📊 生成图片清单:")

for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png') and ('BreakCluster' in f or 'Preoperative' in f):
        print(f"     • {f}")

print()
print("=" * 70)
print("🎉 敏感性分析可视化完成！")
print("=" * 70)

# ============================================================
# 特征频率可视化（完整版）
# ============================================================
from matplotlib.patches import Rectangle

# ------------------------------------------
# 数据加载
# ------------------------------------------
# 主模型
package = joblib.load(f"{DATA_PATH}/Model_Package_{TIMESTAMP}.pkl")
best_model_name = package['best_model_name']
freq_df = pd.read_csv(f"{DATA_PATH}/Feature_Selection_Frequency_{TIMESTAMP}.csv")
rename_df_feature_col(freq_df)  # ✅ 映射为 SCI 展示标签

# 术中模型 & 裂孔聚类模型
pkg_intra = joblib.load(f"{DATA_PATH}/Sensitivity_Model_Package_latest.pkl")
pkg_cluster = joblib.load(f"{DATA_PATH}/Sensitivity_BreakCluster_Package_latest.pkl")

best_intra = pkg_intra['best_model_name']
best_cluster = pkg_cluster['best_model_name']

freq_intra = pd.read_csv(f"{DATA_PATH}/Feature_Selection_Frequency_20260220_2214.csv")
freq_cluster = pd.read_csv(f"{DATA_PATH}/Feature_Selection_Frequency_20260220_2220.csv")
rename_df_feature_col(freq_intra)    # ✅ 映射为 SCI 展示标签
rename_df_feature_col(freq_cluster)  # ✅ 映射为 SCI 展示标签

print(f"主模型: {best_model_name}, 特征数: {len(freq_df[freq_df['Model']==best_model_name])}")
print(f"术中模型: {best_intra}, 特征数: {len(freq_intra[freq_intra['Model']==best_intra])}")
print(f"裂孔聚类: {best_cluster}, 特征数: {len(freq_cluster[freq_cluster['Model']==best_cluster])}")

# 三个场景的配置
SCENARIOS = {
    'Original Preop Model': {
        'timestamp': TIMESTAMP,
        'package': package,
        'best_model': best_model_name,
        'freq_df': freq_df,
    },
    'Pre+Intraoperative Model': {
        'timestamp': '20260220_2214',
        'package': pkg_intra,
        'best_model': best_intra,
        'freq_df': freq_intra,
    },
    'Break Cluster Model': {
        'timestamp': '20260220_2220',
        'package': pkg_cluster,
        'best_model': best_cluster,
        'freq_df': freq_cluster,
    },
}

# ============================================================
# 图1: 棒棒糖图 - LASSO特征选择稳定性
# ============================================================
def plot_lollipop_chart(data, model_name, title_suffix='', save_path=None):
    """
    棒棒糖图：展示LASSO在交叉验证中选择每个特征的稳定性
    """
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
        ax.text(freq + 0.05, i, f'{freq:.1%}',
                va='center', fontsize=7)

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
    plt.close()

# 为三个场景各画一张棒棒糖图
for label, cfg in SCENARIOS.items():
    df_sub = cfg['freq_df']
    df_sub = df_sub[df_sub['Model'] == cfg['best_model']].copy()
    df_sub = df_sub.sort_values('Selection_Freq', ascending=True)
    safe = label.replace(' ', '_')
    plot_lollipop_chart(df_sub, cfg['best_model'],
                        title_suffix=f'{label}',
                        save_path=f'figures/LASSO_Lollipop_{safe}.png')
    print(f"✅ Lollipop saved: {label}")


# ============================================================
# 图2: 热力图 - 三场景特征选择频率对比
# ============================================================
scenario_series = {}
for label, cfg in SCENARIOS.items():
    df_sub = cfg['freq_df']
    df_sub = df_sub[df_sub['Model'] == cfg['best_model']].copy()
    scenario_series[label] = df_sub.set_index('Feature')['Selection_Freq']

freq_wide = pd.DataFrame(scenario_series).fillna(0)
freq_wide = freq_wide.sort_values('Original Preop Model', ascending=True)

def plot_feature_heatmap(data, save_path=None):
    fig, ax = plt.subplots(figsize=(5, max(6, len(data) * 0.35)))
    im = ax.imshow(data.values, aspect='auto', cmap=plt.cm.Blues, vmin=0, vmax=1)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            color = 'white' if val > 0.65 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Selection Frequency', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # ax.set_title('LASSO Feature Selection Across Scenarios', fontsize=10, pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

plot_feature_heatmap(freq_wide,
                     save_path='figures/LASSO_Feature_Heatmap_Compare.png')
print("✅ Heatmap saved")


# ============================================================
# 图3: UpSet 风格点阵图 — 各场景共享/独有特征
# ============================================================
def plot_dot_matrix(data, threshold=0.5, save_path=None):
    selected = (data >= threshold).astype(int)
    selected['Pattern'] = selected.apply(lambda r: tuple(r), axis=1)
    pattern_groups = selected.groupby('Pattern').apply(lambda g: list(g.index))
    patterns = sorted(pattern_groups.items(), key=lambda x: -len(x[1]))
    scenarios = data.columns.tolist()

    fig, (ax_bar, ax_dot) = plt.subplots(
        2, 1, figsize=(max(6, len(patterns) * 0.9), 4),
        gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    x = np.arange(len(patterns))
    counts = [len(feats) for _, feats in patterns]

    ax_bar.bar(x, counts, color=COLORS_LIST[3], width=0.55, edgecolor='white')
    for xi, c in zip(x, counts):
        ax_bar.text(xi, c + 0.15, str(c), ha='center', va='bottom', fontsize=8)
    ax_bar.set_ylabel('Feature Count')
    ax_bar.set_title(f'Feature Overlap Across Scenarios (freq ≥ {threshold:.0%})',
                     fontsize=10, pad=10)
    ax_bar.set_ylim(0, max(counts) * 1.25)

    for i, (pat, _) in enumerate(patterns):
        for j, val in enumerate(pat):
            color = COLORS_LIST[0] if val else '#dddddd'
            ax_dot.scatter(i, j, s=120, color=color, edgecolors='white',
                           linewidth=0.8, zorder=3)
        active = [j for j, v in enumerate(pat) if v]
        if len(active) > 1:
            ax_dot.plot([i] * len(active), active, color=COLORS_LIST[0],
                        linewidth=1.5, zorder=2)

    ax_dot.set_yticks(range(len(scenarios)))
    ax_dot.set_yticklabels(scenarios, fontsize=8)
    ax_dot.set_xticks([])
    ax_dot.set_xlim(-0.5, len(patterns) - 0.5)
    ax_dot.invert_yaxis()
    ax_dot.spines['bottom'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

if len(freq_wide.columns) > 1:
    plot_dot_matrix(freq_wide, threshold=0.5,
                    save_path='figures/LASSO_Feature_Overlap_Dot.png')
    print("✅ Dot matrix saved")


# ============================================================
# 图4: 特征重要性对比图 (Grouped Bar — 三场景)
# ============================================================

def extract_feature_importance(pkg, freq_df_best):
    """
    从模型包中提取最佳模型的特征重要性
    支持: tree-based (.feature_importances_), linear (.coef_)
    若都不可用，回退到 LASSO 选择频率作为替代
    """
    model = pkg.get('best_model', pkg.get('final_model', None))
    features = freq_df_best['Feature'].tolist()

    # tree-based
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        # 可能是 pipeline，取最后一步
        if len(imp) != len(features):
            # 尝试 pipeline 的最后一步
            if hasattr(model, 'named_steps'):
                for step_name in reversed(list(model.named_steps)):
                    step = model.named_steps[step_name]
                    if hasattr(step, 'feature_importances_'):
                        imp = step.feature_importances_
                        break
        if len(imp) == len(features):
            return pd.Series(np.abs(imp), index=features, name='Importance')

    # linear
    if hasattr(model, 'coef_'):
        coef = np.array(model.coef_).ravel()
        if len(coef) == len(features):
            return pd.Series(np.abs(coef), index=features, name='Importance')

    # pipeline 处理
    if hasattr(model, 'named_steps'):
        for step_name in reversed(list(model.named_steps)):
            step = model.named_steps[step_name]
            if hasattr(step, 'feature_importances_'):
                imp = step.feature_importances_
                if len(imp) == len(features):
                    return pd.Series(np.abs(imp), index=features, name='Importance')
            if hasattr(step, 'coef_'):
                coef = np.array(step.coef_).ravel()
                if len(coef) == len(features):
                    return pd.Series(np.abs(coef), index=features, name='Importance')

    # 回退：用选择频率代替
    print("  ⚠️ 无法提取模型系数，回退使用 Selection_Freq")
    return pd.Series(freq_df_best['Selection_Freq'].values,
                     index=features, name='Importance')


def plot_importance_comparison(scenarios_cfg, save_path=None):
    """
    分组柱状图：对比三个场景中各特征的重要性（归一化到 0-1）
    """
    importance_dict = {}
    for label, cfg in scenarios_cfg.items():
        df_sub = cfg['freq_df']
        df_sub = df_sub[df_sub['Model'] == cfg['best_model']].copy()
        imp = extract_feature_importance(cfg['package'], df_sub)
        # 归一化
        if imp.max() > 0:
            imp = imp / imp.max()
        importance_dict[label] = imp

    # 合并
    imp_all = pd.DataFrame(importance_dict).fillna(0)
    # 按主模型重要性降序
    if 'Original Preop Model' in imp_all.columns:
        imp_all = imp_all.sort_values('Original Preop Model', ascending=True)

    n_features = len(imp_all)
    n_scenarios = len(imp_all.columns)
    bar_height = 0.22
    y = np.arange(n_features)

    fig, ax = plt.subplots(figsize=(8, max(6, n_features * 0.45)))

    for i, col in enumerate(imp_all.columns):
        offset = (i - n_scenarios / 2 + 0.5) * bar_height
        ax.barh(y + offset, imp_all[col], height=bar_height * 0.9,
                color=COLORS_LIST[i], label=col, edgecolor='white', linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(imp_all.index, fontsize=8)
    ax.set_xlabel('Selection Frequency (norm.)')
    # ax.set_title('Feature Importance Comparison Across Scenarios', fontsize=10, pad=12)
    ax.set_xlim(0, 1.12)
    sci_legend(ax, loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.15, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

plot_importance_comparison(SCENARIOS,
                           save_path='figures/Feature_Importance_Compare.png')
print("✅ Importance comparison saved")


print("\n===== 所有可视化完成 =====")