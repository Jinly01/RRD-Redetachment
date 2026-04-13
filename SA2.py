import os
os.environ['SCIPY_ARRAY_API'] = '1'
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, brier_score_loss, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.base import clone
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')
import importlib.util
spec = importlib.util.spec_from_file_location("model_training", "2_模型训练.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

MetricsCalculator = mt.MetricsCalculator
MultiModelTrainer = mt.MultiModelTrainer
make_column_transformer = mt.make_column_transformer

# ============================================================
# 配置
# ============================================================
class SensitivityConfig:
    TRAIN_DATA = 'Internal_setA_NaN.csv'
    EXTERNAL_DATA = 'External_setA_NaN.csv'
    TARGET_COL = 'Recurrence'
    RANDOM_STATE = 42
    CV_FOLDS = 5
    CV_REPEATS = 20
    OUTPUT_DIR = 'model_results'
    PREOP_MODEL_PKL = 'model_results/Model_Package_20260216_1724.pkl'


# ============================================================
# 数据加载
# ============================================================
def load_data_with_intraop(cfg):
    """加载数据，特征列表包含术中变量"""
    
    print(f"\n{'='*60}")
    print("敏感性分析 - 加载数据（含术中变量）")
    print(f"{'='*60}")
    
    train_df = pd.read_csv(cfg.TRAIN_DATA, index_col=0)
    external_df = pd.read_csv(cfg.EXTERNAL_DATA, index_col=0)
    
    train_df = train_df.dropna(subset=[cfg.TARGET_COL]).reset_index(drop=True)
    external_df = external_df.dropna(subset=[cfg.TARGET_COL]).reset_index(drop=True)
    
    y_train = train_df[cfg.TARGET_COL].astype(int)
    y_external = external_df[cfg.TARGET_COL].astype(int)
    
    # ============================================================
    # 特征列表 = 术前变量 + 术中变量
    # ============================================================
    
    # --- 术前变量---
    log_features = ['Symptom_Duration']
    numeric_features_preop = ['AL', 'RD_Extent', 'number_of_breaks', 'Largest_Break_Diameter']
    categorical_features_preop = [
        'PVR_Grade_Pre', 
        'Break_Loc_Inferior',
        'Choroidal_Detachment',
        'Lens_Status_Pre',
        'Macular_status',
    ]
    
    # --- 术中变量 ---
    numeric_features_intraop = ['Surgery_Duration']
    
    categorical_features_intraop = [
        'PFCL',
        'Phacovitrectomy'
    ]
    
    # 合并
    numeric_features = numeric_features_preop + numeric_features_intraop
    categorical_features = categorical_features_preop + categorical_features_intraop
    all_features = log_features + numeric_features + categorical_features
    
    # 验证
    missing = [f for f in all_features if f not in train_df.columns]
    if missing:
        raise ValueError(f"❌ 训练集缺少特征: {missing}")
    
    X_train = train_df[all_features].copy()
    X_external = external_df[all_features].copy()
    
    for col in numeric_features + log_features:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_external[col] = pd.to_numeric(X_external[col], errors='coerce')
    
    print(f"\n  术前变量: {len(numeric_features_preop) + len(categorical_features_preop) + len(log_features)} 个")
    print(f"  术中变量: {len(numeric_features_intraop) + len(categorical_features_intraop)} 个")
    print(f"  总特征数: {len(all_features)} 个")
    print(f"  训练集: {len(X_train)} 例, 外部集: {len(X_external)} 例")
    
    return (X_train, X_external, y_train, y_external, 
            train_df, external_df,
            log_features, numeric_features, categorical_features)


# ============================================================
# NRI 计算函数
# ============================================================
def compute_nri(y_true, prob_old, prob_new, threshold=None):

    y_true = np.array(y_true, dtype=int)
    prob_old = np.array(prob_old, dtype=float)
    prob_new = np.array(prob_new, dtype=float)
    
    events = y_true == 1
    non_events = y_true == 0
    n_events = events.sum()
    n_non_events = non_events.sum()
    
    results = {}
    
    # ---- 1. Category-free NRI (cfNRI) ----
    # 事件组：新模型概率升高是好事
    event_up = np.sum(prob_new[events] > prob_old[events])
    event_down = np.sum(prob_new[events] < prob_old[events])
    
    # 非事件组：新模型概率降低是好事
    nonevent_up = np.sum(prob_new[non_events] > prob_old[non_events])
    nonevent_down = np.sum(prob_new[non_events] < prob_old[non_events])
    
    nri_events = (event_up - event_down) / n_events if n_events > 0 else 0
    nri_non_events = (nonevent_down - nonevent_up) / n_non_events if n_non_events > 0 else 0
    cfnri = nri_events + nri_non_events
    
    # cfNRI 的 Z 检验
    se_events = np.sqrt((event_up + event_down) / n_events**2) if n_events > 0 else 0
    se_non_events = np.sqrt((nonevent_up + nonevent_down) / n_non_events**2) if n_non_events > 0 else 0
    se_cfnri = np.sqrt(se_events**2 + se_non_events**2)
    z_cfnri = cfnri / se_cfnri if se_cfnri > 0 else 0
    from scipy.stats import norm
    p_cfnri = 2 * (1 - norm.cdf(abs(z_cfnri)))
    
    results['cfNRI'] = {
        'value': cfnri,
        'events_component': nri_events,
        'non_events_component': nri_non_events,
        'se': se_cfnri,
        'z': z_cfnri,
        'p_value': p_cfnri,
        '95CI': [cfnri - 1.96 * se_cfnri, cfnri + 1.96 * se_cfnri],
        'detail': {
            'event_up': int(event_up), 'event_down': int(event_down),
            'nonevent_up': int(nonevent_up), 'nonevent_down': int(nonevent_down)
        }
    }
    
    # ---- 2. Category-based NRI----
    if threshold is not None:
        if isinstance(threshold, (int, float)):
            thresholds = [threshold]
        else:
            thresholds = sorted(threshold)
        
        # 分类函数
        def classify(probs, cuts):
            return np.digitize(probs, cuts)
        
        cat_old = classify(prob_old, thresholds)
        cat_new = classify(prob_new, thresholds)
        
        # 事件组
        event_up_cat = np.sum(cat_new[events] > cat_old[events])
        event_down_cat = np.sum(cat_new[events] < cat_old[events])
        
        # 非事件组
        nonevent_up_cat = np.sum(cat_new[non_events] > cat_old[non_events])
        nonevent_down_cat = np.sum(cat_new[non_events] < cat_old[non_events])
        
        nri_events_cat = (event_up_cat - event_down_cat) / n_events if n_events > 0 else 0
        nri_non_events_cat = (nonevent_down_cat - nonevent_up_cat) / n_non_events if n_non_events > 0 else 0
        cat_nri = nri_events_cat + nri_non_events_cat
        
        # SE and p-value
        se_cat = np.sqrt(
            (event_up_cat + event_down_cat) / n_events**2 +
            (nonevent_up_cat + nonevent_down_cat) / n_non_events**2
        ) if (n_events > 0 and n_non_events > 0) else 0
        z_cat = cat_nri / se_cat if se_cat > 0 else 0
        p_cat = 2 * (1 - norm.cdf(abs(z_cat)))
        
        # 重分类表
        n_cats = len(thresholds) + 1
        reclass_table_events = np.zeros((n_cats, n_cats), dtype=int)
        reclass_table_non_events = np.zeros((n_cats, n_cats), dtype=int)
        
        for i in range(n_cats):
            for j in range(n_cats):
                reclass_table_events[i, j] = np.sum((cat_old[events] == i) & (cat_new[events] == j))
                reclass_table_non_events[i, j] = np.sum((cat_old[non_events] == i) & (cat_new[non_events] == j))
        
        results['categorical_NRI'] = {
            'value': cat_nri,
            'events_component': nri_events_cat,
            'non_events_component': nri_non_events_cat,
            'se': se_cat,
            'z': z_cat,
            'p_value': p_cat,
            '95CI': [cat_nri - 1.96 * se_cat, cat_nri + 1.96 * se_cat],
            'thresholds_used': thresholds,
            'reclassification_table_events': reclass_table_events,
            'reclassification_table_non_events': reclass_table_non_events,
            'detail': {
                'event_up': int(event_up_cat), 'event_down': int(event_down_cat),
                'nonevent_up': int(nonevent_up_cat), 'nonevent_down': int(nonevent_down_cat)
            }
        }
    
    # ---- 3. IDI (Integrated Discrimination Improvement) ----
    # 作为 NRI 的补充指标
    idi_events = np.mean(prob_new[events]) - np.mean(prob_old[events])
    idi_non_events = np.mean(prob_new[non_events]) - np.mean(prob_old[non_events])
    idi = idi_events - idi_non_events
    
    # IDI 的 Bootstrap SE
    n_boot = 1000
    boot_idis = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        yt, po, pn = y_true[idx], prob_old[idx], prob_new[idx]
        ev, nev = yt == 1, yt == 0
        if ev.sum() > 0 and nev.sum() > 0:
            boot_idis.append(
                (np.mean(pn[ev]) - np.mean(po[ev])) - 
                (np.mean(pn[nev]) - np.mean(po[nev]))
            )
    se_idi = np.std(boot_idis) if boot_idis else 0
    z_idi = idi / se_idi if se_idi > 0 else 0
    p_idi = 2 * (1 - norm.cdf(abs(z_idi)))
    
    results['IDI'] = {
        'value': idi,
        'events_component': idi_events,
        'non_events_component': idi_non_events,
        'se': se_idi,
        'z': z_idi,
        'p_value': p_idi,
        '95CI': [idi - 1.96 * se_idi, idi + 1.96 * se_idi]
    }
    
    return results


# ============================================================
# 风险分层计算
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
# 从 pkl 中安全提取特征列表
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
# 主程序
# ============================================================
if __name__ == "__main__":
    cfg = SensitivityConfig()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    print("="*80)
    print("敏感性分析 — 术中变量模型")
    print("="*80)
    
    # --------------------------------------------------------
    # 1. 加载数据（含术中变量）
    # --------------------------------------------------------
    (X_train, X_ext, y_train, y_ext, 
     train_df, external_df,
     log_feats, num_feats, cat_feats) = load_data_with_intraop(cfg)
    
    # --------------------------------------------------------
    # 2. 构建预处理器 & 训练
    # --------------------------------------------------------
    preprocessor = make_column_transformer(log_feats, num_feats, cat_feats)
    
    trainer = MultiModelTrainer(preprocessor, random_state=cfg.RANDOM_STATE)
    
    results = trainer.train_all(
        X_train, y_train, X_ext, y_ext,
        cv_folds=cfg.CV_FOLDS,
        cv_repeats=cfg.CV_REPEATS
    )
    
    best_model_intraop = results['final_model']
    threshold_intraop = results['optimal_threshold']
    best_name_intraop = results['best_model_name']
    
    # 获取术中模型的预测概率 — 区分 OOF 和 resubstitution
    prob_train_intraop = results['prob_train']        # 全量训练集 resubstitution
    prob_ext_intraop = results['prob_external']       # 外部验证集
    prob_oof_intraop = results.get('prob_oof', None)  # OOF 预测（内部验证主指标）
    
    # --------------------------------------------------------
    # 3. 术中模型风险分层
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("📊 术中模型 — 风险分层分析")
    print(f"{'='*60}")
    
    # 训练集风险分层：优先使用 OOF 概率（与 NRI 数据源保持一致）
    prob_train_intraop_for_strat = prob_oof_intraop if prob_oof_intraop is not None else prob_train_intraop
    
    risk_strat_intraop = {
        'train': compute_risk_strata(y_train, prob_train_intraop_for_strat, threshold_intraop),
        'external': compute_risk_strata(y_ext, prob_ext_intraop, threshold_intraop)
    }
    
    # 打印风险分层结果
    for ds_name, strat in risk_strat_intraop.items():
        print(f"\n  [{ds_name}] 三分位风险分层:")
        for g, label in enumerate(strat['tertile']['labels']):
            s = strat['tertile']['stats'][g]
            print(f"    {label}: n={s['n']}, 事件={s['events']}, "
                  f"事件率={s['event_rate']:.1%}, 平均概率={s['mean_prob']:.3f}")
    
    # --------------------------------------------------------
    # 4. 加载术前模型 → 计算 NRI
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("📊 NRI 分析 — 术前 vs 术前+术中 模型比较")
    print(f"{'='*60}")
    
    # 初始化变量，确保 except 分支和后续保存阶段可引用
    preop_oof_probs = None
    prob_train_preop = None
    prob_ext_preop = None
    preop_threshold = None
    
    try:
        preop_pkg = joblib.load(cfg.PREOP_MODEL_PKL)
        print(f"  ✓ 已加载术前模型: {cfg.PREOP_MODEL_PKL}")
        
        # 获取术前模型对相同数据的预测概率
        preop_model = preop_pkg['best_model']
        preop_threshold = preop_pkg['optimal_threshold']
        
        preop_log, preop_num, preop_cat = extract_feature_lists_from_pkg(preop_pkg)
        preop_features = preop_log + preop_num + preop_cat
        
        print(f"  ✓ 术前模型特征: {preop_features}")
        
        # 用术前特征子集预测（resubstitution）
        X_train_preop = train_df[preop_features].copy()
        X_ext_preop = external_df[preop_features].copy()
        for col in preop_num + preop_log:
            X_train_preop[col] = pd.to_numeric(X_train_preop[col], errors='coerce')
            X_ext_preop[col] = pd.to_numeric(X_ext_preop[col], errors='coerce')
        
        prob_train_preop = preop_model.predict_proba(X_train_preop)[:, 1]
        prob_ext_preop = preop_model.predict_proba(X_ext_preop)[:, 1]
        
        # 获取术前模型的 OOF 预测
        if 'predictions' in preop_pkg and isinstance(preop_pkg['predictions'], dict):
            preop_oof_probs = preop_pkg['predictions'].get('internal_oof_probs', None)
        if preop_oof_probs is None and 'results' in preop_pkg:
            preop_results = preop_pkg['results']
            preop_oof_probs = preop_results.get('prob_oof', None)
        
        # ---- 判断是否可用 OOF ----
        use_oof = (preop_oof_probs is not None and prob_oof_intraop is not None)
        
        # ---- 计算 NRI ----
        # 训练集 NRI：优先使用 OOF 预测（公平比较）
        if use_oof:
            print(f"  ✓ 训练集 NRI 使用 OOF 预测（公平比较）")
            nri_train = compute_nri(y_train, preop_oof_probs, prob_oof_intraop,
                                    threshold=preop_threshold)
        else:
            print(f"  ⚠️ OOF 预测不可用，训练集 NRI 使用 resubstitution（仅供参考）")
            nri_train = compute_nri(y_train, prob_train_preop, prob_train_intraop,
                                    threshold=preop_threshold)
        
        # 外部验证集 NRI：直接用模型预测（无泄露风险）
        nri_ext = compute_nri(y_ext, prob_ext_preop, prob_ext_intraop, 
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
        
        nri_available = True
        
    except FileNotFoundError:
        print(f"  ⚠️ 未找到术前模型文件: {cfg.PREOP_MODEL_PKL}")
        print(f"  请先运行 2_模型训练.py，或修改 PREOP_MODEL_PKL 路径")
        nri_available = False
        nri_train, nri_ext = None, None
        use_oof = False
    
    # --------------------------------------------------------
    # 5. 保存所有数据包
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("💾 保存敏感性分析数据包...")
    print(f"{'='*60}")
    
    # 生成术中模型的曲线数据（供可视化终端）
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_ext_arr = y_ext.values if hasattr(y_ext, 'values') else np.array(y_ext)
    
    curve_data_intraop = results.get('curve_data', {})
    
    # A. 术中模型完整数据包
    sensitivity_package = {
        'best_model': best_model_intraop,
        'best_model_name': best_name_intraop,
        'optimal_threshold': threshold_intraop,
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
            'train_probs': prob_train_intraop,
            'external_probs': prob_ext_intraop,
            'internal_oof_probs': prob_oof_intraop,  
        },
        # 新增曲线数据
        'curve_data': curve_data_intraop,
        'risk_stratification': risk_strat_intraop,
        'metadata': {
            'timestamp': timestamp,
            'analysis_type': 'sensitivity_intraoperative',
            'description': '敏感性分析：术前+术中变量模型',
            'internal_validation_method': 'OOF (Out-of-Fold)',  
        }
    }
    
    sens_pkl = f"{cfg.OUTPUT_DIR}/Sensitivity_Model_Package_{timestamp}.pkl"
    joblib.dump(sensitivity_package, sens_pkl)
    print(f"  ✓ 术中模型数据包: {sens_pkl}")
    
    # B. NRI 比较专用数据包（用于可视化终端）

    if nri_available:
        
        # --- 确定训练集使用的概率（OOF 优先，否则 resubstitution 兜底）---
        prob_train_preop_for_pkg = preop_oof_probs if use_oof else prob_train_preop
        prob_train_intraop_for_pkg = prob_oof_intraop if use_oof else prob_train_intraop
        
        # --- 基于实际使用概率计算训练集风险分层 ---
        risk_strat_train_preop = compute_risk_strata(
            y_train, prob_train_preop_for_pkg, preop_threshold)
        risk_strat_train_intraop = compute_risk_strata(
            y_train, prob_train_intraop_for_pkg, threshold_intraop)
        
        nri_comparison_package = {
            # 标签
            'y_train': np.array(y_train),
            'y_external': np.array(y_ext),
            
            # 术前模型（基线模型）
            'preop_model': {
                'name': 'Preoperative Model',
                # 训练集概率使用 OOF（与 NRI 计算一致）
                'prob_train': np.array(prob_train_preop_for_pkg),
                'prob_train_resub': np.array(prob_train_preop),   # 保留 resubstitution 备用
                'prob_external': np.array(prob_ext_preop),
                'threshold': preop_threshold,
                # 风险分层也基于 OOF 概率
                'risk_strat_train': risk_strat_train_preop,
                'risk_strat_external': compute_risk_strata(y_ext, prob_ext_preop, preop_threshold),
            },
            
            # 术中模型（增强模型）
            'intraop_model': {
                'name': 'Pre+Intraoperative Model',
                # 训练集概率使用 OOF（与 NRI 计算一致）
                'prob_train': np.array(prob_train_intraop_for_pkg),
                'prob_train_resub': np.array(prob_train_intraop),  # 保留 resubstitution 备用
                'prob_external': np.array(prob_ext_intraop),
                'threshold': threshold_intraop,
                # 风险分层也基于 OOF 概率
                'risk_strat_train': risk_strat_train_intraop,
                'risk_strat_external': risk_strat_intraop['external'],
            },
            
            # NRI/IDI 结果
            'nri_results': {
                'train': nri_train,
                'external': nri_ext
            },
            
            # AUC 对比
            'auc_comparison': {
                'train': {
                    'preop': roc_auc_score(y_train, prob_train_preop_for_pkg),
                    'intraop': roc_auc_score(y_train, prob_train_intraop_for_pkg),
                    'data_source': 'OOF' if use_oof else 'resubstitution',
                },
                # 保留 resubstitution AUC 供参考
                'train_resub': {
                    'preop': roc_auc_score(y_train, prob_train_preop),
                    'intraop': roc_auc_score(y_train, prob_train_intraop),
                    'data_source': 'resubstitution',
                },
                'external': {
                    'preop': roc_auc_score(y_ext, prob_ext_preop),
                    'intraop': roc_auc_score(y_ext, prob_ext_intraop),
                    'data_source': 'direct_prediction',
                }
            },
            
            'metadata': {
                'timestamp': timestamp,
                'preop_pkl_source': cfg.PREOP_MODEL_PKL,
                'description': 'NRI/IDI 比较数据包：术前 vs 术前+术中模型',
                # 明确标注训练集数据来源
                'train_data_source': 'OOF' if use_oof else 'resubstitution',
                'note': '训练集 prob_train / risk_strat_train / auc_comparison.train '
                        '均与 NRI 计算使用相同数据源（OOF 优先）；'
                        'resubstitution 版本保存在 _resub 后缀字段中备查。',
            }
        }
        
        nri_pkl = f"{cfg.OUTPUT_DIR}/NRI_Comparison_Data_{timestamp}.pkl"
        joblib.dump(nri_comparison_package, nri_pkl)
        print(f"  ✓ NRI 比较数据包: {nri_pkl}")
        print(f"    训练集数据源: {'OOF' if use_oof else 'resubstitution'}")
        
        # 同时保存 latest 版本
        joblib.dump(nri_comparison_package, f"{cfg.OUTPUT_DIR}/NRI_Comparison_Data_latest.pkl")
        joblib.dump(sensitivity_package, f"{cfg.OUTPUT_DIR}/Sensitivity_Model_Package_latest.pkl")
    
    print(f"\n{'='*60}")
    print("✅ 敏感性分析完成！")
    print(f"{'='*60}")