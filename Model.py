import os
os.environ['SCIPY_ARRAY_API'] = '1'  
import joblib
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # 必须先导入
from sklearn.impute import IterativeImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, 
                                     GridSearchCV, cross_val_predict)
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss, 
                             confusion_matrix, average_precision_score,
                             precision_recall_curve, f1_score, matthews_corrcoef)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import shap
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.base import clone
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel

os.makedirs('figures', exist_ok=True)
# ============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DPI = 600
# ============================================================================
# 3.1 配置类
# ============================================================================
class Config:
    
    TRAIN_DATA = 'Internal_setA_NaN.csv'      # 训练集路径
    EXTERNAL_DATA = 'External_setA_NaN.csv'     # 外部验证集路径
    
    # 关键列名
    TARGET_COL = 'Recurrence'                  # 目标变量列名
    
    # 训练参数
    RANDOM_STATE = 42
    CV_FOLDS = 5                               # 交叉验证折数
    CV_REPEATS = 20                             
    THRESHOLD_METHOD = 'youden'               # 阈值选取方法: 'youden', 'f1', 'sens90'
    
    # 输出配置
    OUTPUT_DIR = 'model_results'               # 结果保存目录

# ============================================================================
# 3.2 辅助函数：构建预处理器
# ============================================================================
def make_column_transformer(log_feats, num_feats, cat_feats):

    # 1. Log 转换流水线
    log_pipeline = Pipeline(steps=[
        ('log', FunctionTransformer(np.log1p, feature_names_out="one-to-one", validate=False)), 
        ('imputer', IterativeImputer(
            random_state=42,
            max_iter=10,
            min_value=0,
            initial_strategy='median',
            verbose=0
        )),
        ('scaler', StandardScaler())
    ])

    # 2. 连续变量流水线
    numeric_pipeline = Pipeline(steps=[
        ('imputer', IterativeImputer(
            random_state=42,
            max_iter=10,
            initial_strategy='median',
            verbose=0
        )),
        ('scaler', StandardScaler())
    ])

    # 3. 分类变量流水线（二分类变量只需插补，不需要编码）
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(
            strategy='most_frequent'  # 用众数填充缺失值
        ))
    ])

    # 4. 组合流水线
    preprocessor = ColumnTransformer(
        transformers=[
            ('log', log_pipeline, log_feats),
            ('num', numeric_pipeline, num_feats),
            ('cat', categorical_pipeline, cat_feats)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor
# ============================================================================
# 1. 数据加载与预处理
# ============================================================================
def load_and_preprocess_data(cfg):
    """加载数据并预处理，同时定义特征类型以便后续Pipeline使用"""
    
    print(f"\n{'='*60}")
    print("数据加载与特征类型定义")
    print(f"正在读取文件路径: {cfg.TRAIN_DATA}")
    print(f"{'='*60}")
    
    # 1. 加载原始数据
    train_df = pd.read_csv(cfg.TRAIN_DATA, index_col=0)
    external_df = pd.read_csv(cfg.EXTERNAL_DATA, index_col=0)
    
    # 2. 清洗：删除目标变量(Recurrence)为NaN的行
    train_df = train_df.dropna(subset=[cfg.TARGET_COL]).reset_index(drop=True)
    external_df = external_df.dropna(subset=[cfg.TARGET_COL]).reset_index(drop=True)
    
    y_train = train_df[cfg.TARGET_COL].astype(int)
    y_external = external_df[cfg.TARGET_COL].astype(int)

    # ============================================================
    # 3. 定义特征类型 (关键步骤)
    # ============================================================ 
    # A. 需要 Log 转换 + 标准化的连续变量 (偏态分布)
    log_features = ['Symptom_Duration'] 
    
    # B. 常规连续变量 (仅需标准化)
    numeric_features = ['AL', 'RD_Extent', 'number_of_breaks', 'Largest_Break_Diameter']
    
    # C. 分类变量 (需要 One-Hot 编码或保持原样)
    # 注意：即使在Excel里是0/1，放入Pipeline也建议视为分类变量处理，以防意外
    categorical_features = [ 
        'PVR_Grade_Pre', 
        'Break_Loc_Inferior',
        'Choroidal_Detachment',
        'Lens_Status_Pre',
        'Macular_status',
    ]
    
    # 合并所有特征
    all_features = log_features + numeric_features + categorical_features
    
    # ============================================================
    # 4. 验证特征存在性
    # ============================================================
    # 检查训练集是否包含所有定义的特征
    missing_features = [f for f in all_features if f not in train_df.columns]
    if missing_features:
        raise ValueError(f"❌ 训练集中缺少以下必要特征: {missing_features}")
    
    # 检查外部验证集是否包含所有定义的特征
    missing_ext = [f for f in all_features if f not in external_df.columns]
    if missing_ext:
        print(f"⚠️ 警告: 外部验证集缺少特征: {missing_ext}，后续预测可能会报错！")

    # ============================================================
    # 5. 构建 X 矩阵
    # ============================================================
    X_train = train_df[all_features].copy()
    X_external = external_df[all_features].copy()
    
    # 强制转换数据类型，防止 object 类型混入数值列导致报错
    for col in numeric_features + log_features:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_external[col] = pd.to_numeric(X_external[col], errors='coerce')
        
    # 确保 X 和 y 长度一致
    assert len(X_train) == len(y_train), "X_train 和 y_train 长度不一致"
    assert len(X_external) == len(y_external), "X_external 和 y_external 长度不一致"
    
    # ============================================================
    # 6. 打印信息
    # ============================================================
    print(f"\n特征分组信息:")
    print(f"  📌 Log转换变量 ({len(log_features)}): {log_features}")
    print(f"  📌 常规连续变量 ({len(numeric_features)}): {numeric_features}")
    print(f"  📌 分类/二值变量 ({len(categorical_features)}): {categorical_features}")
    print(f"  ✅ 总特征数: {len(all_features)}")
    
    print(f"\n数据集规模:")
    print(f"  训练集: {len(X_train)} 例 (事件数: {y_train.sum()}, {y_train.mean()*100:.1f}%)")
    print(f"  外部集: {len(X_external)} 例 (事件数: {y_external.sum()}, {y_external.mean()*100:.1f}%)")
    
    # 返回分类好的特征列表，供下游 Pipeline 使用
    return X_train, X_external, y_train, y_external, train_df, external_df, log_features, numeric_features, categorical_features

# ============================================================================
# 1. MetricsCalculator — 去除冗余，仅保留 Youden 阈值
# ============================================================================
class MetricsCalculator:

    @staticmethod
    def find_optimal_threshold(y_true, y_pred_proba):
        """
        使用 Youden's Index 确定最佳分类阈值
        公式: J = Sensitivity + Specificity - 1

        返回:
            best_threshold (float), metrics (dict)
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        n = len(y_true)

        thresholds = np.arange(0.01, 0.99, 0.01)
        best_j = -np.inf
        best_threshold = 0.5

        for t in thresholds:
            y_pred = (y_pred_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sensitivity + specificity - 1
            if j > best_j:
                best_j = j
                best_threshold = round(t, 2)

        # 计算最佳阈值下的指标
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            'threshold': best_threshold,
            'method': 'youden',
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'f1': f1_score(y_true, y_pred),
            'youden_index': best_j,
        }

        return best_threshold, metrics

    @staticmethod
    def calculate_metrics(y_true, y_prob, threshold, n_bootstrap=1000, random_state=42):
        """
        计算单阈值下的完整指标体系（符合PROBAST要求 + SCI审稿人推荐的不平衡数据指标）
        
        新增指标:
          - G-mean: sqrt(Sensitivity × Specificity), 不平衡数据的核心评估指标
          - Balanced Accuracy: (Sensitivity + Specificity) / 2
          - 所有关键指标的 Bootstrap 95% CI (非仅AUC)
        """
        y_true, y_prob = np.array(y_true), np.array(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        # 1. 区分度
        auc = roc_auc_score(y_true, y_prob)
        aucpr = average_precision_score(y_true, y_prob)

        # 2. 校准度
        brier = brier_score_loss(y_true, y_prob)

        y_prob_clip = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logit_prob = np.log(y_prob_clip / (1 - y_prob_clip))

        try:
            cal_clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            cal_clf.fit(logit_prob.reshape(-1, 1), y_true)
            cal_slope = cal_clf.coef_[0][0]
            cal_intercept = cal_clf.intercept_[0]
        except Exception as e:
            print(f"   ⚠️ 校准计算失败 ({e})")
            cal_slope = np.nan
            cal_intercept = np.nan

        # 3. 分类性能
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        n = len(y_true)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # 不平衡数据核心指标
        g_mean = np.sqrt(sens * spec)  # Geometric Mean
        balanced_acc = (sens + spec) / 2.0  # Balanced Accuracy

        # 4. 临床效用
        if 0 < threshold < 1:
            weight = threshold / (1 - threshold)
            net_benefit = (tp / n) - (fp / n) * weight
        else:
            net_benefit = 0.0

        # 5.Bootstrap 95% CI — 为所有关键指标计算CI
        boot_results = {
            'AUC': [], 'AUCPR': [], 'Brier': [],
            'Sensitivity': [], 'Specificity': [],
            'PPV': [], 'NPV': [], 'F1': [], 'MCC': [],
            'G_mean': [], 'Balanced_Acc': []
        }
        
        if n_bootstrap > 0:
            rng = np.random.RandomState(random_state)
            for _ in range(n_bootstrap):
                indices = rng.randint(0, n, n)
                y_t = y_true[indices]
                y_p = y_prob[indices]
                if len(np.unique(y_t)) < 2:
                    continue
                try:
                    boot_results['AUC'].append(roc_auc_score(y_t, y_p))
                    boot_results['AUCPR'].append(average_precision_score(y_t, y_p))
                    boot_results['Brier'].append(brier_score_loss(y_t, y_p))
                    
                    y_p_bin = (y_p >= threshold).astype(int)
                    b_tn, b_fp, b_fn, b_tp = confusion_matrix(y_t, y_p_bin).ravel()
                    b_sens = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0
                    b_spec = b_tn / (b_tn + b_fp) if (b_tn + b_fp) > 0 else 0
                    b_ppv = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0
                    b_npv = b_tn / (b_tn + b_fn) if (b_tn + b_fn) > 0 else 0
                    b_f1 = 2 * b_ppv * b_sens / (b_ppv + b_sens) if (b_ppv + b_sens) > 0 else 0
                    
                    boot_results['Sensitivity'].append(b_sens)
                    boot_results['Specificity'].append(b_spec)
                    boot_results['PPV'].append(b_ppv)
                    boot_results['NPV'].append(b_npv)
                    boot_results['F1'].append(b_f1)
                    boot_results['MCC'].append(matthews_corrcoef(y_t, y_p_bin))
                    boot_results['G_mean'].append(np.sqrt(b_sens * b_spec))
                    boot_results['Balanced_Acc'].append((b_sens + b_spec) / 2.0)
                except:
                    continue

        def _ci(vals):
            if vals:
                return np.percentile(vals, 2.5), np.percentile(vals, 97.5)
            return np.nan, np.nan

        return {
            'AUC': auc, 'AUC_95CI_Low': _ci(boot_results['AUC'])[0], 'AUC_95CI_High': _ci(boot_results['AUC'])[1],
            'AUCPR': aucpr, 'AUCPR_95CI_Low': _ci(boot_results['AUCPR'])[0], 'AUCPR_95CI_High': _ci(boot_results['AUCPR'])[1],
            'Brier': brier, 'Brier_95CI_Low': _ci(boot_results['Brier'])[0], 'Brier_95CI_High': _ci(boot_results['Brier'])[1],
            'Cal_Slope': cal_slope, 'Cal_Intercept': cal_intercept,
            'Threshold': threshold,
            'Sensitivity': sens, 'Sensitivity_95CI_Low': _ci(boot_results['Sensitivity'])[0], 'Sensitivity_95CI_High': _ci(boot_results['Sensitivity'])[1],
            'Specificity': spec, 'Specificity_95CI_Low': _ci(boot_results['Specificity'])[0], 'Specificity_95CI_High': _ci(boot_results['Specificity'])[1],
            'PPV': ppv, 'PPV_95CI_Low': _ci(boot_results['PPV'])[0], 'PPV_95CI_High': _ci(boot_results['PPV'])[1],
            'NPV': npv, 'NPV_95CI_Low': _ci(boot_results['NPV'])[0], 'NPV_95CI_High': _ci(boot_results['NPV'])[1],
            'F1': f1, 'F1_95CI_Low': _ci(boot_results['F1'])[0], 'F1_95CI_High': _ci(boot_results['F1'])[1],
            'MCC': mcc, 'MCC_95CI_Low': _ci(boot_results['MCC'])[0], 'MCC_95CI_High': _ci(boot_results['MCC'])[1],
            'G_mean': g_mean, 'G_mean_95CI_Low': _ci(boot_results['G_mean'])[0], 'G_mean_95CI_High': _ci(boot_results['G_mean'])[1],
            'Balanced_Acc': balanced_acc, 'Balanced_Acc_95CI_Low': _ci(boot_results['Balanced_Acc'])[0], 'Balanced_Acc_95CI_High': _ci(boot_results['Balanced_Acc'])[1],
            'Net_Benefit': net_benefit,
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'N_Total': n, 'N_Positive': int(tp + fn), 'N_Negative': int(tn + fp),
            # 【新增】保存 bootstrap 原始值，供可视化终端使用
            '_bootstrap_raw': boot_results
        }

    @staticmethod
    def print_metrics_summary(metrics_dict, dataset_name="Dataset"):
        print(f"\n{'='*60}")
        print(f"📊 {dataset_name} - Performance Metrics")
        print(f"{'='*60}")

        def _fmt_ci(key):
            lo = metrics_dict.get(f'{key}_95CI_Low', np.nan)
            hi = metrics_dict.get(f'{key}_95CI_High', np.nan)
            if np.isnan(lo) or np.isnan(hi):
                return ""
            return f" (95% CI: {lo:.4f}-{hi:.4f})"

        print(f"\n区分度 (Discrimination):")
        print(f"  AUC-ROC:    {metrics_dict['AUC']:.4f}{_fmt_ci('AUC')}")
        print(f"  AUPRC:      {metrics_dict['AUCPR']:.4f}{_fmt_ci('AUCPR')}")

        print(f"\n校准度 (Calibration):")
        print(f"  Brier:      {metrics_dict['Brier']:.4f}{_fmt_ci('Brier')}")
        print(f"  Cal Slope:  {metrics_dict['Cal_Slope']:.4f} (理想值=1)")
        print(f"  Cal Intercept: {metrics_dict['Cal_Intercept']:.4f} (理想值=0)")

        print(f"\n分类性能 (Threshold={metrics_dict['Threshold']:.4f}):")
        print(f"  Sensitivity:    {metrics_dict['Sensitivity']:.4f}{_fmt_ci('Sensitivity')}")
        print(f"  Specificity:    {metrics_dict['Specificity']:.4f}{_fmt_ci('Specificity')}")
        print(f"  PPV:            {metrics_dict['PPV']:.4f}{_fmt_ci('PPV')}")
        print(f"  NPV:            {metrics_dict['NPV']:.4f}{_fmt_ci('NPV')}")
        print(f"  F1 Score:       {metrics_dict['F1']:.4f}{_fmt_ci('F1')}")
        print(f"  MCC:            {metrics_dict['MCC']:.4f}{_fmt_ci('MCC')}")
        print(f"  G-mean:         {metrics_dict['G_mean']:.4f}{_fmt_ci('G_mean')}")
        print(f"  Balanced Acc:   {metrics_dict['Balanced_Acc']:.4f}{_fmt_ci('Balanced_Acc')}")

        print(f"\n临床效用: Net Benefit = {metrics_dict['Net_Benefit']:.4f}")
        print(f"\n混淆矩阵: TP={metrics_dict['TP']}, TN={metrics_dict['TN']}, "
              f"FP={metrics_dict['FP']}, FN={metrics_dict['FN']}")
        print(f"{'='*60}\n")


# ============================================================================
# 2. MultiModelTrainer 
# ============================================================================
class MultiModelTrainer:
    """
    符合 PROBAST 标准的多模型训练器
    """

    def __init__(self, preprocessor, random_state=42):
        self.preprocessor = preprocessor
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.all_results = {}
        self.calibrated_models = {}
        self.feature_selection_frequency = {}  # {model_name: {feature: count}}
        self.platt_calibrator = None  # 添加这一行

    # def calibrate_probs(self, raw_probs):
    #     """用训练好的Platt calibrator校准原始概率"""
    #     if self.platt_calibrator is None:
    #         raise ValueError("Platt calibrator未训练，请先调用train_all()")
        
    #     clipped = np.clip(raw_probs, 1e-10, 1 - 1e-10)
    #     logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    #     return self.platt_calibrator.predict_proba(logits)[:, 1]
    # ------------------------------------------------------------------
    # 2.1 模型配置
    # ------------------------------------------------------------------
    def get_model_configs(self, n_pos, n_neg):

        lr_common = {
            'class_weight': 'balanced',
            'max_iter': 10000,
            'random_state': self.random_state
        }

        feature_selector = SelectFromModel(
            LogisticRegression(
                penalty='l1', solver='liblinear',
                C=0.5, class_weight='balanced',
                max_iter=10000, random_state=self.random_state
            ),
            threshold='median'
        )


        configs = {
            # 'LR_LASSO': {
            #     'pipeline': Pipeline([
            #         ('preprocessor', clone(self.preprocessor)),
            #         ('classifier', LogisticRegression(
            #             penalty='l1', solver='liblinear', **lr_common))
            #     ]),
            #     'params': {
            #         'classifier__C': [0.1, 0.5, 1.0]
            #     }
            # },

            'LR_Ridge': {
                'pipeline': Pipeline([
                    ('preprocessor', clone(self.preprocessor)),
                    ('selector', clone(feature_selector)),
                    ('classifier', LogisticRegression(
                        penalty='l2', solver='lbfgs', **lr_common))
                ]),
                'params': {
                    'classifier__C': [0.1, 1.0]
                }
            },

            'RandomForest': {
                'pipeline': Pipeline([
                    ('preprocessor', clone(self.preprocessor)),
                    ('selector', clone(feature_selector)),
                    ('classifier', RandomForestClassifier(
                        n_estimators=100,
                        class_weight='balanced_subsample',
                        random_state=self.random_state,
                        n_jobs=-1
                    ))
                ]),
                'params': {
                    'classifier__max_depth': [2, 3],
                    'classifier__min_samples_leaf': [10]
                }
            },

            'SVM_RBF': {
                'pipeline': Pipeline([
                    ('preprocessor', clone(self.preprocessor)),
                    ('selector', clone(feature_selector)),
                    ('classifier', SVC(
                        kernel='rbf', class_weight='balanced',
                        probability=True, random_state=self.random_state
                    ))
                ]),
                'params': {
                    'classifier__C': [1.0, 5.0]
                }
            },
        }

        if XGBClassifier is not None:
            configs['XGBoost'] = {
                'pipeline': Pipeline([
                    ('preprocessor', clone(self.preprocessor)),
                    ('selector', clone(feature_selector)),
                    ('classifier', XGBClassifier(
                        random_state=self.random_state,
                        eval_metric='logloss',
                        n_estimators=100,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=1,
                        reg_lambda=5,
                        min_child_weight=5
                    ))
                ]),
                'params': {
                    'classifier__max_depth': [2, 3],
                    'classifier__scale_pos_weight': [
                        1, round(n_neg / n_pos)
                    ]
                }
            }

        return configs

    # ------------------------------------------------------------------
    # 2.2 从Pipeline中提取选中的特征名
    # ------------------------------------------------------------------
    def _extract_selected_features(self, fitted_pipeline):

        preprocessor = fitted_pipeline.named_steps['preprocessor']

        # 获取预处理后的特征名
        try:
            all_feature_names = np.array(preprocessor.get_feature_names_out())
        except AttributeError:
            all_feature_names = []
            for trans_name, transformer, columns in preprocessor.transformers_:
                if trans_name in ('drop', 'remainder'):
                    continue
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        all_feature_names.extend(transformer.get_feature_names_out(columns))
                    except:
                        all_feature_names.extend(columns if isinstance(columns, list) else [columns])
                else:
                    all_feature_names.extend(columns if isinstance(columns, list) else [columns])
            all_feature_names = np.array(all_feature_names)

        # 检查显式selector
        if 'selector' in fitted_pipeline.named_steps:
            selector = fitted_pipeline.named_steps['selector']
            mask = selector.get_support()
            # 维度对齐
            if len(mask) <= len(all_feature_names):
                all_feature_names = all_feature_names[:len(mask)]
            selected = list(all_feature_names[mask])
            return all_feature_names, mask, selected

        # 检查隐式选择（LASSO/ElasticNet的零系数）
        classifier = fitted_pipeline.named_steps['classifier']
        if hasattr(classifier, 'coef_'):
            coefs = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
            if len(coefs) <= len(all_feature_names):
                all_feature_names = all_feature_names[:len(coefs)]
            mask = np.abs(coefs) > 1e-10
            selected = list(all_feature_names[mask])
            return all_feature_names, mask, selected

        # 无筛选
        mask = np.ones(len(all_feature_names), dtype=bool)
        return all_feature_names, mask, list(all_feature_names)

    # ------------------------------------------------------------------
    # 2.3 记录特征选择频率
    # ------------------------------------------------------------------
    def _record_feature_selection(self, model_name, fold_idx, all_features, selected_mask):

        if model_name not in self.feature_selection_frequency:
            self.feature_selection_frequency[model_name] = {
                'feature_names': list(all_features),
                'fold_masks': [],       # 每个fold的选择掩码
                'n_folds': 0,
            }

        record = self.feature_selection_frequency[model_name]
        record['fold_masks'].append(selected_mask.copy())
        record['n_folds'] += 1

    def get_feature_frequency_df(self, model_name=None):

        results = []
        target_models = [model_name] if model_name else self.feature_selection_frequency.keys()

        for name in target_models:
            if name not in self.feature_selection_frequency:
                continue
            record = self.feature_selection_frequency[name]
            feat_names = record['feature_names']
            masks = np.array(record['fold_masks'])  # shape: (n_folds, n_features)
            n_folds = record['n_folds']

            if len(masks) == 0:
                continue

            # 每个特征被选中的次数
            counts = masks.sum(axis=0)

            for i, feat in enumerate(feat_names):
                results.append({
                    'Feature': feat,
                    'Selection_Count': int(counts[i]),
                    'Selection_Freq': counts[i] / n_folds,
                    'Total_Folds': n_folds,
                    'Model': name
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(['Model', 'Selection_Freq'], ascending=[True, False])
        return df

    # ------------------------------------------------------------------
    # 2.4 主训练方法 — 真正的嵌套CV
    # ------------------------------------------------------------------
    def train_all(self, X_train, y_train, X_external, y_external,
                  cv_folds=5, cv_repeats=20):
        print(f"\n{'='*60}")
        print(f"开始多模型训练 (嵌套CV: 外层 {cv_folds}×{cv_repeats}, 内层 {cv_folds}-fold)")
        print(f"{'='*60}")

        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        y_ext_arr = y_external.values if isinstance(y_external, pd.Series) else np.array(y_external)

        # 计算正负样本数
        n_pos = int(y_train_arr.sum())
        n_neg = int(len(y_train_arr) - n_pos)
        configs = self.get_model_configs(n_pos, n_neg)

        print(f"样本: {len(y_train_arr)} (正类={n_pos}, 负类={n_neg})")
        print(f"模型数: {len(configs)}")

        # 外层CV
        outer_cv = RepeatedStratifiedKFold(
            n_splits=cv_folds, n_repeats=cv_repeats, random_state=self.random_state
        )
        total_outer_folds = cv_folds * cv_repeats

        cv_results = {}

        # ==============================================================
        # 阶段1: 各模型嵌套CV评估
        # ==============================================================
        for name, cfg in configs.items():
            print(f"\n🚀 正在训练: {name} ...")
            try:
                oof_probs_sum = np.zeros(len(y_train_arr), dtype=np.float64)
                oof_counts = np.zeros(len(y_train_arr), dtype=np.int32)
                fold_aucs = []
                fold_best_params_list = []
                all_fold_metrics = []

                for fold_idx, (train_idx, val_idx) in enumerate(
                        outer_cv.split(X_train, y_train_arr)):

                    # --- 数据切分 ---
                    if isinstance(X_train, pd.DataFrame):
                        X_tr = X_train.iloc[train_idx]
                        X_val = X_train.iloc[val_idx]
                    else:
                        X_tr = X_train[train_idx]
                        X_val = X_train[val_idx]
                    y_tr = y_train_arr[train_idx]
                    y_val = y_train_arr[val_idx]

                    # --- 内层GridSearchCV仅在X_tr上执行 ---
                    inner_cv = StratifiedKFold(
                        n_splits=cv_folds, shuffle=True,
                        random_state=self.random_state + fold_idx
                    )

                    fold_pipeline = clone(cfg['pipeline'])

                    grid = GridSearchCV(
                        estimator=fold_pipeline,
                        param_grid=cfg['params'],
                        cv=inner_cv,
                        scoring='average_precision',  # 不平衡数据用AUPRC选模型
                        n_jobs=-1,
                        refit=True  # 用X_tr全量重训练最佳参数的模型
                    )
                    grid.fit(X_tr, y_tr)

                    fold_best_model = grid.best_estimator_
                    fold_best_params_list.append(grid.best_params_)

                    # --- 记录特征选择频率 ---
                    try:
                        all_feats, sel_mask, sel_feats = self._extract_selected_features(
                            fold_best_model)
                        self._record_feature_selection(name, fold_idx, all_feats, sel_mask)
                    except Exception as e:
                        pass  # 部分模型可能无法提取，静默跳过

                    # --- 在val上预测 ---
                    fold_probs = fold_best_model.predict_proba(X_val)[:, 1]
                    oof_probs_sum[val_idx] += fold_probs
                    oof_counts[val_idx] += 1

                    fold_auc = roc_auc_score(y_val, fold_probs)
                    fold_aucs.append(fold_auc)
                    
                    # 收集每折的完整不平衡数据指标
                    fold_pred = (fold_probs >= 0.5).astype(int)
                    fold_tn, fold_fp, fold_fn, fold_tp = confusion_matrix(y_val, fold_pred).ravel()
                    fold_sens = fold_tp / (fold_tp + fold_fn) if (fold_tp + fold_fn) > 0 else 0
                    fold_spec = fold_tn / (fold_tn + fold_fp) if (fold_tn + fold_fp) > 0 else 0
                    
                    all_fold_metrics.append({
                        'Fold': fold_idx + 1, 
                        'AUC': fold_auc,
                        'AUCPR': average_precision_score(y_val, fold_probs),
                        'MCC': matthews_corrcoef(y_val, fold_pred),
                        'G_mean': np.sqrt(fold_sens * fold_spec),
                        'F1': f1_score(y_val, fold_pred),
                        'Sensitivity': fold_sens,
                        'Specificity': fold_spec,
                    })

                    # 进度
                    if (fold_idx + 1) % (total_outer_folds // 4) == 0:
                        print(f"   ... fold {fold_idx+1}/{total_outer_folds}")

                # --- 汇总OOF ---
                valid_mask = oof_counts > 0
                prob_oof = np.zeros_like(oof_probs_sum)
                prob_oof[valid_mask] = oof_probs_sum[valid_mask] / oof_counts[valid_mask]

                auc_oof = roc_auc_score(y_train_arr, prob_oof)
                brier_oof = brier_score_loss(y_train_arr, prob_oof)
                ap_oof = average_precision_score(y_train_arr, prob_oof)
                
                # OOF整体的MCC和G-mean（使用默认0.5阈值先计算）
                oof_pred_default = (prob_oof >= 0.5).astype(int)
                mcc_oof = matthews_corrcoef(y_train_arr, oof_pred_default)
                oof_tn, oof_fp, oof_fn, oof_tp = confusion_matrix(y_train_arr, oof_pred_default).ravel()
                oof_sens = oof_tp / (oof_tp + oof_fn) if (oof_tp + oof_fn) > 0 else 0
                oof_spec = oof_tn / (oof_tn + oof_fp) if (oof_tn + oof_fp) > 0 else 0
                gmean_oof = np.sqrt(oof_sens * oof_spec)

                # --- 用全量X_train做一次GridSearchCV获取最终best_params ---
                final_grid = GridSearchCV(
                    estimator=clone(cfg['pipeline']),
                    param_grid=cfg['params'],
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                       random_state=self.random_state),
                    scoring='average_precision',  #与内层一致
                    n_jobs=-1,
                    refit=True,
                    return_train_score=True
                )
                final_grid.fit(X_train, y_train_arr)
                self.models[name] = final_grid.best_estimator_
                self.best_params[name] = final_grid.best_params_

                # 提取最终模型的特征选择信息
                all_feats_final, mask_final, selected_final = \
                    self._extract_selected_features(final_grid.best_estimator_)
                dropped_final = list(all_feats_final[~mask_final]) if mask_final is not None else []

                overfit_gap = (final_grid.cv_results_['mean_train_score'][final_grid.best_index_]
                               - final_grid.cv_results_['mean_test_score'][final_grid.best_index_])

                cv_results[name] = {
                    'model_obj': final_grid.best_estimator_,
                    'best_params': final_grid.best_params_,
                    'prob_oof': prob_oof,
                    'auc_oof': auc_oof,
                    'auc_oof_std': np.std(fold_aucs),
                    'brier_oof': brier_oof,
                    'ap_oof': ap_oof,
                    'mcc_oof': mcc_oof,       
                    'gmean_oof': gmean_oof,   
                    'cv_train_auc': final_grid.cv_results_['mean_train_score'][final_grid.best_index_],
                    'cv_val_auc': final_grid.cv_results_['mean_test_score'][final_grid.best_index_],
                    'all_fold_metrics': all_fold_metrics,
                    'overfit_gap': overfit_gap,
                    'all_features': list(all_feats_final),
                    'selected_features': selected_final,
                    'dropped_features': dropped_final,
                    'n_features_in': len(all_feats_final),
                    'n_features_selected': len(selected_final),
                    # 超参数稳定性
                    'fold_best_params': fold_best_params_list,
                }

                print(f"   ✅ 嵌套CV OOF AUC: {auc_oof:.4f} ± {np.std(fold_aucs):.4f} | "
                      f"AUPRC: {ap_oof:.4f} | MCC: {mcc_oof:.4f} | G-mean: {gmean_oof:.4f} | "
                      f"Brier: {brier_oof:.4f} | 特征: {len(selected_final)}/{len(all_feats_final)}")
                
                print(f"   📋 最佳参数: {fold_best_params_list[-1]}")
            except Exception as e:
                print(f"   ❌ {name} 训练失败: {e}")
                import traceback
                traceback.print_exc()

        # ==============================================================
        # 阶段1.5: Stacking 集成模型
        # ==============================================================
        print(f"\n{'='*60}")
        print("🔗 构建Stacking集成模型 (优先选择不同类型模型)")
        print(f"{'='*60}")

        try:
            # 定义模型类型，用于多样性筛选
            model_types = {
                'LR_Ridge': 'linear',
                'SVM_RBF': 'kernel',
                'RandomForest': 'tree',
                'XGBoost': 'boosting'
            }
            
            # 按OOF AUC降序排列
            sorted_models = sorted(cv_results.items(),
                                   key=lambda x: x[1]['auc_oof'], reverse=True)
            
            # 优先选择不同类型的top模型
            stack_names = []
            selected_types = set()
            
            for name, res in sorted_models:
                if len(stack_names) >= 3:  # 已选够3个，停止
                    break
                    
                model_type = model_types.get(name, 'other')
                
                # 优先选择新类型的模型
                if model_type not in selected_types:
                    stack_names.append(name)
                    selected_types.add(model_type)
            
            # 如果不同类型模型不足3个，补充同类型的高性能模型
            if len(stack_names) < 3:
                for name, res in sorted_models:
                    if name not in stack_names:
                        stack_names.append(name)
                        if len(stack_names) >= 3:
                            break

            print(f"   堆叠基模型: {stack_names}")
            print(f"   模型类型: {[model_types.get(n, 'other') for n in stack_names]}")

            # 构建StackingClassifier的estimators列表
            estimators = []
            for sname in stack_names:
                base_pipe = clone(self.models[sname])
                estimators.append((sname, base_pipe))

            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(
                    penalty='l2', C=1.0, max_iter=10000,
                    class_weight='balanced', random_state=self.random_state
                ),
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=self.random_state),
                stack_method='predict_proba',
                passthrough=False,  # 仅用基模型的预测概率，不传原始特征
                n_jobs=-1
            )

            # 对Stacking也做嵌套CV
            print("   正在对Stacking执行嵌套CV...")
            stack_oof_sum = np.zeros(len(y_train_arr), dtype=np.float64)
            stack_oof_counts = np.zeros(len(y_train_arr), dtype=np.int32)
            stack_fold_aucs = []
            stack_fold_metrics = []

            # 用较少的repeats加速（Stacking本身已经是集成）
            stack_outer_cv = RepeatedStratifiedKFold(
                n_splits=cv_folds, n_repeats=min(5, cv_repeats),
                random_state=self.random_state
            )

            for fold_idx, (train_idx, val_idx) in enumerate(
                    stack_outer_cv.split(X_train, y_train_arr)):

                if isinstance(X_train, pd.DataFrame):
                    X_tr = X_train.iloc[train_idx]
                    X_val = X_train.iloc[val_idx]
                else:
                    X_tr = X_train[train_idx]
                    X_val = X_train[val_idx]
                y_tr = y_train_arr[train_idx]
                y_val = y_train_arr[val_idx]

                fold_stack = clone(stacking_clf)
                fold_stack.fit(X_tr, y_tr)
                fold_probs = fold_stack.predict_proba(X_val)[:, 1]

                stack_oof_sum[val_idx] += fold_probs
                stack_oof_counts[val_idx] += 1

                fold_auc = roc_auc_score(y_val, fold_probs)
                stack_fold_aucs.append(fold_auc)
                stack_fold_metrics.append({'Fold': fold_idx + 1, 'AUC': fold_auc})

            # OOF汇总
            valid_mask = stack_oof_counts > 0
            stack_prob_oof = np.zeros_like(stack_oof_sum)
            stack_prob_oof[valid_mask] = stack_oof_sum[valid_mask] / stack_oof_counts[valid_mask]

            stack_auc_oof = roc_auc_score(y_train_arr, stack_prob_oof)
            stack_brier_oof = brier_score_loss(y_train_arr, stack_prob_oof)
            stack_ap_oof = average_precision_score(y_train_arr, stack_prob_oof)

            # 用全量数据训练最终Stacking模型
            final_stacking = clone(stacking_clf)
            final_stacking.fit(X_train, y_train_arr)
            self.models['Stacking'] = final_stacking

            # 存入cv_results
            # 计算Stacking的MCC和G-mean
            stack_pred_default = (stack_prob_oof >= 0.5).astype(int)
            stack_mcc_oof = matthews_corrcoef(y_train_arr, stack_pred_default)
            s_tn, s_fp, s_fn, s_tp = confusion_matrix(y_train_arr, stack_pred_default).ravel()
            s_sens = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0
            s_spec = s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0
            stack_gmean_oof = np.sqrt(s_sens * s_spec)
            
            cv_results['Stacking'] = {
                'model_obj': final_stacking,
                'best_params': {'base_models': stack_names},
                'prob_oof': stack_prob_oof,
                'auc_oof': stack_auc_oof,
                'auc_oof_std': np.std(stack_fold_aucs),
                'brier_oof': stack_brier_oof,
                'ap_oof': stack_ap_oof,
                'mcc_oof': stack_mcc_oof,       
                'gmean_oof': stack_gmean_oof,    
                'cv_train_auc': np.nan,  # Stacking无内层GridSearchCV
                'cv_val_auc': np.nan,
                'all_fold_metrics': stack_fold_metrics,
                'overfit_gap': 0.0,  # 无法直接计算
                'all_features': cv_results[stack_names[0]].get('all_features', []),
                'selected_features': [f"Stack({','.join(stack_names)})"],
                'dropped_features': [],
                'n_features_in': cv_results[stack_names[0]].get('n_features_in', 0),
                'n_features_selected': len(stack_names),
                'fold_best_params': [],
            }

            print(f"   ✅ Stacking OOF AUC: {stack_auc_oof:.4f} ± {np.std(stack_fold_aucs):.4f}")

        except Exception as e:
            print(f"   ❌ Stacking 构建失败: {e}")
            import traceback
            traceback.print_exc()
            
        # ==============================================================
        # 保存特征选择频率
        # ==============================================================
        print(f"\n{'='*60}")
        print("📋 特征选择频率统计")
        print(f"{'='*60}")

        freq_df = self.get_feature_frequency_df()
        if not freq_df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            freq_path = f"{Config.OUTPUT_DIR}/Feature_Selection_Frequency_{timestamp}.csv"
            freq_df.to_csv(freq_path, index=False, encoding='utf-8-sig')
            print(f"✅ 特征选择频率已保存: {freq_path}")

            # 打印各模型的频率摘要
            for model_name in freq_df['Model'].unique():
                sub = freq_df[freq_df['Model'] == model_name]
                n_folds = sub['Total_Folds'].iloc[0]
                high_freq = sub[sub['Selection_Freq'] >= 0.7]
                print(f"\n  {model_name} ({n_folds} folds):")
                print(f"    稳定选中(≥70%): {len(high_freq)} 个特征")
                if len(high_freq) > 0:
                    for _, row in high_freq.iterrows():
                        print(f"      {row['Feature']}: {row['Selection_Freq']:.1%} "
                              f"({row['Selection_Count']}/{n_folds})")

        # 同时生成特征-模型选择矩阵
        self._print_feature_selection_matrix(cv_results)

        # ==============================================================
        # 阶段2: 模型选择
        # ==============================================================
        print(f"\n{'='*60}")
        print("🏆 阶段2: 模型选择")
        print(f"{'='*60}")

        best_name, scores, valid_models, selection_reason = self._select_best_model(cv_results, y_train_arr)

        best_res = valid_models[best_name]
        reason = (
            f"{selection_reason} | "
            f"嵌套CV OOF AUC={best_res['auc_oof']:.4f}, "
            f"AP={best_res.get('ap_oof', 0):.4f}, "
            f"Brier={best_res['brier_oof']:.4f}, "
            f"MCC={best_res.get('mcc_oof', 0):.4f}"
        )
        print(f"\n✅ 选定最佳模型: {best_name}")
        print(f"   理由: {reason}")

        self._print_cv_comparison_table(valid_models)

        # ==============================================================
        # 阶段3: 阈值确定
        # ==============================================================
        print(f"\n{'='*60}")
        print("🎯 阶段3: 阈值确定（youden，基于嵌套CV OOF概率）")
        print(f"{'='*60}")

        best_oof_prob = cv_results[best_name]['prob_oof']
        best_threshold, threshold_metrics = MetricsCalculator.find_optimal_threshold(
            y_train_arr, best_oof_prob
        )
        self.best_threshold = best_threshold

        print(f"   最佳阈值: {best_threshold:.4f}")
        print(f"   Sensitivity: {threshold_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {threshold_metrics['specificity']:.4f}")
        print(f"   PPV: {threshold_metrics['ppv']:.4f}")
        print(f"   NPV: {threshold_metrics['npv']:.4f}")

        # ==============================================================
        # 阶段4: 用全量训练集训练最终模型（使用GridSearchCV已调好的模型）
        # ==============================================================
        print(f"\n{'='*60}")
        print("🔧 阶段4: 最终模型（已用全量训练集 GridSearchCV 训练完毕）")
        print(f"{'='*60}")

        # 注意：self.models[best_name] 已经是用全量X_train GridSearchCV训练的最佳模型
        final_pipeline = self.models[best_name]
        print(f"   模型: {best_name}")
        print(f"   参数: {self.best_params.get(best_name, 'N/A')}")

        # ==============================================================
        # 阶段5: 内部验证(OOF) + 外部验证
        # ==============================================================
        print(f"\n{'='*60}")
        print("📈 阶段5: 模型评估 - 内部验证(OOF) + 外部验证(泛化)")
        print(f"{'='*60}")

        # 内部验证：使用OOF预测（非resubstitution，避免过拟合偏差）
        prob_oof_best = cv_results[best_name]['prob_oof']
        prob_ext_final = final_pipeline.predict_proba(X_external)[:, 1]
        
        # 同时保留全量训练集预测（仅用于参考，不作为主要报告指标）
        prob_train_final = final_pipeline.predict_proba(X_train)[:, 1]

        train_metrics = MetricsCalculator.calculate_metrics(
            y_train_arr, prob_oof_best, best_threshold,
            n_bootstrap=1000, random_state=self.random_state
        )
        ext_metrics = MetricsCalculator.calculate_metrics(
            y_ext_arr, prob_ext_final, best_threshold,
            n_bootstrap=1000, random_state=self.random_state
        )

        MetricsCalculator.print_metrics_summary(train_metrics, "内部验证集（OOF预测）")
        MetricsCalculator.print_metrics_summary(ext_metrics, "外部验证集")
        
        # 保存 ROC / PR 曲线原始数据（供可视化终端绘图）
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        curve_data = {}
        for label, y_t, y_p in [('internal_oof', y_train_arr, prob_oof_best),
                                  ('external', y_ext_arr, prob_ext_final)]:
            fpr, tpr, roc_thresh = roc_curve(y_t, y_p)
            prec, rec, pr_thresh = precision_recall_curve(y_t, y_p)
            
            # 校准曲线数据
            try:
                prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=10, strategy='quantile')
            except:
                prob_true, prob_pred = np.array([]), np.array([])
            
            curve_data[label] = {
                'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresh},
                'pr': {'precision': prec, 'recall': rec, 'thresholds': pr_thresh},
                'calibration': {'prob_true': prob_true, 'prob_pred': prob_pred}
            }

        # ==============================================================
        # 组装 final_results
        # ==============================================================
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        final_results = {
            'best_model_name': best_name,
            'final_model': final_pipeline,
            'best_params': cv_results[best_name]['best_params'],
            'optimal_threshold': best_threshold,
            'threshold_method': 'youden',
            'cv_results': cv_results,
            'selection_reason': reason,
            'train_metrics': train_metrics,       # 基于OOF预测
            'external_metrics': ext_metrics,
            
            # 核心概率数据
            'prob_train': prob_train_final,        # 全量训练集resubstitution（仅参考）
            'prob_external': prob_ext_final,
            'prob_oof': cv_results[best_name]['prob_oof'],  # 内部验证主指标
            
            # 曲线原始数据（供可视化终端直接绘图）
            'curve_data': curve_data,
            
            # 移除校准器和校准后的概率，设为 None
            'platt_calibrator': None,
            'prob_external_calibrated': None,
            'prob_train_calibrated': None,
            'external_metrics_calibrated': None,
            
            # 风险分层 (仅保留原始概率的分层)
            'risk_stratification': {
                'train': self._compute_risk_strata(y_train_arr, prob_train_final, best_threshold),
                'external': self._compute_risk_strata(y_ext_arr, prob_ext_final, best_threshold)
            },
            
            # 内部验证方法标记
            'internal_validation_method': 'OOF (Out-of-Fold)',
        }

        return final_results

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _select_best_model(self, cv_results, y_train_arr):
        
        # 定义模型复杂度排名（数值越小越简单）
        simplicity_rank = {
            'LR_ElasticNet': 1, 'LR_Ridge': 2,
            'SVM_RBF': 3, 'RandomForest': 4, 'XGBoost': 5, 'Stacking': 6
        }

        # ==============================================================
        # Step 1: 过滤不稳定模型（标准：AUC 标准差 < 0.10）
        # ==============================================================
        valid_models_dict = {
            name: res for name, res in cv_results.items()
            if res['auc_oof_std'] < 0.10
        }

        if not valid_models_dict:
            print("⚠️ 所有模型稳定性欠佳，使用全部候选")
            valid_models_dict = cv_results

        # ==============================================================
        # Step 2: 过拟合安全阀（overfit_gap >= 0.15 的模型降级警告）
        # ==============================================================
        OVERFIT_THRESHOLD = 0.15
        overfit_warnings = set()
        
        for name, res in valid_models_dict.items():
            gap = res.get('overfit_gap', 0.0)
            if gap >= OVERFIT_THRESHOLD:
                overfit_warnings.add(name)
                print(f"  ⚠️ {name}: overfit_gap={gap:.4f} >= {OVERFIT_THRESHOLD} (泛化风险较高)")
        
        # 优先使用无过拟合风险的模型；如果全部都有风险，则保留全部
        safe_models_dict = {
            name: res for name, res in valid_models_dict.items()
            if name not in overfit_warnings
        }
        if safe_models_dict:
            working_dict = safe_models_dict
            if overfit_warnings:
                print(f"  ✓ 已将 {len(overfit_warnings)} 个过拟合风险模型降级，"
                      f"剩余 {len(safe_models_dict)} 个候选")
        else:
            print("  ⚠️ 所有模型均存在过拟合风险，保留全部候选（仅作为警告）")
            working_dict = valid_models_dict

        # ==============================================================
        # Step 3: 转换为 DataFrame 并构建排名
        # ==============================================================
        model_stats = []
        for name, res in working_dict.items():
            model_stats.append({
                'Model': name,
                'AUC': res['auc_oof'],
                'AP': res.get('ap_oof', 0),
                'MCC': res.get('mcc_oof', 0),
                'G_mean': res.get('gmean_oof', 0),
                'Brier': res['brier_oof'],
                'Overfit_Gap': res.get('overfit_gap', 0.0),
                'Simplicity': simplicity_rank.get(name, 99)
            })
        df = pd.DataFrame(model_stats)

        # ==============================================================
        # Step 4: AUC 第一梯队（与最高 AUC 差距 < 0.01）
        # ==============================================================
        best_auc = df['AUC'].max()
        tier_1 = df[df['AUC'] >= (best_auc - 0.01)].copy()

        # ==============================================================
        # Step 5: 梯队内按 AP 排序（不平衡数据关键指标）
        # ==============================================================
        tier_1 = tier_1.sort_values(by='AP', ascending=False).reset_index(drop=True)
        best_ap = tier_1.iloc[0]['AP']
        ap_candidates = tier_1[tier_1['AP'] >= (best_ap - 0.02)].copy()

        # ==============================================================
        # Step 6: AP 接近时，比较 Brier Score（校准度）
        # ==============================================================
        ap_candidates = ap_candidates.sort_values(by='Brier', ascending=True).reset_index(drop=True)
        best_brier = ap_candidates.iloc[0]['Brier']
        brier_candidates = ap_candidates[ap_candidates['Brier'] <= (best_brier + 0.002)].copy()

        # ==============================================================
        # Step 7: Brier 接近时，比较 MCC（不平衡数据分类质量）
        # MCC 取值 [-1, 1]，越大越好
        # ==============================================================
        if len(brier_candidates) > 1:
            brier_candidates = brier_candidates.sort_values(by='MCC', ascending=False).reset_index(drop=True)
            best_mcc = brier_candidates.iloc[0]['MCC']
            mcc_candidates = brier_candidates[brier_candidates['MCC'] >= (best_mcc - 0.01)].copy()
        else:
            mcc_candidates = brier_candidates.copy()

        # ==============================================================
        # Step 8: MCC 仍接近，选最简单模型（奥卡姆剃刀）
        # ==============================================================
        if len(mcc_candidates) > 1:
            mcc_candidates = mcc_candidates.sort_values(by='Simplicity', ascending=True)
            best_name = mcc_candidates.iloc[0]['Model']
            selection_reason = "AUC/AP/Brier/MCC均接近，选最简模型(奥卡姆剃刀)"
        elif len(brier_candidates) > 1 and len(mcc_candidates) == 1:
            best_name = mcc_candidates.iloc[0]['Model']
            selection_reason = "AUC/AP/Brier接近，MCC最优(不平衡数据分类质量)"
        elif len(ap_candidates) > 1 and len(brier_candidates) == 1:
            best_name = brier_candidates.iloc[0]['Model']
            selection_reason = "AUC/AP接近，Brier最优(校准度)"
        elif len(tier_1) > 1 and len(ap_candidates) == 1:
            best_name = ap_candidates.iloc[0]['Model']
            selection_reason = "AUC接近，AP最优(少数类识别)"
        else:
            best_name = tier_1.iloc[0]['Model']
            selection_reason = "AUC最优"

        # 补充过拟合降级信息
        if overfit_warnings and best_name not in overfit_warnings:
            selection_reason += " [已排除过拟合风险模型]"

        # ==============================================================
        # 打印排名表
        # ==============================================================
        print(f"\n{'Model':<20} | {'AUC':<8} | {'AP':<8} | {'Brier':<8} | "
              f"{'MCC':<8} | {'OvfitGap':<9} | {'Simple':<7} | Status")
        print("-" * 110)

        sorted_models = sorted(cv_results.keys(),
                               key=lambda x: cv_results[x]['auc_oof'],
                               reverse=True)

        for name in sorted_models:
            res = cv_results[name]
            simple_score = simplicity_rank.get(name, 99)
            gap = res.get('overfit_gap', 0.0)

            flags = []
            if name == best_name:
                flags.append(f"★ [{selection_reason}]")
            if res['auc_oof_std'] >= 0.10:
                flags.append("⚠️Unstable")
            if name in overfit_warnings:
                flags.append(f"⚠️Overfit({gap:.3f})")
            if name not in tier_1['Model'].values and name in working_dict:
                flags.append("非第一梯队")
            if name not in working_dict and name in valid_models_dict:
                flags.append("已降级")
            status_str = " | ".join(flags) if flags else ""

            print(f"{name:<20} | {res['auc_oof']:<8.4f} | "
                  f"{res.get('ap_oof', 0):<8.4f} | {res['brier_oof']:<8.4f} | "
                  f"{res.get('mcc_oof', 0):<8.4f} | {gap:<9.4f} | "
                  f"{simple_score:<7} | {status_str}")

        print("-" * 110)

        # 返回结果（新增 selection_reason）
        scores = {name: res['auc_oof'] for name, res in cv_results.items()}

        return best_name, scores, valid_models_dict, selection_reason

    def _print_feature_selection_matrix(self, cv_results):
        """打印特征选择矩阵"""
        all_unique_features = sorted(set(
            feat for res in cv_results.values() for feat in res.get('all_features', [])
        ))
        if not all_unique_features:
            return

        matrix = pd.DataFrame(index=all_unique_features)
        for model_name, res in cv_results.items():
            if model_name == 'Stacking':
                continue
            matrix[model_name] = [
                '✓' if feat in res.get('selected_features', []) else '✗'
                for feat in all_unique_features
            ]

        matrix['选中次数'] = matrix.apply(lambda row: sum(1 for v in row if v == '✓'), axis=1)
        matrix = matrix.sort_values('选中次数', ascending=False)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        matrix.to_csv(f"{Config.OUTPUT_DIR}/Feature_Selection_Matrix_{timestamp}.csv", encoding='utf-8-sig')
        print(f"\n特征选择矩阵:")
        print(matrix.to_string())

    def _print_cv_comparison_table(self, cv_results_dict):
        """打印CV性能对比表"""
        print(f"\n{'='*100}")
        print("📊 所有模型CV性能对比")
        print(f"{'='*100}")

        table_data = []
        for name, res in cv_results_dict.items():
            table_data.append({
                'Model': name,
                'OOF_AUC': f"{res['auc_oof']:.4f}",
                'CV_Std': f"{res['auc_oof_std']:.4f}",
                'AP_Score': f"{res['ap_oof']:.4f}",
                'Brier': f"{res['brier_oof']:.4f}",
                'MCC': f"{res.get('mcc_oof', 0):.4f}",
                'G_mean': f"{res.get('gmean_oof', 0):.4f}",
                'Overfit_Gap': f"{res.get('overfit_gap', 0.0):.4f}",
                'N_Features': f"{res['n_features_selected']}/{res['n_features_in']}"
            })

        df = pd.DataFrame(table_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        df.to_csv(f"{Config.OUTPUT_DIR}/Model_CV_Comparison_{timestamp}.csv", index=False, encoding='utf-8-sig')
        print(df.to_string(index=False))
        print(f"{'='*100}\n")

    @staticmethod
    def _compute_risk_strata(y_true, y_prob, threshold, n_groups=3):
        """计算风险分层统计"""
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
                    stats[g] = {'n': 0, 'events': 0, 'event_rate': 0,
                                'mean_prob': 0, 'median_prob': 0}
                    continue
                stats[g] = {
                    'n': int(n_total),
                    'events': int(y_true[mask].sum()),
                    'non_events': int(n_total - y_true[mask].sum()),
                    'event_rate': float(y_true[mask].mean()),
                    'mean_prob': float(y_prob[mask].mean()),
                    'median_prob': float(np.median(y_prob[mask])),
                    'prob_range': [float(y_prob[mask].min()), float(y_prob[mask].max())]
                }
            return stats

        # Hosmer-Lemeshow
        try:
            from scipy.stats import chi2
            n_hl = 10
            sorted_idx = np.argsort(y_prob)
            groups_hl = np.array_split(sorted_idx, n_hl)
            hl_stat = 0
            hl_table = []
            for g_idx in groups_hl:
                obs_e = y_true[g_idx].sum()
                exp_e = y_prob[g_idx].sum()
                obs_n = len(g_idx) - obs_e
                exp_n = len(g_idx) - exp_e
                hl_table.append({'n': len(g_idx), 'obs_events': int(obs_e),
                                 'exp_events': float(exp_e)})
                if exp_e > 0:
                    hl_stat += (obs_e - exp_e) ** 2 / exp_e
                if exp_n > 0:
                    hl_stat += (obs_n - exp_n) ** 2 / exp_n
            hl_p = 1 - chi2.cdf(hl_stat, n_hl - 2)
        except Exception:
            hl_stat, hl_p, hl_table = np.nan, np.nan, []

        return {
            'y_true': y_true, 'y_prob': y_prob,
            'tertile': {'cuts': tertile_cuts.tolist(), 'groups': risk_group_tertile,
                        'labels': ['Low', 'Medium', 'High'],
                        'stats': group_stats(risk_group_tertile, 3)},
            'binary': {'threshold': float(threshold), 'groups': risk_group_binary,
                       'labels': ['Low Risk', 'High Risk'],
                       'stats': group_stats(risk_group_binary, 2)},
            'fixed': {'cuts': fixed_cuts, 'groups': risk_group_fixed,
                      'labels': ['Low (<10%)', 'Medium (10-30%)', 'High (>30%)'],
                      'stats': group_stats(risk_group_fixed, 3)},
            'hosmer_lemeshow': {'statistic': float(hl_stat), 'p_value': float(hl_p),
                                'table': hl_table}
        }
# ============================================================================
# 3.3 辅助函数：生成并保存预测CSV 
# ============================================================================
def save_prediction_csv(model, X, y, threshold, output_path, dataset_name="Data"):

    # 1. 计算概率
    # predict_proba 返回 [class0_prob, class1_prob]，我们需要 class1
    probs = model.predict_proba(X)[:, 1]
    
    # 2. 根据最佳阈值生成预测类别 (0或1)
    preds = (probs >= threshold).astype(int)
    
    # 3. 构建DataFrame
    result_df = pd.DataFrame({
        'True_Label': y,                # 真实标签
        'Pred_Prob': probs,             # 预测概率 (0-1)
        'Pred_Label': preds,            # 预测结果 (基于阈值)
        'Risk_Group': ['High' if p == 1 else 'Low' for p in preds] # 风险分组文本
    }, index=X.index)
    
    # 4. 保存
    result_df.to_csv(output_path, index_label='Patient_ID', encoding='utf-8-sig')
    print(f"   > [{dataset_name}] 预测详情已保存: {os.path.basename(output_path)}")
    return result_df

def save_detailed_results(best_model_name, results, X_train, y_train, X_ext, y_ext, timestamp):
    """
    专门用于保存后续绘图所需的高级分析数据
    """
    output_dir = Config.OUTPUT_DIR
    
    # ---------------------------------------------------------
    # 1. 保存 Internal_Val_OOF_Preds (用于内部验证 ROC/校准/DCA)
    # ---------------------------------------------------------
    # 从 trainer 的结果中提取最佳模型的 OOF 预测值
    # 假设 results['cv_results'][best_model_name] 包含 'oof_preds'
    if 'oof_preds' in results['cv_results'][best_model_name]:
        oof_probs = results['cv_results'][best_model_name]['oof_preds']
        oof_df = pd.DataFrame({
            'Patient_ID': X_train.index,
            'True_Label': y_train,
            'Pred_Prob': oof_probs
        })
        oof_path = f"{output_dir}/Internal_Val_OOF_Preds_{best_model_name}_{timestamp}.csv"
        oof_df.to_csv(oof_path, index=False, encoding='utf-8-sig')
        print(f"✅ 内部验证 OOF 预测值已保存 (用于训练集校准/DCA): {oof_path}")

    # ---------------------------------------------------------
    # 2. 保存 External_Val_Preds (用于外部验证 ROC/校准/DCA)
    # ---------------------------------------------------------
    # 使用训练好的最终模型对外部集进行预测
    ext_probs = results['final_model'].predict_proba(X_ext)[:, 1]
    ext_df = pd.DataFrame({
        'Patient_ID': X_ext.index,
        'True_Label': y_ext,
        'Pred_Prob': ext_probs
    })
    ext_path = f"{output_dir}/External_Val_Preds_{best_model_name}_{timestamp}.csv"
    ext_df.to_csv(ext_path, index=False, encoding='utf-8-sig')
    print(f"✅ 外部验证集预测值已保存 (用于泛化能力验证): {ext_path}")

    # ---------------------------------------------------------
    # 3. 保存 Internal_Val_CV_Metrics (用于 AUC 置信区间/箱线图)
    # ---------------------------------------------------------
    # 提取每一折的指标
    if 'cv_scores' in results['cv_results'][best_model_name]:
        cv_scores = results['cv_results'][best_model_name]['cv_scores'] # 预期是一个 list 或 array
        metrics_df = pd.DataFrame({
            'Fold': range(1, len(cv_scores) + 1),
            'AUC': cv_scores
        })
        metrics_path = f"{output_dir}/Internal_Val_CV_Metrics_{best_model_name}_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"✅ 交叉验证指标已保存 (用于统计检验/箱线图): {metrics_path}")

# --------------------------------------------------------
# 3.4 主程序入口
# --------------------------------------------------------
if __name__ == "__main__":
    # 0. 准备目录
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        
    print(f"{'='*60}")
    print(f"🚀 视网膜脱离复发预测模型训练任务启动")
    print(f"🕒 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        # --------------------------------------------------------
        # 1. 加载数据 & 定义特征
        # --------------------------------------------------------
        X_train, X_ext, y_train, y_ext, train_df, external_df, log_feats, num_feats, cat_feats = load_and_preprocess_data(Config)

        # --------------------------------------------------------
        # 2. 构建预处理器
        # --------------------------------------------------------
        preprocessor = make_column_transformer(log_feats, num_feats, cat_feats)

        # --------------------------------------------------------
        # 3. 训练与评估
        # --------------------------------------------------------
        trainer = MultiModelTrainer(preprocessor, random_state=Config.RANDOM_STATE)
        
        results = trainer.train_all(
            X_train, y_train, 
            X_ext, y_ext, 
            cv_folds=Config.CV_FOLDS, 
            cv_repeats=Config.CV_REPEATS
        )

        # 获取关键结果
        best_model = results['final_model']
        optimal_thresh = results['optimal_threshold']
        best_model_name = results['best_model_name']

        # --------------------------------------------------------
        # 4. 保存可视化所需的模型包
        # --------------------------------------------------------
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        viz_package_path = f"{Config.OUTPUT_DIR}/Model_Package_{timestamp}.pkl"
        
        print(f"\n{'='*60}")
        print("💾 保存可视化模型包...")
        print(f"{'='*60}")
        
        # 构建可视化脚本需要的数据包
        viz_package = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'optimal_threshold': optimal_thresh,
            'preprocessor': preprocessor,
            'trainer': trainer,  
            'trained_models': trainer.models,
            'results': results,
            'datasets': {                                # 数据集
                'X_train': X_train,
                'y_train': y_train,
                'X_ext': X_ext,
                'y_ext': y_ext
            },
            'feature_info': {                            # 特征信息
                'log_features': log_feats,
                'num_features': num_feats,
                'cat_features': cat_feats
            },
            # 【新增】曲线原始数据（供可视化终端直接绘图）
            'curve_data': results.get('curve_data', {}),
            'metadata': {
                'timestamp': timestamp,
                'random_state': Config.RANDOM_STATE,
                'internal_validation_method': 'OOF (Out-of-Fold)',  
            }
        }
        
        try:
            joblib.dump(viz_package, viz_package_path)
            print(f"✅ [保存成功] 可视化模型包: {viz_package_path}")
            print(f"   - 最佳模型: {best_model_name}")
            print(f"   - 最优阈值: {optimal_thresh:.4f}")
            print(f"   - 包含模型数: {len(trainer.models)}")
        except Exception as e:
            print(f"❌ [保存失败] {str(e)}")
            raise
        
        # --------------------------------------------------------
        # 4. 外部验证 Bootstrap 稳定性分析 (1000次)
        # --------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"🔄 正在进行外部验证 Bootstrap 稳定性分析 (n=1000)...")
        print(f"   模型: {best_model_name}")
        print(f"{'='*60}")
        
        from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
        from sklearn.utils import resample
        
        n_bootstrap = 1000
        bootstrap_metrics = {
            'AUC': [], 'Brier': [], 'AP': [], 
            'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': [],
            'F1': [], 'MCC': [], 'G_mean': [], 'Balanced_Acc': []  
        }
        
        # 保存每次 Bootstrap 的预测概率（用于后续 ROC 曲线绘制）
        bootstrap_predictions = []
        
        np.random.seed(Config.RANDOM_STATE)
        
        for i in range(n_bootstrap):
            # 有放回抽样
            indices = resample(np.arange(len(X_ext)), random_state=i, stratify=y_ext)
            X_boot = X_ext.iloc[indices]
            y_boot = y_ext.iloc[indices]
            
            y_pred_proba = best_model.predict_proba(X_boot)[:, 1]
            y_pred_label = (y_pred_proba >= optimal_thresh).astype(int)
            
            # 保存预测结果（用于后续可视化）
            bootstrap_predictions.append({
                'iteration': i,
                'true_labels': y_boot.values,
                'pred_probs': y_pred_proba,
                'indices': indices
            })
            
            # 计算指标
            bootstrap_metrics['AUC'].append(roc_auc_score(y_boot, y_pred_proba))
            bootstrap_metrics['Brier'].append(brier_score_loss(y_boot, y_pred_proba))
            bootstrap_metrics['AP'].append(average_precision_score(y_boot, y_pred_proba))
            
            # 混淆矩阵指标
            tn = np.sum((y_boot.values == 0) & (y_pred_label == 0))
            tp = np.sum((y_boot.values == 1) & (y_pred_label == 1))
            fn = np.sum((y_boot.values == 1) & (y_pred_label == 0))
            fp = np.sum((y_boot.values == 0) & (y_pred_label == 1))
            
            bootstrap_metrics['Sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            bootstrap_metrics['Specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            bootstrap_metrics['PPV'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            bootstrap_metrics['NPV'].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            
            # 不平衡数据核心指标
            b_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            b_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            b_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            b_f1 = 2 * b_ppv * b_sens / (b_ppv + b_sens) if (b_ppv + b_sens) > 0 else 0
            bootstrap_metrics['F1'].append(b_f1)
            bootstrap_metrics['MCC'].append(matthews_corrcoef(y_boot, y_pred_label))
            bootstrap_metrics['G_mean'].append(np.sqrt(b_sens * b_spec))
            bootstrap_metrics['Balanced_Acc'].append((b_sens + b_spec) / 2.0)
            
            if (i + 1) % 100 == 0:
                print(f"  ✓ 已完成 {i+1}/{n_bootstrap} 次 Bootstrap 采样")
        
        # 计算 95% CI
        bootstrap_summary = {}
        for metric, values in bootstrap_metrics.items():
            bootstrap_summary[metric] = {
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Median': np.median(values),
                '95%_CI_Lower': np.percentile(values, 2.5),
                '95%_CI_Upper': np.percentile(values, 97.5)
            }
        
        # 保存 Bootstrap 结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        bootstrap_df = pd.DataFrame(bootstrap_summary).T
        bootstrap_df.to_csv(f"{Config.OUTPUT_DIR}/External_Val_Bootstrap_1000_{timestamp}.csv", encoding='utf-8-sig')
        
        # 保存所有 Bootstrap 原始值（用于后续绘制置信区间）
        bootstrap_raw_df = pd.DataFrame(bootstrap_metrics)
        bootstrap_raw_df.to_csv(f"{Config.OUTPUT_DIR}/External_Val_Bootstrap_RawData_{timestamp}.csv", index=False, encoding='utf-8-sig')
        
        # 保存 Bootstrap 预测数据（用于 ROC 曲线置信带绘制）
        joblib.dump(bootstrap_predictions, f"{Config.OUTPUT_DIR}/Bootstrap_Predictions_{timestamp}.pkl")
        
        print(f"\n📊 Bootstrap 稳定性分析完成!")
        print(f"  - AUC: {bootstrap_summary['AUC']['Mean']:.3f} (95% CI: {bootstrap_summary['AUC']['95%_CI_Lower']:.3f}-{bootstrap_summary['AUC']['95%_CI_Upper']:.3f})")

        # --------------------------------------------------------
        # 5.5 保存综合指标汇总表
        # --------------------------------------------------------
        print(f"\n{'='*60}")
        print("📋 保存综合指标汇总表 (论文 Table 专用)...")
        print(f"{'='*60}")
        
        # 内部验证指标 (基于OOF)
        internal_metrics = results['train_metrics']
        external_metrics = results['external_metrics']
        
        def _fmt_metric_row(metrics, name):
            """格式化单行指标：Value (95% CI)"""
            rows = {}
            for key in ['AUC', 'AUCPR', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                        'F1', 'MCC', 'G_mean', 'Balanced_Acc', 'Brier']:
                val = metrics.get(key, np.nan)
                lo = metrics.get(f'{key}_95CI_Low', np.nan)
                hi = metrics.get(f'{key}_95CI_High', np.nan)
                if np.isnan(lo) or np.isnan(hi):
                    rows[key] = f"{val:.4f}"
                else:
                    rows[key] = f"{val:.4f} ({lo:.4f}-{hi:.4f})"
            rows['Dataset'] = name
            rows['Threshold'] = f"{metrics.get('Threshold', np.nan):.4f}"
            rows['Cal_Slope'] = f"{metrics.get('Cal_Slope', np.nan):.4f}"
            rows['Cal_Intercept'] = f"{metrics.get('Cal_Intercept', np.nan):.4f}"
            rows['Net_Benefit'] = f"{metrics.get('Net_Benefit', np.nan):.4f}"
            rows['N_Total'] = metrics.get('N_Total', '')
            rows['N_Positive'] = metrics.get('N_Positive', '')
            return rows
        
        summary_rows = [
            _fmt_metric_row(internal_metrics, 'Internal (OOF)'),
            _fmt_metric_row(external_metrics, 'External'),
        ]
        summary_df = pd.DataFrame(summary_rows)
        
        # 重新排列列顺序
        col_order = ['Dataset', 'N_Total', 'N_Positive', 'AUC', 'AUCPR', 
                     'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                     'F1', 'MCC', 'G_mean', 'Balanced_Acc',
                     'Brier', 'Cal_Slope', 'Cal_Intercept', 'Net_Benefit', 'Threshold']
        summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]
        
        summary_path = f"{Config.OUTPUT_DIR}/Comprehensive_Metrics_Summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ 综合指标汇总表已保存: {summary_path}")
        print(f"     可直接用于论文 Table")
        
        # 同时保存所有模型的OOF对比表
        all_models_summary = []
        for mname, mres in results['cv_results'].items():
            all_models_summary.append({
                'Model': mname,
                'OOF_AUC': f"{mres['auc_oof']:.4f} ± {mres['auc_oof_std']:.4f}",
                'OOF_AUPRC': f"{mres['ap_oof']:.4f}",
                'OOF_MCC': f"{mres.get('mcc_oof', 0):.4f}",
                'OOF_G_mean': f"{mres.get('gmean_oof', 0):.4f}",
                'OOF_Brier': f"{mres['brier_oof']:.4f}",
                'N_Features': f"{mres['n_features_selected']}/{mres['n_features_in']}",
            })
        all_models_df = pd.DataFrame(all_models_summary)
        all_models_path = f"{Config.OUTPUT_DIR}/All_Models_OOF_Comparison_{timestamp}.csv"
        all_models_df.to_csv(all_models_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ 所有模型OOF对比表已保存: {all_models_path}")

        # --------------------------------------------------------
        # 6. 保存高级分析数据 (ROC/DCA/Calibration/SHAP)
        # --------------------------------------------------------
        print(f"\n{'='*60}")
        print("📊 正在准备高级分析数据 (ROC/DCA/Calibration/SHAP)...")
        print(f"{'='*60}")

        oof_probs = results['cv_results'][best_model_name]['prob_oof']

        # A. 内部验证 OOF 预测值 (用于内部校准、DCA)
        df_internal_oof = pd.DataFrame({
            'Patient_ID': X_train.index,
            'True_Label': y_train,
            'Pred_Prob': oof_probs
        })
        df_internal_oof.to_csv(f"{Config.OUTPUT_DIR}/Internal_Val_OOF_Preds_{best_model_name}_{timestamp}.csv", index=False, encoding='utf-8-sig')

        # B. 外部验证预测值
        ext_probs_final = best_model.predict_proba(X_ext)[:, 1]
        df_external = pd.DataFrame({
            'Patient_ID': X_ext.index,
            'True_Label': y_ext,
            'Pred_Prob': ext_probs_final
        })
        df_external.to_csv(f"{Config.OUTPUT_DIR}/External_Val_Preds_{best_model_name}_{timestamp}.csv", index=False, encoding='utf-8-sig')
        
        # 保存风险分层详细 CSV（用于临床分析）
        risk_strat = results.get('risk_stratification', {})
        if risk_strat:
            for dataset_name, strat_data in [('Internal', risk_strat.get('train')), 
                                              ('External', risk_strat.get('external'))]:
                if strat_data is None:
                    continue
                strat_df = pd.DataFrame({
                    'True_Label': strat_data['y_true'],
                    'Pred_Prob': strat_data['y_prob'],
                    'Risk_Tertile': strat_data['tertile']['groups'],
                    'Risk_Binary': strat_data['binary']['groups'],
                    'Risk_Fixed': strat_data['fixed']['groups']
                })
                strat_df.to_csv(
                    f"{Config.OUTPUT_DIR}/Risk_Stratification_{dataset_name}_{timestamp}.csv",
                    index=False, encoding='utf-8-sig'
                )
            print(f"  ✓ 风险分层 CSV 已保存")

        # C. 交叉验证指标明细 (带容错)
        metrics_key = 'all_fold_metrics' if 'all_fold_metrics' in results['cv_results'][best_model_name] else 'cv_scores'
        cv_metrics_data = results['cv_results'][best_model_name].get(metrics_key, [])
        cv_metrics_df = pd.DataFrame(cv_metrics_data) if isinstance(cv_metrics_data, (list, dict)) else pd.DataFrame({'Score': cv_metrics_data})
        cv_metrics_df.to_csv(f"{Config.OUTPUT_DIR}/Internal_Val_CV_Metrics_Detail_{timestamp}.csv", index=False, encoding='utf-8-sig')

        # D. 错误病例深度分析
        error_df = df_internal_oof.copy()
        error_df['Pred_Label'] = (error_df['Pred_Prob'] >= optimal_thresh).astype(int)
        error_df['Error_Type'] = 'Correct'
        error_df.loc[(error_df['True_Label']==1) & (error_df['Pred_Label']==0), 'Error_Type'] = 'False_Negative'
        error_df.loc[(error_df['True_Label']==0) & (error_df['Pred_Label']==1), 'Error_Type'] = 'False_Positive'
        error_analysis = error_df.merge(X_train, left_on='Patient_ID', right_index=True)
        error_analysis.to_csv(f"{Config.OUTPUT_DIR}/Error_Deep_Analysis_{timestamp}.csv", index=False, encoding='utf-8-sig')

        # E. 保存特征名称与处理后的矩阵 (SHAP 分析专用)
        try:
            # 第一步：应用预处理 (Scaling/Imputing)
            # 获取 Pipeline 中的 preprocessor 步骤
            preproc_step = best_model.named_steps['preprocessor']
            X_train_transformed = preproc_step.transform(X_train)
            X_ext_transformed = preproc_step.transform(X_ext)
        
            # 获取原始特征名（带防御性处理）
            try:
                final_feature_names = preproc_step.get_feature_names_out()
            except AttributeError:
                print("⚠️ get_feature_names_out() 不可用，手动构建特征名...")
                # 手动从 ColumnTransformer 的配置中提取特征名
                final_feature_names = []
                for trans_name, transformer, columns in preproc_step.transformers_:
                    if trans_name == 'remainder':
                        continue
                    if isinstance(columns, list):
                        final_feature_names.extend(columns)
                    elif isinstance(columns, str):
                        final_feature_names.append(columns)
                final_feature_names = np.array(final_feature_names)
                print(f"  ✓ 手动构建完成: {len(final_feature_names)} 个特征")

            # 第二步：检查是否应用了特征筛选 (Selector)
            if 'selector' in best_model.named_steps:
                selector_step = best_model.named_steps['selector']

                # 再次转换数据：从 10个特征 -> 6个特征
                X_train_transformed = selector_step.transform(X_train_transformed)
                X_ext_transformed = selector_step.transform(X_ext_transformed)
            
                # 更新特征名：只保留被选中的
                mask = selector_step.get_support()
                final_feature_names = final_feature_names[mask]

                print(f"ℹ️ 检测到特征筛选: SHAP数据维度已调整为 {X_train_transformed.shape[1]} 列")

            # 第三步：打包保存
            analysis_package = {
                'X_train_transformed': X_train_transformed,
                'X_ext_transformed': X_ext_transformed,
                'feature_names': final_feature_names,
                'optimal_threshold': optimal_thresh,
                'best_model_step': best_model.named_steps['classifier'] # 只存分类器部分用于SHAP
            }

            joblib.dump(analysis_package, f"{Config.OUTPUT_DIR}/SHAP_Analysis_Data_{timestamp}.pkl")
            print("✅ SHAP 分析数据包已保存 (维度已自动对齐)")

        except Exception as e:
            print(f"⚠️ SHAP 数据准备失败: {e}")
            import traceback
            traceback.print_exc()

        # --------------------------------------------------------
        # 7. 保存模型打包文件 (Pickle) - [无校准版]
        # --------------------------------------------------------
        
        # 从 final_results 获取训练集预测概率
        prob_train = results['prob_train']
        # 确保外部验证概率变量名一致 (使用 results 中的)
        prob_ext = results['prob_external']
        
        save_package = {
            # 原始数据
            'X_train': X_train, 
            'y_train': y_train,
            'X_external': X_ext, 
            'y_external': y_ext,
            
            # 特征信息
            'feature_names': {
                'numeric': num_feats, 
                'log': log_feats, 
                'categorical': cat_feats
            },
            
            
            # 模型信息
            'best_model': best_model,
            'best_model_name': best_model_name,
            'optimal_threshold': optimal_thresh,
            'preprocessor': preprocessor,
            
            # 移除校准模型字段
            'calibrated_model': None,
            
            # 交叉验证结果
            'cv_results': results['cv_results'],
            'final_metrics': {
                'train': results['train_metrics'], 
                'external': results['external_metrics']
            },
            'all_trained_models': trainer.models,
            
            # 预测概率 - 仅保存原始概率
            'predictions': {
                'train_probs': prob_train,
                'external_probs': prob_ext,
                'internal_oof_probs': results['prob_oof'],
            },
            
            # 曲线原始数据
            'curve_data': results.get('curve_data', {}),
            
            # 移除校准后指标对比
            'calibrated_metrics': None,
            
            # Bootstrap 稳定性分析结果
            'bootstrap_analysis': {
                'summary': bootstrap_summary,
                'raw_metrics': bootstrap_metrics,
                'predictions': bootstrap_predictions,
                'n_iterations': n_bootstrap
            },
            
            'risk_stratification': results.get('risk_stratification', None),
            
            # 元数据
            'internal_validation_method': 'OOF (Out-of-Fold)',
        }

        pkl_filename = f"{Config.OUTPUT_DIR}/Model_Package_{timestamp}.pkl"
        joblib.dump(save_package, pkl_filename)
        print(f"\n✅ 模型包已保存至: {pkl_filename}")
        print(f"   包含内容:")
        print(f"   - 最佳模型 (Uncalibrated)")
        print(f"   - 预测概率 (train + external)")
        print(f"   - 交叉验证结果")
        print(f"   - Bootstrap分析")
        print(f"   - 风险分层结果")
        # --------------------------------------------------------
        # 8. 保存 DeLong 检验专用数据包
        # --------------------------------------------------------
        print(f"\n📦 保存 DeLong 检验数据包（使用 OOF 预测）...")
        print(f"{'='*60}")
        # 收集所有模型的预测概率
        all_model_predictions = {}

        # 重新构建外部验证 DataFrame（仅用于外部验证预测）
        all_features = log_feats + num_feats + cat_feats
        X_ext_df = external_df[all_features].copy()

        # 强制转换数据类型
        for col in num_feats + log_feats:
            X_ext_df[col] = pd.to_numeric(X_ext_df[col], errors='coerce')

        print(f"  📊 数据格式: X_ext={type(X_ext_df).__name__}, shape={X_ext_df.shape}")
        print(f"  📋 待处理模型: {list(trainer.models.keys())}")
        print(f"  ℹ️  内部验证: 使用已保存的 OOF 预测")
        print(f"  ℹ️  外部验证: 使用融合模型预测")

        # ===================================================================
        # 方法：从 cv_results 中提取 OOF 预测
        # ===================================================================
        for model_name in trainer.models.keys():
            try:
                # ====== 1. 内部验证：使用 OOF 预测 ======
                cv_result = results['cv_results'].get(model_name, None)
                
                if cv_result is None:
                    print(f"  ⚠️ {model_name} - 未找到 CV 结果")
                    continue
                
                # ✅ 提取 OOF 预测
                oof_probs = cv_result.get('prob_oof', None)
                
                if oof_probs is None:
                    print(f"  ⚠️ {model_name} - 未找到 OOF 预测")
                    continue
                
                # 验证 OOF 预测的长度
                if len(oof_probs) != len(y_train):
                    print(f"  ⚠️ {model_name} - OOF 预测长度不匹配: {len(oof_probs)} vs {len(y_train)}")
                    continue
                
                # ====== 2. 外部验证：使用融合模型预测 ======
                model = trainer.models[model_name]
                
                if hasattr(model, 'predict_proba'):
                    ext_probs = model.predict_proba(X_ext_df)[:, 1]
                elif hasattr(model, 'decision_function'):
                    ext_probs = model.decision_function(X_ext_df)
                else:
                    print(f"  ⚠️ {model_name} - 没有 predict_proba 或 decision_function 方法")
                    continue
                
                # ====== 3. 保存到字典 ======
                all_model_predictions[model_name] = {
                    'internal_probs': oof_probs,  # ✅ OOF 预测
                    'external_probs': ext_probs,
                    'data_source': 'OOF'  # 标记数据来源
                }
                
                # 快速验证：计算 AUC + AUPRC + MCC
                from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
                auc_int = roc_auc_score(y_train, oof_probs)
                auc_ext = roc_auc_score(y_ext, ext_probs)
                ap_int = average_precision_score(y_train, oof_probs)
                ap_ext = average_precision_score(y_ext, ext_probs)
                diff = auc_int - auc_ext
                
                # 保存更多指标到predictions字典
                all_model_predictions[model_name]['auc_internal'] = auc_int
                all_model_predictions[model_name]['auc_external'] = auc_ext
                all_model_predictions[model_name]['aucpr_internal'] = ap_int
                all_model_predictions[model_name]['aucpr_external'] = ap_ext
                
                # 诊断提示
                status = "✅" if diff < 0.10 else "⚠️"
                print(f"  {status} {model_name:20s} - OOF预测已提取")
                print(f"      Internal AUC: {auc_int:.4f}, External AUC: {auc_ext:.4f}, Diff: {diff:+.4f}")
                
            except Exception as e:
                print(f"  ✗ {model_name} 处理失败: {e}")
                import traceback
                traceback.print_exc()

        # ===================================================================
        # 备选方案
        # ===================================================================
        if not all_model_predictions:
            print(f"\n  ❌ 警告：没有成功获取任何模型预测！")
            print(f"  尝试备选方案：仅保存最佳模型...")
            
            try:
                # 最佳模型的 OOF 预测
                best_cv_result = results['cv_results'][best_model_name]
                oof_probs_best = best_cv_result['prob_oof']
                
                # 外部验证预测
                if hasattr(best_model, 'predict_proba'):
                    ext_probs_best = best_model.predict_proba(X_ext_df)[:, 1]
                else:
                    ext_probs_best = best_model.decision_function(X_ext_df)
                
                all_model_predictions[best_model_name] = {
                    'internal_probs': oof_probs_best,  # ✅ OOF 预测
                    'external_probs': ext_probs_best,
                    'data_source': 'OOF'
                }
                print(f"  ✓ 备选方案成功: {best_model_name}")
                
            except Exception as e:
                print(f"  ✗ 备选方案也失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError("无法获取任何模型的 OOF 预测，请检查训练流程")

        # ===================================================================
        # 最终验证：检查内外部 AUC 差距
        # ===================================================================
        print(f"\n  📊 数据包质量检查:")
        print(f"  {'-'*96}")
        print(f"  {'模型名称':20s} | {'Int AUC':10s} | {'Ext AUC':10s} | {'Int AUPRC':10s} | {'Ext AUPRC':10s} | {'AUC差距':8s} | {'状态':10s}")
        print(f"  {'-'*96}")

        issues_found = []
        for model_name, preds in all_model_predictions.items():
            auc_int = roc_auc_score(y_train, preds['internal_probs'])
            auc_ext = roc_auc_score(y_ext, preds['external_probs'])
            ap_int = preds.get('aucpr_internal', average_precision_score(y_train, preds['internal_probs']))
            ap_ext = preds.get('aucpr_external', average_precision_score(y_ext, preds['external_probs']))
            diff = auc_int - auc_ext
            
            # 诊断标准
            if diff > 0.15:
                status = "❌ 异常"
                issues_found.append(f"{model_name}: 差距过大 ({diff:+.4f})")
            elif diff > 0.10:
                status = "⚠️ 可疑"
            else:
                status = "✅ 正常"
            
            print(f"  {model_name:20s} | {auc_int:10.4f} | {auc_ext:10.4f} | {ap_int:10.4f} | {ap_ext:10.4f} | {diff:+8.4f} | {status:10s}")

        print(f"  {'-'*76}")

        if issues_found:
            print(f"\n  ⚠️ 发现 {len(issues_found)} 个潜在问题:")
            for issue in issues_found:
                print(f"      - {issue}")
            print(f"  ⚠️ 建议检查这些模型的训练过程")
        else:
            print(f"\n  ✅ 所有模型的内外部 AUC 差距均在合理范围内")
            print(f"  ✅ 确认使用了 OOF 预测")

        # ===================================================================
        # 保存 DeLong 数据包
        # ===================================================================
        print(f"\n  💾 正在保存 DeLong 数据包...")

        delong_package = {
            # 真实标签
            'y_internal': y_train.values if hasattr(y_train, 'values') else y_train,
            'y_external': y_ext.values if hasattr(y_ext, 'values') else y_ext,
            
            # 模型预测（包含 OOF）
            'model_predictions': all_model_predictions,
            'model_names': list(all_model_predictions.keys()),
            
            # 元数据
            'best_model_name': best_model_name,
            'timestamp': timestamp,
            
            # 详细信息
            'metadata': {
                'n_internal_samples': len(y_train),
                'n_external_samples': len(y_ext),
                'features': all_features,
                'data_source': 'OOF',  # ✅ 明确标记
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cv_config': {
                    'n_folds': Config.CV_FOLDS,
                    'n_repeats': Config.CV_REPEATS,
                    'total_folds': Config.CV_FOLDS * Config.CV_REPEATS
                },
                'description': '内部验证使用 Out-of-Fold 交叉验证预测，外部验证使用融合模型预测'
            }
        }

        # 保存带时间戳的版本
        delong_pkl_filename = f"{Config.OUTPUT_DIR}/DeLong_Data_Package_{timestamp}.pkl"
        joblib.dump(delong_package, delong_pkl_filename)
        print(f"  ✓ DeLong 数据包已保存: {delong_pkl_filename}")

        # 保存 latest 版本
        delong_latest = f"{Config.OUTPUT_DIR}/DeLong_Data_Package_latest.pkl"
        joblib.dump(delong_package, delong_latest)
        print(f"  ✓ 最新版本已保存: {delong_latest}")

        print(f"\n  ✅ DeLong 数据包保存完成!")
        print(f"  ✅ 包含 {len(all_model_predictions)} 个模型的 OOF 预测")
        print(f"  ✅ 数据源: {delong_package['metadata']['data_source']}")
        print(f"{'='*60}")

        # ===================================================================
        # 保存一份简化的诊断报告
        # ===================================================================
        diagnostic_report = {
            'timestamp': timestamp,
            'data_source': 'OOF',
            'n_models': len(all_model_predictions),
            'auc_summary': {}
        }

        for model_name, preds in all_model_predictions.items():
            auc_int = roc_auc_score(y_train, preds['internal_probs'])
            auc_ext = roc_auc_score(y_ext, preds['external_probs'])
            
            diagnostic_report['auc_summary'][model_name] = {
                'internal_auc': float(auc_int),
                'external_auc': float(auc_ext),
                'difference': float(auc_int - auc_ext),
                'is_valid': abs(auc_int - auc_ext) < 0.15
            }
        print(f"\n✨ 任务成功完成!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
