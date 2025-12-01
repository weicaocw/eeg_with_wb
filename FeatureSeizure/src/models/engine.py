import xgboost as xgb
import numpy as np
import logging
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)

class ModelEngine:
    def __init__(self, config):
        self.model_type = config['model']['type']
        raw_params = config['model'].get('params', {})
        if isinstance(raw_params, dict) and self.model_type in raw_params:
            self.params = raw_params.get(self.model_type, {})
        else:
            # backward compatibility: allow flat params dict
            self.params = raw_params
        self.model = None
        
    def train(self, X_train, y_train):
        # 计算 scale_pos_weight
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        ratio = float(num_neg) / num_pos if num_pos > 0 else 1.0
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                **self.params,
                scale_pos_weight=ratio,
                use_label_encoder=False
            )
            self.model.fit(X_train, y_train)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.params)
            self.model.fit(X_train, y_train)
        elif self.model_type == 'svm':
            svm_params = self.params.copy()
            svm_params.setdefault('probability', True)
            self.model = make_pipeline(
                StandardScaler(),
                SVC(**svm_params)
            )
            self.model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def evaluate(self, X_test, y_test, feature_names=None):
        if not self.model:
            logger.error("Model not trained.")
            return
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        logger.info("\n" + "="*40)
        logger.info("Classification Report")
        logger.info("="*40)
        
        # Handle case where test set has only one class
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 1:
            target_names = ['Normal'] if unique_classes[0] == 0 else ['Seizure']
            logger.info(classification_report(y_test, y_pred, target_names=target_names, digits=4))
        else:
            logger.info(classification_report(y_test, y_pred, target_names=['Normal', 'Seizure'], digits=4))
        
        # Only compute AUC if we have both classes
        if len(unique_classes) > 1:
            auc = roc_auc_score(y_test, y_prob)
            logger.info(f"ROC AUC: {auc:.4f}")
        else:
            auc = None
            logger.info("ROC AUC: N/A (only one class in test set)")
        
        # Confusion matrix
        if len(unique_classes) == 1:
            if unique_classes[0] == 0:  # Only Normal
                tn = np.sum((y_test == 0) & (y_pred == 0))
                fp = np.sum((y_test == 0) & (y_pred == 1))
                fn = 0
                tp = 0
            else:  # Only Seizure
                tn = 0
                fp = 0
                fn = np.sum((y_test == 1) & (y_pred == 0))
                tp = np.sum((y_test == 1) & (y_pred == 1))
        else:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        logger.info(f"Sensitivity: {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        self._log_feature_importance(feature_names)
        
        return {
            'auc': auc, 
            'sensitivity': sensitivity, 
            'specificity': specificity,
            'model': self.model
        }

    def _log_feature_importance(self, feature_names):
        if self.model_type not in {"xgboost", "random_forest"}:
            logger.info(f"Feature importance not supported for model type: {self.model_type}")
            return
        if not feature_names:
            logger.info("Feature names unavailable; skipping feature importance logging.")
            return
        importances = getattr(self.model, 'feature_importances_', None)
        if importances is None:
            logger.info("Model does not provide feature_importances_.")
            return
        top_k = min(10, len(importances), len(feature_names))
        sorted_idx = np.argsort(importances)[::-1][:top_k]
        logger.info("Top feature importances:")
        for idx in sorted_idx:
            if idx < len(feature_names):
                logger.info(f"  {feature_names[idx]}: {importances[idx]:.4f}")
            else:
                logger.info(f"  [index {idx} out of range]: {importances[idx]:.4f}")