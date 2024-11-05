import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import joblib
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class ModelConfig:
    """模型配置管理類"""
    def __init__(self):
        self.RANDOM_STATE = 42
        self.TEST_SIZE = 0.2
        self.N_SPLITS = 5
        self.EARLY_STOPPING_ROUNDS = 50
        
        # 模型保存路徑
        self.MODEL_PATH = Path("D:/Min/Python/Project/tw_stock_analysis/models")
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        
        # 模型評估指標閾值
        self.ACCURACY_THRESHOLD = 0.6
        self.PRECISION_THRESHOLD = 0.6
        self.RECALL_THRESHOLD = 0.6
        self.F1_THRESHOLD = 0.6

class ModelValidator:
    """模型驗證工具類"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_input_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> bool:
        """驗證輸入數據的有效性"""
        try:
            if X.empty:
                self.logger.error("輸入特徵數據為空")
                return False
                
            if y is not None and len(X) != len(y):
                self.logger.error("特徵數據和標籤長度不匹配")
                return False
                
            # 檢查缺失值
            if X.isnull().any().any():
                self.logger.warning("特徵數據中存在缺失值")
                return False
                
            # 檢查無限值
            if np.isinf(X.values).any():
                self.logger.error("特徵數據中存在無限值")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"數據驗證過程發生錯誤: {str(e)}")
            return False

    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """驗證預測結果的有效性"""
        try:
            if len(predictions) == 0:
                self.logger.error("預測結果為空")
                return False
                
            if np.isnan(predictions).any():
                self.logger.error("預測結果中存在缺失值")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"預測結果驗證失敗: {str(e)}")
            return False

class ModelTrainer:
    """模型訓練工具類"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validator = ModelValidator(config)

    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        feature_columns: List[str], 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """準備訓練數據"""
        try:
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # 驗證數據
            if not self.validator.validate_input_data(X, y):
                raise ValueError("數據驗證失敗")
                
            return X, y
            
        except Exception as e:
            self.logger.error(f"準備訓練數據時發生錯誤: {str(e)}")
            raise

    def train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """切分訓練和測試數據（考慮時間序列特性）"""
        try:
            # 使用時間序列分割
            tscv = TimeSeriesSplit(n_splits=self.config.N_SPLITS)
            
            # 獲取最後一個分割
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"數據分割過程發生錯誤: {str(e)}")
            raise

class ModelEvaluator:
    """模型評估工具類"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validator = ModelValidator(config)

    def calculate_metrics(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """計算模型評估指標"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
            
            self.logger.info(f"模型評估指標: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"計算評估指標時發生錯誤: {str(e)}")
            raise

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """評估模型性能"""
        try:
            if not self.validator.validate_input_data(X_test, y_test):
                raise ValueError("測試數據驗證失敗")
                
            # 進行預測
            y_pred = model.predict(X_test)
            
            if not self.validator.validate_predictions(y_pred):
                raise ValueError("預測結果驗證失敗")
                
            # 計算評估指標
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # 檢查是否達到閾值要求
            if (metrics['accuracy'] < self.config.ACCURACY_THRESHOLD or
                metrics['precision'] < self.config.PRECISION_THRESHOLD or
                metrics['recall'] < self.config.RECALL_THRESHOLD or
                metrics['f1'] < self.config.F1_THRESHOLD):
                self.logger.warning("模型性能未達到閾值要求")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"模型評估過程發生錯誤: {str(e)}")
            raise

class ModelSaver:
    """模型保存和加載工具類"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def save_model(self, model, model_name: str, metrics: Dict[str, float]) -> bool:
        """保存模型及其評估指標"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = self.config.MODEL_PATH / f"{model_name}_{timestamp}.joblib"
            metrics_path = self.config.MODEL_PATH / f"{model_name}_{timestamp}_metrics.json"
            
            # 保存模型
            joblib.dump(model, model_path)
            
            # 保存評估指標
            pd.Series(metrics).to_json(metrics_path)
            
            self.logger.info(f"模型已保存至: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型時發生錯誤: {str(e)}")
            return False

    def load_model(self, model_path: Union[str, Path]) -> Optional[object]:
        """加載已保存的模型"""
        try:
            model = joblib.load(model_path)
            self.logger.info(f"已成功加載模型: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"加載模型時發生錯誤: {str(e)}")
            return None

class ModelPredictor:
    """模型預測工具類"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validator = ModelValidator(config)

    def predict(
        self, 
        model, 
        X: pd.DataFrame, 
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """使用模型進行預測"""
        try:
            if not self.validator.validate_input_data(X):
                raise ValueError("預測數據驗證失敗")
            
            # 進行預測
            predictions = model.predict(X)
            
            if not self.validator.validate_predictions(predictions):
                raise ValueError("預測結果驗證失敗")
            
            if return_proba and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                return predictions, probabilities
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"模型預測過程發生錯誤: {str(e)}")
            raise

def setup_model_utils():
    """初始化模型工具"""
    try:
        config = ModelConfig()
        trainer = ModelTrainer(config)
        evaluator = ModelEvaluator(config)
        saver = ModelSaver(config)
        predictor = ModelPredictor(config)
        
        return {
            'config': config,
            'trainer': trainer,
            'evaluator': evaluator,
            'saver': saver,
            'predictor': predictor
        }
        
    except Exception as e:
        logging.error(f"初始化模型工具時發生錯誤: {str(e)}")
        raise