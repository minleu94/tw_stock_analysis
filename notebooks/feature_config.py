from dataclasses import dataclass, field
from typing import Dict, Optional
import logging
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime

@dataclass
class FeatureConfig:
    """特徵生成器配置類別"""
    
    # 基礎路徑設定
    BASE_DIR: str = "D:/Min/Python/Project/FA_Data"
    META_DATA_DIR: str = "meta_data"
    TEST_DATA_DIR: str = "test_data"
    LOG_DIR: str = "logs"
    FEATURES_DIR: str = "features"
    INDUSTRY_ANALYSIS_DIR: str = "industry_analysis"
    INDUSTRY_CORRELATION_DIR: str = "industry_correlation"
    BACKUP_DIR: str = "backup"

    # 編碼設定
    ENCODING: str = 'utf-8'
    
    # 私有記錄器實例
    _logger: Optional[logging.Logger] = None
    
    # 測試資料設定
    TEST_SETTING: Dict = field(default_factory=lambda: {
        'test_stocks': ['2330', '2317', '1101', '2891', '2303'],
        'start_date': '2024-09-30',
        'end_date': '2024-12-30'
    })
    
    # 產業分析參數
    INDUSTRY_PARAMS: Dict = field(default_factory=lambda: {
        'analysis_start_date': '2023-01-01',
        'analysis_end_date': '2024-12-31',
        'min_data_days': 30,
        'update_frequency': 'daily'
    })
    
    # 特徵計算參數
    FEATURE_PARAMS: Dict = field(default_factory=lambda: {
        'volume': {
            'short_period': 5,
            'long_period': 20,
            'min_volume': 1000
        },
        'volatility': {
            'short_period': 5,
            'long_period': 20,
            'std_window': 20
        },
        'trend': {
            'ma_period': 20,
            'momentum_window': 10
        },
        'technical': {
            'rsi_period': 14,
            'macd_params': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'kd_params': {
                'fastk_period': 9,
                'slowk_period': 3,
                'slowd_period': 3
            },
            'bollinger_params': {
                'window': 20,
                'num_std': 2
            }
        }
    })
    
    def __post_init__(self):
        """初始化後設定"""
        self._setup_paths()
        if self._logger is None:
            self._logger = logging.getLogger('FeatureConfig')
            self._logger.setLevel(logging.INFO)
    
    def _setup_paths(self):
        """設定並建立必要的目錄結構"""
        self.base_path = Path(self.BASE_DIR)
        self.meta_data_path = self.base_path / self.META_DATA_DIR
        self.test_data_path = self.base_path / self.TEST_DATA_DIR
        self.log_path = self.base_path / self.LOG_DIR
        self.features_path = self.base_path / self.FEATURES_DIR
        self.industry_analysis_path = self.base_path / self.INDUSTRY_ANALYSIS_DIR
        self.industry_correlation_path = self.base_path / self.INDUSTRY_CORRELATION_DIR
        self.backup_path = self.base_path / self.BACKUP_DIR
        
        self._create_directories()
    
    def _create_directories(self):
        """建立必要的目錄"""
        directories = [
            self.test_data_path,
            self.log_path,
            self.features_path,
            self.industry_correlation_path,
            self.backup_path,
            self.meta_data_path / "backup"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_feature_filename(self, stock_id: str, industry: str) -> str:
        """生成特徵檔案名稱
        
        Args:
            stock_id: 股票代碼
            industry: 產業名稱（例如：水泥類報酬指數）
            
        Returns:
            str: 特徵檔案名稱
        """
        # 從產業名稱中提取基礎名稱（例如：從"水泥類報酬指數"提取"水泥"）
        base_name = industry.split('類')[0] if '類' in industry else industry.split('_')[0]
        
        # 處理時間格式
        start_date = self.TEST_SETTING['start_date'].replace('-', '')
        end_date = self.TEST_SETTING['end_date'].replace('-', '')
        
        # 返回格式化的檔案名稱
        return f"{base_name}_{start_date}_{end_date}_{end_date}.json"

    def backup_important_files(self):
        """備份重要文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        important_files = [
            "meta_data/companies.csv",
            "meta_data/industry_index.csv",
            "meta_data/market_index.csv",
            "test_data/test_stock_data.csv",
            "test_data/test_tech_data.csv"
        ]
        
        for file_path in important_files:
            src_path = self.base_path / file_path
            if src_path.exists():
                file_name = src_path.name
                backup_name = f"{file_name.rsplit('.', 1)[0]}_{timestamp}.{file_name.rsplit('.', 1)[1]}"
                dst_path = self.backup_path / backup_name
                shutil.copy2(src_path, dst_path)
                if self._logger:
                    self._logger.info(f"已備份 {file_path} 到 {dst_path}") 