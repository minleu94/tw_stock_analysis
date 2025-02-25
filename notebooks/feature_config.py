from dataclasses import dataclass, field
from typing import Dict, Optional
import logging
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime, timedelta
import sys

@dataclass
class FeatureConfig:
    """特徵生成器配置類別"""
    
    # 基礎路徑設定
    PROJECT_DIR: str = str(Path(__file__).parent)  # 使用當前文件所在目錄作為專案目錄
    DATA_DIR: str = "D:/Min/Python/Project/FA_Data"  # 固定的資料目錄
    META_DATA_DIR: str = "meta_data"
    TEST_DATA_DIR: str = "test_data"
    LOG_DIR: str = "logs"
    FEATURES_DIR: str = "features"
    INDUSTRY_ANALYSIS_DIR: str = "industry_analysis"
    INDUSTRY_CORRELATION_DIR: str = "industry_correlation"
    BACKUP_DIR: str = "backup"
    TECHNICAL_ANALYSIS_DIR: str = "technical_analysis"
    
    # 產業分析相關目錄
    INDUSTRY_PRICE_INDEX_DIR: str = "price_index"
    INDUSTRY_RETURN_INDEX_DIR: str = "return_index"
    INDUSTRY_CORRELATION_WEEKLY_DIR: str = "weekly"
    INDUSTRY_CORRELATION_MONTHLY_DIR: str = "monthly"
    
    # 文件格式設定
    TECHNICAL_FILE_PATTERN: str = "{stock_id}_indicators.csv"
    INDUSTRY_PRICE_INDEX_PATTERN: str = "{industry}_{data_start_date}_{data_end_date}_{report_date}.json"
    INDUSTRY_RETURN_INDEX_PATTERN: str = "{industry}_{data_start_date}_{data_end_date}_{report_date}.json"
    INDUSTRY_CORRELATION_WEEKLY_PATTERN: str = "industry_correlation_{date}.csv"
    INDUSTRY_CORRELATION_MONTHLY_PATTERN: str = "industry_correlation_{date}.csv"
    
    # 編碼設定
    ENCODING: str = 'utf-8-sig'
    
    # 技術指標欄位映射
    TECHNICAL_COLUMN_MAPPING: Dict = field(default_factory=lambda: {
        'KD_K': 'slowk',
        'KD_D': 'slowd',
        'MA30': 'SMA30',
        'DEMA_30': 'DEMA30',
        'EMA_30': 'EMA30'
    })
    
    # 技術指標參數
    TECHNICAL_PARAMS: Dict = field(default_factory=lambda: {
        'ma': {
            'periods': [5, 10, 20, 30, 60],
            'types': ['SMA', 'EMA', 'DEMA']
        },
        'rsi': {'period': 14},
        'macd': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        },
        'kd': {
            'fastk_period': 9,
            'slowk_period': 3,
            'slowd_period': 3
        }
    })
    
    # 測試資料設定
    TEST_SETTING: Dict = field(default_factory=lambda: {
        'test_stocks': ['2330', '2317'],
        'start_date': '2023-01-03',
        'end_date': '2024-11-12'
    })
    
    # 產業分析參數
    INDUSTRY_PARAMS: Dict = field(default_factory=lambda: {
        'min_data_days': 30,
        'update_frequency': 'daily'
    })
    
    def __post_init__(self):
        """初始化後的設置"""
        # 設置基礎路徑
        self._project_path = Path(self.PROJECT_DIR)
        self._data_path = Path(self.DATA_DIR)
        
        # 設置子目錄路徑
        self._meta_data_path = self._data_path / self.META_DATA_DIR
        self._test_data_path = self._data_path / self.TEST_DATA_DIR
        self._features_path = self._data_path / self.FEATURES_DIR
        self._technical_analysis_path = self._data_path / self.TECHNICAL_ANALYSIS_DIR
        self._industry_analysis_path = self._data_path / self.INDUSTRY_ANALYSIS_DIR
        self._industry_correlation_path = self._data_path / self.INDUSTRY_CORRELATION_DIR
        self._log_path = self._project_path / self.LOG_DIR
        self._backup_path = self._data_path / self.BACKUP_DIR
        
        # 創建必要的目錄
        self._create_directories()
    
    def _create_directories(self):
        """創建必要的目錄"""
        directories = [
            self._project_path,
            self._data_path,
            self._meta_data_path,
            self._test_data_path,
            self._features_path,
            self._technical_analysis_path,
            self._industry_analysis_path,
            self._industry_correlation_path,
            self._log_path,
            self._backup_path,
            # 產業分析相關子目錄
            self._industry_analysis_path / self.INDUSTRY_PRICE_INDEX_DIR,
            self._industry_analysis_path / self.INDUSTRY_RETURN_INDEX_DIR,
            self._industry_correlation_path / self.INDUSTRY_CORRELATION_WEEKLY_DIR,
            self._industry_correlation_path / self.INDUSTRY_CORRELATION_MONTHLY_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def project_path(self) -> Path:
        return self._project_path
        
    @property
    def data_path(self) -> Path:
        return self._data_path
        
    @property
    def meta_data_path(self) -> Path:
        return self._meta_data_path
        
    @property
    def test_data_path(self) -> Path:
        return self._test_data_path
        
    @property
    def features_path(self) -> Path:
        return self._features_path
        
    @property
    def technical_analysis_path(self) -> Path:
        return self._technical_analysis_path
        
    @property
    def industry_analysis_path(self) -> Path:
        return self._industry_analysis_path
        
    @property
    def industry_correlation_path(self) -> Path:
        return self._industry_correlation_path
        
    @property
    def log_path(self) -> Path:
        return self._log_path
        
    @property
    def backup_path(self) -> Path:
        return self._backup_path
    
    def get_feature_filename(self, stock_id: str, industry_name: Optional[str] = None) -> str:
        """獲取特徵檔案名稱"""
        start_date = self.TEST_SETTING['start_date'].replace('-', '')
        end_date = self.TEST_SETTING['end_date'].replace('-', '')
        process_date = datetime.now().strftime('%Y%m%d')
        if industry_name:
            # 確保產業名稱不包含特殊字符
            clean_industry_name = industry_name.strip()
            return f"{stock_id}_{clean_industry_name}_{start_date}_{end_date}_{process_date}.csv"
        return f"{stock_id}_features_{start_date}_{end_date}_{process_date}.csv"
    
    def get_combined_feature_filename(self) -> str:
        """獲取合併特徵檔案名稱"""
        start_date = self.TEST_SETTING['start_date'].replace('-', '')
        end_date = self.TEST_SETTING['end_date'].replace('-', '')
        process_date = datetime.now().strftime('%Y%m%d')
        return f"combined_features_{start_date}_{end_date}_{process_date}.csv"
    
    def get_stock_data_path(self) -> Path:
        """獲取股票數據檔案路徑"""
        return self._meta_data_path / 'stock_data_whole.csv'

    def get_tech_data_path(self) -> Path:
        """獲取技術分析資料檔案路徑"""
        data_type = 'test' if self.USE_TEST_DATA else 'prod'
        path = self.data_path / self.DATA_FILES[data_type]['tech_data']
        if self._logger:
            self._logger.info(f"使用技術分析資料: {path}")
        return path
    
    def _standardize_industry_name(self, industry_name: str) -> str:
        """標準化產業名稱"""
        # 移除常見後綴
        name = industry_name.replace('類報酬指數', '')\
                          .replace('類指數', '')\
                          .replace('類日報酬兩倍指數', '')\
                          .replace('類日報酬反向一倍指數', '')\
                          .replace('類', '')\
                          .strip()
        
        # 根據特定規則添加後綴
        if name in ['水泥', '食品', '塑膠', '紡織纖維', '電機機械', '電器電纜', '化學', '生技醫療',
                   '玻璃陶瓷', '造紙', '鋼鐵', '橡膠', '汽車', '半導體', '電腦及週邊設備', '光電',
                   '通信網路', '電子零組件', '電子通路', '資訊服務', '其他電子']:
            return f"{name}工業"
        elif name in ['建材營造', '航運', '觀光餐旅', '金融保險', '貿易百貨', '油電燃氣',
                     '居家生活', '數位雲端', '運動休閒', '綠能環保']:
            return f"{name}業"
        
        return name
        
    def get_industry_price_index_path(self, industry_name: str) -> Path:
        """
        取得產業指數檔案的路徑
        Args:
            industry_name: 產業名稱
        Returns:
            Path: 產業指數檔案的路徑
        """
        # 標準化產業名稱
        industry_name = self._standardize_industry_name(industry_name)
        
        # 從產業對照檔案中取得對應的分析檔案名稱
        mapping_df = pd.read_csv(self.meta_data_path / 'industry_mapping_analysis.csv')
        matched_row = mapping_df[mapping_df['標準化產業'] == industry_name]
        
        if not matched_row.empty:
            analysis_file_name = matched_row['產業分析檔案'].iloc[0]
            # 使用固定的日期格式
            file_name = f"{analysis_file_name}_20230101_20250122_20250122.json"
            return self.industry_analysis_path / 'price_index' / file_name
        else:
            raise ValueError(f"找不到產業 {industry_name} 的對應分析檔案名稱")

    def get_industry_return_index_path(self, industry_name: str, start_date: str, end_date: str) -> Path:
        """獲取產業報酬指數分析檔案路徑
        
        Args:
            industry_name: 產業名稱
            start_date: 開始日期 (YYYYMMDD)
            end_date: 結束日期 (YYYYMMDD)
            
        Returns:
            Path: 產業報酬指數分析檔案路徑
        """
        try:
            # 標準化產業名稱
            standardized_name = self._standardize_industry_name(industry_name)
            
            # 生成檔案名稱
            current_date = datetime.now().strftime('%Y%m%d')
            file_name = f"{standardized_name}_{start_date}_{end_date}_{current_date}.json"
            
            # 構建完整路徑
            file_path = self.industry_analysis_path / self.INDUSTRY_RETURN_INDEX_DIR / file_name
            
            if not file_path.parent.exists():
                self._logger.warning(f"目錄不存在，建立目錄: {file_path.parent}")
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            return file_path
            
        except Exception as e:
            self._logger.error(f"生成產業報酬指數檔案路徑時發生錯誤: {str(e)}")
            raise

    def get_industry_correlation_weekly_path(self, date: str) -> Path:
        """獲取週度產業關聯性文件路徑"""
        filename = self.INDUSTRY_CORRELATION_WEEKLY_PATTERN.format(date=date)
        path = self.industry_correlation_path / filename
        if self._logger:
            self._logger.info(f"週度產業關聯性路徑: {path}")
        return path

    def get_industry_correlation_monthly_path(self, date: str) -> Path:
        """獲取月度產業關聯性文件路徑"""
        filename = self.INDUSTRY_CORRELATION_MONTHLY_PATTERN.format(date=date)
        path = self.industry_correlation_path / filename
        if self._logger:
            self._logger.info(f"月度產業關聯性路徑: {path}")
        return path 