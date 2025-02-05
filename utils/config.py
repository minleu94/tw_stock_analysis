from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class TWStockConfig:
    """台股數據分析核心配置"""
    
    # 基礎路徑配置
    base_dir: Path = Path("C:/Users/archi/Python/Project/tw_stock_analysis")
    
    # 數據目錄
    data_dir: Path = None
    daily_price_dir: Path = None 
    meta_data_dir: Path = None
    technical_dir: Path = None
    
    # 關鍵檔案路徑
    market_index_file: Path = None
    industry_index_file: Path = None
    stock_data_file: Path = None
    
    # 數據參數
    default_start_date: str = "2014-01-01"
    backup_keep_days: int = 7
    min_data_days: int = 30  # 技術分析最小所需天數
    
    def __post_init__(self):
        """初始化衍生屬性"""
        # 設定數據目錄
        self.data_dir = self.base_dir / 'data'
        self.daily_price_dir = self.data_dir / 'daily_price'
        self.meta_data_dir = self.data_dir / 'meta_data'
        self.technical_dir = self.data_dir / 'technical_analysis'
        
        # 設定關鍵檔案路徑
        self.market_index_file = self.meta_data_dir / 'market_index.csv'
        self.industry_index_file = self.meta_data_dir / 'industry_index.csv'
        self.stock_data_file = self.meta_data_dir / 'stock_data_whole.csv'
        
        # 確保所需目錄存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """確保所需目錄結構存在"""
        directories = [
            self.daily_price_dir,
            self.meta_data_dir,
            self.technical_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def backup_dir(self) -> Path:
        """備份目錄路徑"""
        return self.meta_data_dir / 'backup'
    
    def get_technical_file(self, stock_id: str) -> Path:
        """取得特定股票的技術分析檔案路徑"""
        return self.technical_dir / f'{stock_id}_indicators.csv'
    
    def get_daily_price_file(self, date: str) -> Path:
        """取得特定日期的價格檔案路徑"""
        return self.daily_price_dir / f'{date}.csv'


class MarketDateRange:
    """市場數據日期範圍控制"""
    def __init__(self, start_date: str = None, end_date: str = None):
        self.end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        self.start_date = start_date
        
    @classmethod
    def last_n_days(cls, n: int) -> 'MarketDateRange':
        """創建最近 n 天的日期範圍"""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=n)
        return cls(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    
    @classmethod
    def last_month(cls) -> 'MarketDateRange':
        """創建最近一個月的日期範圍"""
        return cls.last_n_days(30)
    
    @classmethod
    def last_quarter(cls) -> 'MarketDateRange':
        """創建最近一季的日期範圍"""
        return cls.last_n_days(90)
    
    @classmethod
    def last_year(cls) -> 'MarketDateRange':
        """創建最近一年的日期範圍"""
        return cls.last_n_days(365)
    
    @classmethod
    def year_to_date(cls) -> 'MarketDateRange':
        """創建今年至今的日期範圍"""
        return cls(
            start_date=datetime.today().replace(month=1, day=1).strftime('%Y-%m-%d')
        )
        
    @property
    def date_range_str(self) -> str:
        """返回日期範圍的字符串表示"""
        return f"從 {self.start_date or '最早'} 到 {self.end_date}"