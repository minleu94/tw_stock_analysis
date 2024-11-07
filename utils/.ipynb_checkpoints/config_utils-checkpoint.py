# utils/config_utils.py

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

class ConfigLoader:
    """配置加載器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加載器
        
        Args:
            config_path: 配置文件路徑，如果為None則使用默認路徑
        """
        self.config_path = config_path or self._get_default_config_path()
        
    def _get_default_config_path(self) -> str:
        """獲取默認配置文件路徑"""
        return str(Path(__file__).parent.parent / 'config' / 'config.yaml')
        
    def load_config(self) -> Dict[str, Any]:
        """
        加載配置文件
        
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(self.config_path):
                return self._get_default_config()
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加載配置文件時發生錯誤: {str(e)}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """獲取默認配置"""
        return {
            'base_dir': "D:/Min/Python/Project/FA_Data",
            'data_dirs': {
                'meta_data': "meta_data",
                'daily_price': "daily_price",
                'technical': "technical_analysis",
                'features': "features",
                'backup': "backup",
                'logs': "logs"
            },
            'data_processing': {
                'min_data_points': 30,
                'backup_days': 7,
                'batch_size': 1000
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_encoding': 'utf-8'
            },
            'network': {
                'timeout': 30,
                'max_retries': 3,
                'retry_delay': 5
            }
        }
        
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            
        Returns:
            是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            return True
        except Exception as e:
            print(f"保存配置文件時發生錯誤: {str(e)}")
            return False

def config_to_yaml(config: Any) -> str:
    """
    將配置對象轉換為YAML格式字符串
    
    Args:
        config: 配置對象(通常是dataclass實例)
        
    Returns:
        YAML格式的配置字符串
    """
    try:
        if hasattr(config, '__dataclass_fields__'):
            # 如果是dataclass實例，先轉換為字典
            config_dict = asdict(config)
        else:
            # 否則假設它已經是字典
            config_dict = config
            
        return yaml.dump(config_dict, allow_unicode=True, default_flow_style=False)
    except Exception as e:
        print(f"轉換配置到YAML時發生錯誤: {str(e)}")
        return ""

# 配置相關的數據類
@dataclass
class NetworkConfig:
    """網路請求配置"""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5

@dataclass
class PathConfig:
    """路徑配置"""
    base_dir: str = "D:/Min/Python/Project/FA_Data"
    meta_data: str = "meta_data"
    daily_price: str = "daily_price"
    technical: str = "technical_analysis"
    features: str = "features"
    backup: str = "backup"
    logs: str = "logs"

@dataclass
class ProcessingConfig:
    """數據處理配置"""
    min_data_points: int = 30
    backup_days: int = 7
    batch_size: int = 1000

@dataclass
class LoggingConfig:
    """日誌配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_encoding: str = "utf-8"

@dataclass
class SystemConfig:
    """系統總配置"""
    network: NetworkConfig = NetworkConfig()
    paths: PathConfig = PathConfig()
    processing: ProcessingConfig = ProcessingConfig()
    logging: LoggingConfig = LoggingConfig()