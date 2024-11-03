import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

class ConfigLoader:
    """配置檔案載入器"""
    
    @staticmethod
    def load_config(config_path: Path) -> Optional[Dict[str, Any]]:
        """從 YAML 文件載入配置
        
        Args:
            config_path: 配置文件路徑
            
        Returns:
            Dict[str, Any]: 配置字典，如果檔案不存在則返回 None
        """
        try:
            if not config_path.exists():
                return None
                
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            print(f"載入配置文件時發生錯誤: {str(e)}")
            return None
            
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Path) -> bool:
        """保存配置到 YAML 文件
        
        Args:
            config: 配置字典
            config_path: 保存路徑
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 確保目錄存在
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                
            return True
            
        except Exception as e:
            print(f"保存配置文件時發生錯誤: {str(e)}")
            return False
            
    @staticmethod
    def update_config(config_path: Path, updates: Dict[str, Any]) -> bool:
        """更新現有配置文件
        
        Args:
            config_path: 配置文件路徑
            updates: 要更新的配置項
            
        Returns:
            bool: 是否成功更新
        """
        try:
            # 載入現有配置
            current_config = ConfigLoader.load_config(config_path) or {}
            
            # 更新配置
            current_config.update(updates)
            
            # 保存更新後的配置
            return ConfigLoader.save_config(current_config, config_path)
            
        except Exception as e:
            print(f"更新配置文件時發生錯誤: {str(e)}")
            return False
            
    @classmethod
    def create_default_config(cls, save_path: Path) -> bool:
        """創建預設配置文件
        
        Args:
            save_path: 保存路徑
            
        Returns:
            bool: 是否成功創建
        """
        default_config = {
            'BASE_DIR': "D:/Min/Python/Project/FA_Data",
            'META_DATA_DIR': "meta_data",
            'BACKUP_DIR': "backup",
            'LOG_DIR': "logs",
            'FEATURES_DIR': "features",
            'DATA_PROCESSING': {
                'min_data_points': 30,
                'backup_days': 7,
                'missing_threshold': 0.1,
                'correlation_threshold': 0.95,
                'min_date_range': 30,
                'batch_size': 1000
            },
            'TECH_PARAMS': {
                'SMA': {'timeperiod': 30},
                'RSI': {'timeperiod': 14},
                'MACD': {
                    'fastperiod': 12,
                    'slowperiod': 26,
                    'signalperiod': 9
                },
                'STOCH': {
                    'fastk_period': 5,
                    'slowk_period': 3,
                    'slowd_period': 3
                }
            }
        }
        
        return cls.save_config(default_config, save_path)

def config_to_yaml(config_obj: Any) -> Dict[str, Any]:
    """將配置對象轉換為可序列化的字典
    
    Args:
        config_obj: 配置對象實例
        
    Returns:
        Dict[str, Any]: 可序列化的配置字典
    """
    return asdict(config_obj)