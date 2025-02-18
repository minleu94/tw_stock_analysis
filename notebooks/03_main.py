import logging
from pathlib import Path
from datetime import datetime
import sys
from feature_config import FeatureConfig
from feature_manager import FeatureManager
from feature_validator import FeatureValidator

def setup_logging(log_path: Path) -> logging.Logger:
    """設定日誌系統"""
    # 確保日誌目錄存在
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 創建logger
    logger = logging.getLogger("FeatureGeneration")
    logger.setLevel(logging.INFO)
    
    # 清除現有的處理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 創建文件處理器
    log_file = log_path / f'feature_generation_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 創建控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 創建格式器
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加處理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def validate_config(config: FeatureConfig, logger: logging.Logger) -> bool:
    """驗證配置是否有效"""
    try:
        if not config.TEST_SETTING.get('start_date') or not config.TEST_SETTING.get('end_date'):
            logger.error("缺少必要的日期配置")
            return False
        if not config.TEST_SETTING.get('test_stocks'):
            logger.error("缺少測試股票列表")
            return False
        
        # 添加更多驗證
        if not isinstance(config.TEST_SETTING.get('test_stocks'), list):
            logger.error("test_stocks 必須是列表格式")
            return False
            
        # 驗證日期格式
        try:
            datetime.strptime(config.TEST_SETTING['start_date'], '%Y-%m-%d')
            datetime.strptime(config.TEST_SETTING['end_date'], '%Y-%m-%d')
        except ValueError:
            logger.error("日期格式錯誤，應為 YYYY-MM-DD")
            return False
            
        # 驗證開始日期小於結束日期
        if config.TEST_SETTING['start_date'] > config.TEST_SETTING['end_date']:
            logger.error("開始日期不能大於結束日期")
            return False
            
        return True
    except Exception as e:
        logger.error(f"配置驗證失敗: {str(e)}")
        return False

def main():
    """主程序"""
    logger = None
    try:
        # 初始化配置
        config = FeatureConfig()
        
        # 設定日誌
        logger = setup_logging(config.log_path)
        
        # 驗證配置
        if not validate_config(config, logger):
            logger.error("配置驗證失敗，程序終止")
            sys.exit(1)
        
        # 添加配置文件存在性檢查
        if not config.project_path.exists():
            raise FileNotFoundError(f"找不到專案目錄：{config.project_path}")
            
        # 添加數據目錄檢查
        if not config.data_path.exists():
            raise FileNotFoundError(f"找不到數據目錄：{config.data_path}")
        
        # 記錄開始時間
        start_time = datetime.now()
        logger.info("開始特徵生成流程...")
        logger.info(f"處理時間範圍: {config.TEST_SETTING['start_date']} 到 {config.TEST_SETTING['end_date']}")
        logger.info(f"處理股票數量: {len(config.TEST_SETTING['test_stocks'])}")
        
        # 初始化特徵管理器
        logger.info("初始化特徵管理器...")
        manager = FeatureManager(config)
        
        # 生成特徵
        logger.info("開始生成特徵...")
        if not manager.generate_features():
            logger.error("特徵生成失敗")
            return
            
        logger.info("特徵生成完成")
        
        # 驗證特徵
        logger.info("開始驗證特徵...")
        validator = FeatureValidator(config)
        if not validator.validate_features():
            logger.warning("特徵驗證發現問題")
            return
            
        logger.info("特徵驗證通過")
            
        # 記錄結束時間和執行時間
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"特徵生成流程結束，總執行時間: {execution_time}")
        
        # 記錄記憶體使用情況
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"最終記憶體使用: {memory_info.rss / 1024 / 1024:.2f} MB")
        
    except ImportError as e:
        if logger:
            logger.error(f"模組導入錯誤: {str(e)}")
        else:
            print(f"錯誤: 模組導入失敗: {str(e)}")
    except FileNotFoundError as e:
        if logger:
            logger.error(f"找不到必要的文件: {str(e)}")
        else:
            print(f"錯誤: 找不到必要的文件: {str(e)}")
    except PermissionError as e:
        if logger:
            logger.error(f"權限錯誤: {str(e)}")
        else:
            print(f"錯誤: 權限不足: {str(e)}")
    except Exception as e:
        if logger:
            logger.error(f"執行過程中發生錯誤: {str(e)}", exc_info=True)
        else:
            print(f"錯誤: {str(e)}")
        raise
    finally:
        if logger:
            logger.info("特徵生成流程結束")

if __name__ == "__main__":
    main() 