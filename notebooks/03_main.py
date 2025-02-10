import logging
from pathlib import Path
from datetime import datetime
from feature_config import FeatureConfig
from feature_manager import FeatureManager
from feature_validator import FeatureValidator

def setup_logging(log_path: Path):
    """設定日誌系統"""
    # 確保日誌目錄存在
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 設定日誌格式
    log_file = log_path / f'feature_generation_{datetime.now():%Y%m%d}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main():
    """主程序"""
    try:
        # 初始化配置
        config = FeatureConfig()
        
        # 設定日誌
        setup_logging(config.log_path)
        logger = logging.getLogger("FeatureGeneration")
        logger.info("開始特徵生成流程...")
        
        # 初始化特徵管理器
        logger.info("初始化特徵管理器...")
        manager = FeatureManager(config)
        
        # 生成特徵
        logger.info("開始生成特徵...")
        if manager.generate_features():
            logger.info("特徵生成完成")
            
            # 驗證特徵
            logger.info("開始驗證特徵...")
            validator = FeatureValidator(config)
            if validator.validate_features():
                logger.info("特徵驗證通過")
            else:
                logger.warning("特徵驗證發現問題")
        else:
            logger.error("特徵生成失敗")
            
    except Exception as e:
        logger.error(f"執行過程中發生錯誤: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("特徵生成流程結束")

if __name__ == "__main__":
    main() 