import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_config import FeatureConfig
from feature_manager import FeatureManager
from feature_validator import FeatureValidator

def setup_logging():
    """設定記錄器"""
    # 確保日誌目錄存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 設定日誌格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"logs/feature_pipeline_{datetime.now():%Y%m%d}.log", encoding='utf-8-sig'),
            logging.StreamHandler()
        ]
    )

def run_feature_generation(config):
    """執行特徵生成流程"""
    logger = logging.getLogger("FeatureGeneration")
    logger.info("開始特徵生成流程...")
    
    # 初始化特徵管理器
    manager = FeatureManager(config)
    
    # 生成特徵
    if manager.process_all_industries():
        logger.info("特徵生成完成")
        return True
    else:
        logger.error("特徵生成失敗")
        return False

def run_feature_validation(config):
    """執行特徵驗證流程"""
    logger = logging.getLogger("FeatureValidation")
    logger.info("開始特徵驗證流程...")
    
    # 初始化特徵驗證器
    validator = FeatureValidator(config)
    
    # 驗證特徵
    if validator.validate_features():
        logger.info("特徵驗證通過")
        return True
    else:
        logger.error("特徵驗證失敗")
        return False

def main():
    """主程式"""
    # 設定記錄器
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化配置
        config = FeatureConfig()
        
        # 執行特徵生成
        generation_success = run_feature_generation(config)
        
        # 執行特徵驗證
        validation_success = run_feature_validation(config)
        
        # 輸出最終結果
        if generation_success and validation_success:
            logger.info("特徵生成與驗證流程全部完成")
        else:
            if not generation_success:
                logger.error("特徵生成流程失敗")
            if not validation_success:
                logger.error("特徵驗證流程失敗")
                
    except Exception as e:
        logger.error(f"執行過程中發生錯誤: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 