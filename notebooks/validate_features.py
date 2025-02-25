import logging
from feature_config import FeatureConfig
from feature_validator import FeatureValidator

def setup_logging():
    """設定記錄器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """主程式"""
    # 設定記錄器
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化配置
        config = FeatureConfig()
        
        # 初始化驗證器
        validator = FeatureValidator(config)
        
        # 驗證特徵檔案
        logger.info("開始驗證特徵檔案...")
        validation_result = validator.validate_features()
        
        # 獲取文件結構信息（需要單獨調用）
        existing_files, missing_files, industry_status = validator._validate_file_structure()
        
        # 輸出驗證結果
        logger.info("\n產業特徵檔案狀態報告：")
        logger.info("-" * 50)
        for industry, status in industry_status.items():
            logger.info(f"{industry}: {status}")
            
        logger.info("\n詳細的特徵檔案列表：")
        logger.info("-" * 50)
        for file in existing_files:
            logger.info(f"- {file}")
            
        logger.info("\n缺失的產業特徵：")
        logger.info("-" * 50)
        for industry in missing_files:
            logger.info(f"- {industry}")
            
        # 驗證特徵內容
        logger.info("\n開始驗證特徵內容...")
        for file in existing_files:
            if validator.validate_feature_content(file):
                logger.info(f"特徵檔案內容驗證通過: {file}")
            else:
                logger.error(f"特徵檔案內容驗證失敗: {file}")
                
        # 合併特徵
        logger.info("\n開始合併特徵檔案...")
        if validator.combine_features(existing_files):
            logger.info("特徵合併完成")
        else:
            logger.error("特徵合併失敗")
            
        # 輸出整體驗證結果
        if validation_result:
            logger.info("特徵驗證全部通過")
        else:
            logger.warning("特徵驗證發現問題")
            
    except Exception as e:
        logger.error(f"執行過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 