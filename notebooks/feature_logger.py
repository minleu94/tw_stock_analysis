import logging

class FeatureLogger:
    def __init__(self, config=None):
        self.logger = logging.getLogger('FeatureGenerator')
        self.logger.setLevel(logging.INFO)
        
        # 如果沒有處理器，添加基本處理器
        if not self.logger.handlers:
            # 添加控制台處理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(console_handler)
            
            # 如果有配置，添加檔案處理器
            if config:
                log_file = config.get_log_path()
                file_handler = logging.FileHandler(
                    log_file, 
                    encoding=config.ENCODING if hasattr(config, 'ENCODING') else 'utf-8'
                )
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
                )
                self.logger.addHandler(file_handler) 