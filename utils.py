"""
工具函数模块
"""

import torch
import logging

def setup_logging(level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_device():
    """获取计算设备（GPU或CPU）"""
    logger = logging.getLogger(__name__)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        logger.warning("⚠️ GPU not available, using CPU")
    
    return device

