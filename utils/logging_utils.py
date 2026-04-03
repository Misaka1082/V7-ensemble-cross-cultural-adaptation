#!/usr/bin/env python3
"""
日志工具函数 - 修复版本
包含create_summary_report方法
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import yaml

class LoggingUtils:
    """日志工具类"""
    
    @staticmethod
    def setup_logger(name: str = "4.1.9",
                    log_level: str = "INFO",
                    log_file: Optional[str] = None,
                    console_output: bool = True) -> logging.Logger:
        """设置日志记录器"""
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有的处理器
        logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def create_experiment_logger(experiment_name: str,
                                log_dir: str = "logs",
                                log_level: str = "INFO") -> logging.Logger:
        """创建实验日志记录器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.log"
        
        logger = LoggingUtils.setup_logger(
            name=f"experiment_{experiment_name}",
            log_level=log_level,
            log_file=str(log_file),
            console_output=True
        )
        
        logger.info(f"实验开始: {experiment_name}")
        logger.info(f"日志文件: {log_file}")
        logger.info(f"日志级别: {log_level}")
        
        return logger
    
    @staticmethod
    def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
        """记录配置信息"""
        logger.info("=" * 80)
        logger.info("配置信息")
        logger.info("=" * 80)
        
        for section, settings in config.items():
            logger.info(f"[{section}]")
            if isinstance(settings, dict):
                for key, value in settings.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {settings}")
        
        logger.info("=" * 80)
    
    @staticmethod
    def log_data_info(logger: logging.Logger, 
                     data_info: Dict[str, Any]) -> None:
        """记录数据信息"""
        logger.info("=" * 80)
        logger.info("数据信息")
        logger.info("=" * 80)
        
        for key, value in data_info.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"  {subkey}: {subvalue}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("=" * 80)
    
    @staticmethod
    def log_model_info(logger: logging.Logger,
                      model_info: Dict[str, Any]) -> None:
        """记录模型信息"""
        logger.info("=" * 80)
        logger.info("模型信息")
        logger.info("=" * 80)
        
        for key, value in model_info.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"  {subkey}: {subvalue}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("=" * 80)
    
    @staticmethod
    def log_training_start(logger: logging.Logger,
                          epoch: int,
                          total_epochs: int) -> None:
        """记录训练开始"""
        logger.info(f"开始训练 Epoch {epoch}/{total_epochs}")
    
    @staticmethod
    def log_training_progress(logger: logging.Logger,
                            epoch: int,
                            total_epochs: int,
                            train_loss: float,
                            val_loss: Optional[float] = None,
                            metrics: Optional[Dict[str, float]] = None) -> None:
        """记录训练进度"""
        progress = f"Epoch {epoch}/{total_epochs} - "
        progress += f"训练损失: {train_loss:.6f}"
        
        if val_loss is not None:
            progress += f", 验证损失: {val_loss:.6f}"
        
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            progress += f", 指标: [{metric_str}]"
        
        logger.info(progress)
    
    @staticmethod
    def log_epoch_summary(logger: logging.Logger,
                         epoch: int,
                         train_loss: float,
                         val_loss: float,
                         metrics: Dict[str, float],
                         learning_rate: float) -> None:
        """记录epoch摘要"""
        logger.info("-" * 80)
        logger.info(f"Epoch {epoch} 摘要")
        logger.info("-" * 80)
        logger.info(f"训练损失: {train_loss:.6f}")
        logger.info(f"验证损失: {val_loss:.6f}")
        logger.info(f"学习率: {learning_rate:.6f}")
        
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        logger.info("-" * 80)
    
    @staticmethod
    def log_early_stopping(logger: logging.Logger,
                          patience: int,
                          best_epoch: int,
                          best_val_loss: float) -> None:
        """记录早停信息"""
        logger.warning("=" * 80)
        logger.warning("早停触发")
        logger.warning("=" * 80)
        logger.warning(f"耐心值: {patience}")
        logger.warning(f"最佳epoch: {best_epoch}")
        logger.warning(f"最佳验证损失: {best_val_loss:.6f}")
        logger.warning("=" * 80)
    
    @staticmethod
    def log_training_complete(logger: logging.Logger,
                            total_epochs: int,
                            best_epoch: int,
                            best_val_loss: float,
                            training_time: float) -> None:
        """记录训练完成"""
        logger.info("=" * 80)
        logger.info("训练完成")
        logger.info("=" * 80)
        logger.info(f"总epoch数: {total_epochs}")
        logger.info(f"最佳epoch: {best_epoch}")
        logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"训练时间: {training_time:.2f} 秒")
        logger.info(f"平均每epoch时间: {training_time/total_epochs:.2f} 秒")
        logger.info("=" * 80)
    
    @staticmethod
    def log_evaluation_results(logger: logging.Logger,
                              results: Dict[str, Any]) -> None:
        """记录评估结果"""
        logger.info("=" * 80)
        logger.info("评估结果")
        logger.info("=" * 80)
        
        for category, category_results in results.items():
            logger.info(f"[{category}]")
            
            if isinstance(category_results, dict):
                for key, value in category_results.items():
                    if isinstance(value, dict):
                        logger.info(f"  {key}:")
                        for subkey, subvalue in value.items():
                            logger.info(f"    {subkey}: {subvalue}")
                    else:
                        logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {category_results}")
        
        logger.info("=" * 80)
    
    @staticmethod
    def log_error(logger: logging.Logger,
                 error: Exception,
                 context: Optional[str] = None) -> None:
        """记录错误信息"""
        logger.error("=" * 80)
        logger.error("错误发生")
        logger.error("=" * 80)
        
        if context:
            logger.error(f"上下文: {context}")
        
        logger.error(f"错误类型: {type(error).__name__}")
        logger.error(f"错误信息: {str(error)}")
        
        import traceback
        logger.error("堆栈跟踪:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(line)
        
        logger.error("=" * 80)
    
    @staticmethod
    def log_warning(logger: logging.Logger,
                   warning: str,
                   context: Optional[str] = None) -> None:
        """记录警告信息"""
        if context:
            logger.warning(f"[{context}] {warning}")
        else:
            logger.warning(warning)
    
    @staticmethod
    def log_info(logger: logging.Logger,
                info: str,
                context: Optional[str] = None) -> None:
        """记录一般信息"""
        if context:
            logger.info(f"[{context}] {info}")
        else:
            logger.info(info)
    
    @staticmethod
    def log_debug(logger: logging.Logger,
                 debug_info: str,
                 context: Optional[str] = None) -> None:
        """记录调试信息"""
        if context:
            logger.debug(f"[{context}] {debug_info}")
        else:
            logger.debug(debug_info)
    
    @staticmethod
    def save_results_to_json(results: Dict[str, Any],
                            file_path: str) -> None:
        """保存结果到JSON文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            """递归转换对象为JSON可序列化类型"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            elif hasattr(obj, 'item'):  # numpy标量类型
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy数组
                return obj.tolist()
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                # 尝试转换为字符串
                try:
                    return str(obj)
                except:
                    return None
        
        # 转换结果
        serializable_results = convert_to_serializable(results)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def create_summary_report(logger: logging.Logger,
                            summary_data: Dict[str, Any]) -> str:
        """创建训练摘要报告"""
        logger.info("生成训练摘要报告...")
        
        # 构建报告
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DeepFM心理学跨文化适应预测模型训练摘要")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 训练信息
        report_lines.append("训练信息:")
        training_info = summary_data.get('训练信息', {})
        for key, value in training_info.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # 性能指标
        report_lines.append("性能指标:")
        performance_metrics = summary_data.get('性能指标', {})
        for metric_name, metric_value in performance_metrics.items():
            if isinstance(metric_value, (int, float)):
                report_lines.append(f"  {metric_name}: {metric_value:.4f}")
            else:
                report_lines.append(f"  {metric_name}: {metric_value}")
        report_lines.append("")
        
        # 损失变化
        report_lines.append("损失变化:")
        loss_changes = summary_data.get('损失变化', {})
        for key, value in loss_changes.items():
            if value is not None and isinstance(value, (int, float)):
                report_lines.append(f"  {key}: {value:.6f}")
            else:
                report_lines.append(f"  {key}: {value}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    @staticmethod
    def print_progress_bar(iteration: int, 
                          total: int, 
                          prefix: str = '', 
                          suffix: str = '', 
                          length: int = 50, 
                          fill: str = '█') -> None:
        """打印进度条"""
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
        
        # 当完成时打印新行
        if iteration == total:
            print()
