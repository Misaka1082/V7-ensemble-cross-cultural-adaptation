#!/usr/bin/env python3
"""
模型工具函数 - 修复版本
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """设置随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def get_device(device_id: int = 0) -> torch.device:
        """获取设备"""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
            print(f"使用GPU: {torch.cuda.get_device_name(device_id)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device("cpu")
            print("使用CPU")
        
        return device
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict:
        """统计模型参数"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'trainable_percentage': trainable_params / total_params * 100 if total_params > 0 else 0
        }
    
    @staticmethod
    def initialize_weights(module: nn.Module, 
                          initialization: str = 'xavier_uniform') -> None:
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if initialization == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif initialization == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif initialization == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif initialization == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif initialization == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            elif initialization == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"不支持的初始化方法: {initialization}")
            
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    @staticmethod
    def create_optimizer(model: nn.Module, 
                        optimizer_name: str = 'adam',
                        learning_rate: float = 0.001,
                        weight_decay: float = 0.0,
                        **kwargs) -> torch.optim.Optimizer:
        """创建优化器"""
        # 确保eps是浮点数
        if 'eps' in kwargs:
            kwargs['eps'] = float(kwargs['eps'])
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9),
                **{k: v for k, v in kwargs.items() if k != 'momentum'}
            )
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer,
                        scheduler_name: str = 'reduce_on_plateau',
                        **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if scheduler_name == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=kwargs.get('patience', 5),
                factor=kwargs.get('factor', 0.5),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_name == 'cosine_warmup':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_name is None:
            return None
        else:
            raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
    
    @staticmethod
    def create_loss_function(loss_name: str = 'mse',
                            **kwargs) -> nn.Module:
        """创建损失函数"""
        if loss_name == 'mse':
            return nn.MSELoss(**kwargs)
        elif loss_name == 'mae':
            return nn.L1Loss(**kwargs)
        elif loss_name == 'huber':
            return nn.HuberLoss(**kwargs)
        elif loss_name == 'smooth_l1':
            return nn.SmoothL1Loss(**kwargs)
        elif loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss(**kwargs)
        elif loss_name == 'bce':
            return nn.BCELoss(**kwargs)
        elif loss_name == 'bce_with_logits':
            return nn.BCEWithLogitsLoss(**kwargs)
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
    
    @staticmethod
    def create_early_stopping(patience: int = 10,
                             min_delta: float = 0.001,
                             restore_best_weights: bool = True) -> 'EarlyStopping':
        """创建早停策略"""
        class EarlyStopping:
            def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
                self.patience = patience
                self.min_delta = min_delta
                self.restore_best_weights = restore_best_weights
                self.best_loss = float('inf')
                self.counter = 0
                self.best_state_dict = None
                self.early_stop = False
            
            def __call__(self, val_loss, model=None):
                if val_loss < self.best_loss - self.min_delta:
                    self.best_loss = val_loss
                    self.counter = 0
                    if self.restore_best_weights and model is not None:
                        self.best_state_dict = model.state_dict().copy()
                    return False  # 继续训练
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                        return True  # 停止训练
                    return False
            
            def restore_best_model(self, model):
                if self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
        
        return EarlyStopping(patience, min_delta, restore_best_weights)
    
    @staticmethod
    def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor],
                         y_pred: Union[np.ndarray, torch.Tensor],
                         metrics: List[str]) -> Dict[str, float]:
        """计算评估指标"""
        from sklearn.metrics import (
            r2_score, mean_squared_error, mean_absolute_error,
            mean_absolute_percentage_error, accuracy_score,
            precision_score, recall_score, f1_score, roc_auc_score
        )
        
        # 转换为numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        results = {}
        
        for metric in metrics:
            if metric == 'r2':
                results['r2'] = float(r2_score(y_true, y_pred))
            elif metric == 'rmse':
                results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif metric == 'mse':
                results['mse'] = float(mean_squared_error(y_true, y_pred))
            elif metric == 'mae':
                results['mae'] = float(mean_absolute_error(y_true, y_pred))
            elif metric == 'mape':
                try:
                    results['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
                except:
                    results['mape'] = float('nan')
            elif metric == 'accuracy':
                # 需要转换为分类
                y_pred_class = np.round(y_pred).astype(int)
                y_true_class = np.round(y_true).astype(int)
                results['accuracy'] = float(accuracy_score(y_true_class, y_pred_class))
            elif metric == 'precision':
                y_pred_class = np.round(y_pred).astype(int)
                y_true_class = np.round(y_true).astype(int)
                results['precision'] = float(precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0))
            elif metric == 'recall':
                y_pred_class = np.round(y_pred).astype(int)
                y_true_class = np.round(y_true).astype(int)
                results['recall'] = float(recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0))
            elif metric == 'f1':
                y_pred_class = np.round(y_pred).astype(int)
                y_true_class = np.round(y_true).astype(int)
                results['f1'] = float(f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0))
            elif metric == 'auc':
                try:
                    results['auc'] = float(roc_auc_score(y_true, y_pred))
                except:
                    results['auc'] = float('nan')
        
        return results
    
    @staticmethod
    def create_checkpoint(model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         metrics: Dict,
                         file_path: str) -> None:
        """创建模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, file_path)
    
    @staticmethod
    def load_checkpoint(file_path: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: Optional[torch.device] = None) -> Dict:
        """加载模型检查点"""
        if device is None:
            device = torch.device('cpu')
        
        checkpoint = torch.load(file_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    @staticmethod
    def freeze_layers(model: nn.Module, 
                     layer_names: List[str]) -> None:
        """冻结指定层"""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    @staticmethod
    def unfreeze_layers(model: nn.Module,
                       layer_names: List[str]) -> None:
        """解冻指定层"""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
    
    @staticmethod
    def get_layer_outputs(model: nn.Module,
                         input_data: torch.Tensor,
                         layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """获取指定层的输出"""
        outputs = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                outputs[name] = output.detach()
            return hook
        
        # 注册钩子
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            model.eval()
            _ = model(input_data)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return outputs
    
    @staticmethod
    def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
        """计算梯度范数"""
        gradient_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_norms[name] = float(param.grad.norm().item())
        
        return gradient_norms
    
    @staticmethod
    def compute_weight_norms(model: nn.Module) -> Dict[str, float]:
        """计算权重范数"""
        weight_norms = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_norms[name] = float(param.norm().item())
        
        return weight_norms
    
    @staticmethod
    def create_model_summary(model: nn.Module,
                            input_size: Tuple) -> str:
        """创建模型摘要"""
        from torchsummary import summary
        
        device = next(model.parameters()).device
        summary_str = str(summary(model, input_size, device=device.type))
        
        return summary_str
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """优化模型用于推理"""
        model.eval()
        
        # 融合BatchNorm层
        if hasattr(model, 'fuse_model'):
            model.fuse_model()
        
        # 转换为推理模式
        model = torch.jit.script(model)
        
        return model