#!/usr/bin/env python3
"""
数据工具函数
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DataUtils:
    """数据工具类"""
    
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict:
        """加载配置文件"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def load_data(file_path: str, **kwargs) -> pd.DataFrame:
        """加载数据文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 根据文件扩展名选择加载方式
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix == '.feather':
            return pd.read_feather(file_path, **kwargs)
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    @staticmethod
    def save_data(data: Union[pd.DataFrame, np.ndarray], 
                  file_path: str, 
                  **kwargs) -> None:
        """保存数据文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据文件扩展名选择保存方式
        if file_path.suffix == '.csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, **kwargs)
            else:
                pd.DataFrame(data).to_csv(file_path, **kwargs)
        elif file_path.suffix == '.parquet':
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path, **kwargs)
            else:
                pd.DataFrame(data).to_parquet(file_path, **kwargs)
        elif file_path.suffix == '.feather':
            if isinstance(data, pd.DataFrame):
                data.to_feather(file_path, **kwargs)
            else:
                pd.DataFrame(data).to_feather(file_path, **kwargs)
        elif file_path.suffix == '.pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict:
        """检查数据质量"""
        quality_report = {
            'sample_count': len(df),
            'feature_count': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # 检查缺失值
        missing = df.isnull().sum()
        quality_report['missing_values']['total'] = int(missing.sum())
        quality_report['missing_values']['percentage'] = float(missing.sum() / (len(df) * len(df.columns)))
        
        if missing.sum() > 0:
            quality_report['missing_values']['by_column'] = {
                col: int(count) for col, count in missing[missing > 0].items()
            }
        
        # 数据类型统计
        quality_report['data_types'] = df.dtypes.value_counts().to_dict()
        
        # 数值型特征统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            quality_report['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'skew': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        # 分类变量统计
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            quality_report['categorical_stats'][col] = {
                'unique_count': int(value_counts.shape[0]),
                'top_values': value_counts.head(5).to_dict()
            }
        
        return quality_report
    
    @staticmethod
    def split_data(df: pd.DataFrame, 
                   target_col: str,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42,
                   stratify_col: Optional[str] = None) -> Tuple:
        """划分训练集、验证集、测试集"""
        from sklearn.model_selection import train_test_split
        
        # 分离特征和目标
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 分层策略
        stratify = df[stratify_col] if stratify_col else None
        
        # 先划分训练+验证和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # 调整验证集大小（相对于剩余数据）
        val_size_adjusted = val_size / (1 - test_size)
        
        # 再划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify.iloc[X_temp.index] if stratify is not None else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """创建特征交互项"""
        df_interaction = df.copy()
        
        for var1, var2 in feature_pairs:
            if var1 in df.columns and var2 in df.columns:
                interaction_name = f"{var1}_{var2}_interaction"
                df_interaction[interaction_name] = df[var1] * df[var2]
        
        return df_interaction
    
    @staticmethod
    def create_time_segments(df: pd.DataFrame, 
                            time_col: str,
                            segments: List[int]) -> pd.DataFrame:
        """创建时间分段特征"""
        df_time = df.copy()
        
        if time_col not in df.columns:
            return df_time
        
        # 创建时间分段
        for i in range(len(segments) - 1):
            segment_name = f"time_{segments[i]}_{segments[i+1]}"
            mask = (df[time_col] >= segments[i]) & (df[time_col] < segments[i+1])
            df_time[segment_name] = mask.astype(int)
        
        return df_time
    
    @staticmethod
    def normalize_features(df: pd.DataFrame, 
                          method: str = 'standard',
                          exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """特征标准化"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        df_norm = df.copy()
        scalers = {}
        
        # 排除列
        if exclude_cols is None:
            exclude_cols = []
        
        # 只处理数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
            
            # 重塑数据以适应scaler
            data = df[col].values.reshape(-1, 1)
            df_norm[col] = scaler.fit_transform(data).flatten()
            scalers[col] = scaler
        
        return df_norm, scalers
    
    @staticmethod
    def encode_categorical_features(df: pd.DataFrame,
                                   method: str = 'onehot',
                                   exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """分类特征编码"""
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        
        df_encoded = df.copy()
        encoders = {}
        
        # 排除列
        if exclude_cols is None:
            exclude_cols = []
        
        # 只处理分类特征
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        for col in categorical_cols:
            if method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[[col]])
                
                # 创建新列名
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # 添加新列
                for i, feature_name in enumerate(feature_names):
                    df_encoded[feature_name] = encoded[:, i]
                
                # 删除原始列
                df_encoded = df_encoded.drop(columns=[col])
                
            elif method == 'label':
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col])
            
            else:
                raise ValueError(f"不支持的编码方法: {method}")
            
            encoders[col] = encoder
        
        return df_encoded, encoders
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict:
        """检测异常值"""
        outliers_report = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            if method == 'iqr':
                # IQR方法
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
            elif method == 'zscore':
                # Z-score方法
                mean = data.mean()
                std = data.std()
                
                z_scores = np.abs((data - mean) / std)
                outliers = data[z_scores > threshold]
            
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
            
            outliers_report[col] = {
                'outlier_count': int(len(outliers)),
                'outlier_percentage': float(len(outliers) / len(data)),
                'outlier_indices': outliers.index.tolist(),
                'outlier_values': outliers.values.tolist()
            }
        
        return outliers_report
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, 
                             method: str = 'mean',
                             categorical_method: str = 'mode') -> pd.DataFrame:
        """处理缺失值"""
        df_filled = df.copy()
        
        # 处理数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if method == 'mean':
                    fill_value = df[col].mean()
                elif method == 'median':
                    fill_value = df[col].median()
                elif method == 'mode':
                    fill_value = df[col].mode()[0]
                elif method == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"不支持的数值型缺失值处理方法: {method}")
                
                df_filled[col] = df[col].fillna(fill_value)
        
        # 处理分类特征
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if df[col].isnull().any():
                if categorical_method == 'mode':
                    fill_value = df[col].mode()[0]
                elif categorical_method == 'unknown':
                    fill_value = 'unknown'
                else:
                    raise ValueError(f"不支持的分类型缺失值处理方法: {categorical_method}")
                
                df_filled[col] = df[col].fillna(fill_value)
        
        return df_filled
    
    @staticmethod
    def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
        """创建特征摘要"""
        summary_data = []
        
        for col in df.columns:
            col_info = {
                'feature': col,
                'dtype': str(df[col].dtype),
                'non_null_count': int(df[col].count()),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df))
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'unique_count': int(df[col].nunique())
                })
            else:
                col_info.update({
                    'unique_count': int(df[col].nunique()),
                    'top_value': str(df[col].mode()[0]) if not df[col].mode().empty else 'N/A',
                    'top_value_count': int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0
                })
            
            summary_data.append(col_info)
        
        return pd.DataFrame(summary_data)