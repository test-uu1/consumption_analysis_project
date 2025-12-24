import pandas as pd
import numpy as np
from pathlib import Path
import logging


class DataLoader:
    """数据加载器 - 负责读取各种格式的数据"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_survey_data(self, filepath):
        """加载问卷数据"""
        self.logger.info(f"正在加载问卷数据: {filepath}")

        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'latin1']

        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                self.logger.info(f"使用 {encoding} 编码成功加载数据")
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"无法读取文件: {filepath}")

    def load_external_data(self, directory):
        """加载外部数据（宏观报告等）"""
        data_dict = {}

        for filepath in Path(directory).glob("*.csv"):
            df_name = filepath.stem
            try:
                data_dict[df_name] = pd.read_csv(filepath)
                self.logger.info(f"加载外部数据: {df_name}")
            except Exception as e:
                self.logger.warning(f"无法加载 {filepath}: {str(e)}")

        return data_dict

    def validate_data(self, df, data_type='survey'):
        """数据基本验证"""
        self.logger.info("开始数据验证")

        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }

        # 列名标准化
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]

        # 检查必要列是否存在
        required_cols = self.config.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.logger.warning(f"缺少必要列: {missing_cols}")

        self.logger.info(f"验证完成: {validation_report}")

        return df, validation_report