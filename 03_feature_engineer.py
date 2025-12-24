import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineer:
    """特征工程 - 创建分析所需特征"""

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def create_consumption_features(self, df):
        """创建消费结构特征"""
        # 确保百分比列存在
        required_cols = []
        all_cols = []

        # 收集所有消费类别列
        for category in ['survival', 'development', 'enjoyment']:
            cols = self.config['features']['consumption_categories'][category]
            cols_exist = [col for col in cols if col in df.columns]
            required_cols.extend(cols_exist)
            all_cols.append({
                'category': category,
                'columns': cols_exist
            })

        # 检查是否有足够的列
        if len(required_cols) < 3:
            raise ValueError(f"消费数据不足，找到的列: {required_cols}")

        # 计算各类别占比
        for item in all_cols:
            if item['columns']:
                df[f'{item["category"]}_消费占比'] = df[item['columns']].mean(axis=1)

        # 计算总消费占比
        total_cols = [col for col in df.columns if '消费占比' in col]
        df['总消费占比'] = df[total_cols].sum(axis=1)

        # 标准化处理
        if self.config['features'].get('normalize', True):
            proportion_cols = [col for col in df.columns if '消费占比' in col]
            df[proportion_cols] = df[proportion_cols].clip(lower=0, upper=100)

        return df

    def create_attitude_features(self, df):
        """创建消费态度特征"""
        attitude_config = self.config['features']['attitude_columns']

        # 检查态度列是否存在
        attitude_cols = []
        for key, col_name in attitude_config.items():
            if col_name in df.columns:
                attitude_cols.append(col_name)
                # 重命名以便后续使用
                df.rename(columns={col_name: f'{key}_得分'}, inplace=True)
            else:
                self.logger.warning(f"态度列不存在: {col_name}")

        # 如果有多个态度列，创建综合得分
        if len(attitude_cols) >= 2:
            df['理性倾向综合得分'] = df[[col for col in df.columns
                                         if '性价比' in col or '理性' in col]].mean(axis=1)
            df['感性倾向综合得分'] = df[[col for col in df.columns
                                         if '情感' in col or '感性' in col]].mean(axis=1)

            # 创建态度类型标签
            conditions = [
                (df['理性倾向综合得分'] > df['感性倾向综合得分']),
                (df['理性倾向综合得分'] < df['感性倾向综合得分']),
                (df['理性倾向综合得分'] == df['感性倾向综合得分'])
            ]
            choices = ['理性主导', '感性主导', '平衡型']
            df['消费态度类型'] = np.select(conditions, choices, default='未知')

        return df

    def create_demographic_features(self, df):
        """创建人口统计特征"""
        # 收入分段
        if '月可支配收入' in df.columns:
            income_bins = self.config.get('income_bins', [0, 1500, 2500, 3500, 10000])
            income_labels = self.config.get('income_labels',
                                            ['低收入', '中低收入', '中高收入', '高收入'])

            df['收入分段'] = pd.cut(df['月可支配收入'],
                                    bins=income_bins,
                                    labels=income_labels,
                                    include_lowest=True)

        # 年级编码
        if '年级' in df.columns:
            grade_order = {'大一': 1, '大二': 2, '大三': 3, '大四': 4}
            df['年级编码'] = df['年级'].map(grade_order)
            df['年级编码'] = df['年级编码'].fillna(0).astype(int)

        return df

    def run(self, df):
        """执行特征工程流程"""
        self.logger.info("开始特征工程")

        df = self.create_consumption_features(df)
        df = self.create_attitude_features(df)
        df = self.create_demographic_features(df)

        self.logger.info(f"特征工程完成，新增列: {[col for col in df.columns if col not in self.original_columns]}")

        return df