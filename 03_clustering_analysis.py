import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
import logging


class ConsumptionClustering:
    """消费画像聚类分析"""

    def __init__(self, config):
        self.config = config['clustering']
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.results = {}

    def select_features(self, df):
        """选择用于聚类的特征"""
        feature_cols = []

        for feature in self.config['features']:
            if feature in df.columns:
                feature_cols.append(feature)
            else:
                # 尝试查找类似的列
                similar_cols = [col for col in df.columns if feature in col]
                if similar_cols:
                    feature_cols.extend(similar_cols[:1])  # 只取第一个匹配的

        self.logger.info(f"选择的聚类特征: {feature_cols}")

        if len(feature_cols) < 2:
            raise ValueError("至少需要2个特征进行聚类分析")

        return df[feature_cols].copy()

    def determine_optimal_clusters(self, X_scaled):
        """确定最佳聚类数量"""
        n_clusters_range = range(2, 11)
        inertia_values = []
        silhouette_scores = []

        self.logger.info("寻找最佳聚类数...")

        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=self.config['hyperparams']['random_state'],
                            n_init=self.config['hyperparams']['n_init'])
            kmeans.fit(X_scaled)

            inertia_values.append(kmeans.inertia_)

            if n_clusters > 1:  # silhouette_score需要至少2个聚类
                silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(silhouette_avg)

        # 寻找肘点
        diff = np.diff(inertia_values)
        diff_ratio = diff[1:] / diff[:-1]

        if len(diff_ratio) > 0:
            elbow_point = np.argmin(diff_ratio) + 3  # +3是因为从2开始，且diff少一个
        else:
            elbow_point = 4  # 默认值

        # 选择最佳聚类数
        if 'n_clusters' in self.config and self.config['n_clusters'] > 0:
            optimal_k = self.config['n_clusters']
            self.logger.info(f"使用配置的聚类数: {optimal_k}")
        else:
            # 自动选择：肘点法和轮廓系数法的平衡
            if silhouette_scores:
                best_silhouette = np.argmax(silhouette_scores) + 2
                optimal_k = int((elbow_point + best_silhouette) / 2)
            else:
                optimal_k = elbow_point

            optimal_k = max(3, min(optimal_k, 6))  # 限制在3-6之间
            self.logger.info(f"自动选择的聚类数: {optimal_k} (肘点: {elbow_point})")

        # 绘制肘部法则图
        self._plot_elbow_method(n_clusters_range, inertia_values,
                                silhouette_scores if silhouette_scores else None,
                                optimal_k)

        return optimal_k

    def perform_clustering(self, X_scaled, n_clusters):
        """执行K-means聚类"""
        self.logger.info(f"执行K-means聚类，k={n_clusters}")

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.config['hyperparams']['random_state'],
            n_init=self.config['hyperparams']['n_init'],
            max_iter=self.config['hyperparams']['max_iter']
        )

        labels = self.model.fit_predict(X_scaled)

        # 计算评估指标
        if n_clusters > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)

            self.logger.info(f"轮廓系数: {silhouette_avg:.3f}")
            self.logger.info(f"Davies-Bouldin指数: {db_score:.3f}")

            self.results['metrics'] = {
                'silhouette_score': silhouette_avg,
                'davies_bouldin_score': db_score,
                'inertia': self.model.inertia_
            }

        return labels

    def analyze_clusters(self, df, features, labels):
        """分析每个聚类的特征"""
        df['cluster_label'] = labels

        cluster_analysis = {}

        for cluster_id in range(self.model.n_clusters):
            cluster_data = df[df['cluster_label'] == cluster_id]

            analysis = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'features_mean': {},
                'demographics': {}
            }

            # 特征均值
            for feature in features:
                if feature in cluster_data.columns:
                    analysis['features_mean'][feature] = cluster_data[feature].mean()

            # 人口统计信息
            if '年级' in cluster_data.columns:
                analysis['demographics']['grade_dist'] = cluster_data['年级'].value_counts().to_dict()

            if '收入分段' in cluster_data.columns:
                analysis['demographics']['income_dist'] = cluster_data['收入分段'].value_counts().to_dict()

            # 命名聚类（基于特征）
            analysis['suggested_name'] = self._name_cluster(analysis['features_mean'])

            cluster_analysis[cluster_id] = analysis

            self.logger.info(f"聚类{cluster_id} ({analysis['suggested_name']}): "
                             f"{analysis['size']}人 ({analysis['percentage']:.1f}%)")

        self.results['cluster_analysis'] = cluster_analysis
        self.results['cluster_centers'] = self.model.cluster_centers_

        return cluster_analysis

    def _name_cluster(self, feature_means):
        """根据特征为聚类命名"""
        names = self.config.get('portrait_names', {})

        for cluster_id, name in names.items():
            if isinstance(cluster_id, int) and cluster_id < len(names):
                return name
        if '发展型消费占比' in feature_means and '性价比倾向' in feature_means:
            dev_score = feature_means['发展型消费占比']
            value_score = feature_means['性价比倾向']

            if dev_score > 30:
                return "价值投资型"
            elif value_score > 3.5:
                return "精明务实型"
            else:
                return "情感驱动型"

        return f"聚类_{list(feature_means.keys())[0][:10]}"

    def _plot_elbow_method(self, k_range, inertias, silhouettes, optimal_k):
        """绘制肘部法则图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 肘部图
        ax1.plot(k_range, inertias, 'bo-', linewidth=2)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('聚类数量 (k)', fontsize=12)
        ax1.set_ylabel('误差平方和', fontsize=12)
        ax1.set_title('肘部法则', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 轮廓系数图
        if silhouettes:
            ax2.plot(range(2, len(silhouettes) + 2), silhouettes, 'go-', linewidth=2)
            ax2.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
            ax2.set_xlabel('聚类数量 (k)', fontsize=12)
            ax2.set_ylabel('轮廓系数', fontsize=12)
            ax2.set_title('轮廓系数法', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/figures/clustering_elbow_method.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, filepath='models/kmeans_model.pkl'):
        """保存训练好的模型"""
        if self.model:
            joblib.dump(self.model, filepath)
            self.logger.info(f"模型已保存至: {filepath}")

    def run(self, df):
        """执行完整的聚类分析流程"""
        self.logger.info("开始聚类分析")

        # 1. 选择特征
        X = self.select_features(df)

        # 2. 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. 确定最佳聚类数
        optimal_k = self.determine_optimal_clusters(X_scaled)

        # 4. 执行聚类
        labels = self.perform_clustering(X_scaled, optimal_k)

        # 5. 分析结果
        cluster_analysis = self.analyze_clusters(df, X.columns, labels)

        # 6. 保存模型
        self.save_model()

        self.logger.info("聚类分析完成")

        return {
            'labels': labels,
            'model': self.model,
            'analysis': cluster_analysis,
            'results': self.results
        }