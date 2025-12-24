import argparse
import yaml
import sys
from pathlib import Path

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_processing import DataPipeline
from src.analysis import AnalysisEngine
from src.visualization import Visualizer
from src.reporting import ReportGenerator
from utils.logger import setup_logger


def main():
    """主执行函数"""
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='大学生消费趋势分析系统')
    parser.add_argument('--config', type=str, default='config/analysis_params.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['preprocess', 'analyze', 'visualize', 'report', 'full'],
                        help='运行模式')
    args = parser.parse_args()

    # 2. 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. 初始化日志
    logger = setup_logger(config['logging'])
    logger.info("开始大学生消费趋势分析")

    try:
        # 4. 数据预处理
        if args.mode in ['preprocess', 'full']:
            logger.info("阶段1: 数据预处理")
            data_pipeline = DataPipeline(config['data'])
            df_cleaned = data_pipeline.run()
            logger.info(f"数据清洗完成，样本数: {len(df_cleaned)}")

        # 5. 数据分析
        if args.mode in ['analyze', 'full']:
            logger.info("阶段2: 数据分析")
            analyzer = AnalysisEngine(config['analysis'])
            analysis_results = analyzer.run(df_cleaned)
            logger.info("分析完成")

        # 6. 可视化生成
        if args.mode in ['visualize', 'full']:
            logger.info("阶段3: 可视化生成")
            visualizer = Visualizer(config['visualization'])
            visualizer.generate_all(analysis_results, df_cleaned)
            logger.info("可视化图表生成完成")

        # 7. 报告生成
        if args.mode in ['report', 'full']:
            logger.info("阶段4: 报告生成")
            reporter = ReportGenerator(config['reporting'])
            reporter.generate_full_report(analysis_results)
            logger.info("报告生成完成")

        logger.info("所有任务完成!")

    except Exception as e:
        logger.error(f"执行过程中出现错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()