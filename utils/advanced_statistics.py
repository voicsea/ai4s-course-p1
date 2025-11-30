"""
高级统计分析与显著性检验模块
Statistical Analysis & Significance Testing Module

Features:
  - Pairwise T-Test (配对 t 检验)
  - 多重比较校正 (Multiple Comparison Correction)
  - 论文级显著性标记
  - LaTeX 表格自动生成
"""

import numpy as np
from typing import Dict, Tuple, List
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class AdvancedStatisticalAnalyzer:
    """高级统计分析器"""
    
    @staticmethod
    def ttest_independent(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
        """
        独立样本 t 检验
        
        Args:
            sample1, sample2: 两个样本数组
        
        Returns:
            (t_statistic, p_value)
        """
        try:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(sample1, sample2)
            return float(t_stat), float(p_val)
        except ImportError:
            # Fallback: 手动计算
            return AdvancedStatisticalAnalyzer._manual_ttest(sample1, sample2)
    
    @staticmethod
    def _manual_ttest(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
        """
        手动计算独立样本 t 检验（当 scipy 不可用时）
        """
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        # 合并方差
        pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
        
        # t 统计量
        t_stat = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2) + 1e-8)
        
        # 自由度
        df = n1 + n2 - 2
        
        # 近似 p 值（使用 t 分布）
        # 这里使用简单近似
        from scipy import stats
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return float(t_stat), float(p_val)
    
    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Bonferroni 多重比较校正
        
        Args:
            p_values: p 值列表
            alpha: 显著性水平
        
        Returns:
            是否显著的布尔列表
        """
        corrected_alpha = alpha / len(p_values)
        return [p < corrected_alpha for p in p_values]
    
    @staticmethod
    def format_pvalue(p_value: float, significance_level: float = 0.05) -> str:
        """
        格式化 p 值为显著性标记
        
        Returns:
            '**' for p < 0.01
            '*'  for p < 0.05
            ''   otherwise
        """
        if p_value < 0.01:
            return "**"
        elif p_value < significance_level:
            return "*"
        else:
            return ""
    
    @staticmethod
    def perform_pairwise_comparison(results: Dict, metric: str = 'PSNR') -> Dict:
        """
        执行两两比较
        
        Args:
            results: 所有模型的评估结果
            metric: 要比较的指标名称
        
        Returns:
            包含 p 值和显著性标记的字典
        """
        model_names = list(results.keys())
        n_models = len(model_names)
        
        # 初始化比较矩阵
        pvalue_matrix = np.ones((n_models, n_models))
        significance_matrix = {}
        
        for i, model1 in enumerate(model_names):
            significance_matrix[model1] = {}
            
            for j, model2 in enumerate(model_names):
                if i == j:
                    continue
                
                # 获取两个模型的指标值
                if metric not in results[model1]['metrics'] or \
                   metric not in results[model2]['metrics']:
                    continue
                
                vals1 = np.array(results[model1]['metrics'][metric].get('all_values', []))
                vals2 = np.array(results[model2]['metrics'][metric].get('all_values', []))
                
                if len(vals1) == 0 or len(vals2) == 0:
                    continue
                
                # 执行 t 检验
                _, p_val = AdvancedStatisticalAnalyzer.ttest_independent(vals1, vals2)
                pvalue_matrix[i, j] = p_val
                
                # 生成显著性标记
                sig_mark = AdvancedStatisticalAnalyzer.format_pvalue(p_val)
                significance_matrix[model1][model2] = {
                    'p_value': p_val,
                    'significance': sig_mark
                }
        
        return significance_matrix


class PublicationTableGenerator:
    """论文级表格生成器"""
    
    @staticmethod
    def generate_latex_table_with_pvalues(results: Dict, 
                                         significance: Dict,
                                         metrics: List[str] = None) -> str:
        """
        生成包含显著性标记的 LaTeX 表格
        
        Args:
            results: 评估结果
            significance: 显著性检验结果
            metrics: 要包含的指标列表
        
        Returns:
            LaTeX 表格代码
        """
        if metrics is None:
            metrics = ['PSNR', 'SSIM', 'RMSE', 'MAE']
        
        model_names = list(results.keys())
        
        # 找出最佳模型
        best_models = {
            metric: max(
                model_names,
                key=lambda m: results[m]['metrics'].get(metric, {}).get('mean', -np.inf)
                if metric in ['PSNR', 'SSIM', 'R2'] else
                -results[m]['metrics'].get(metric, {}).get('mean', np.inf)
            )
            for metric in metrics
            if metric in results[model_names[0]]['metrics']
        }
        
        # 找出次优模型
        second_best = {}
        for metric in metrics:
            if metric not in results[model_names[0]]['metrics']:
                continue
            
            is_higher_better = metric in ['PSNR', 'SSIM', 'R2']
            
            vals = [
                (m, results[m]['metrics'][metric]['mean'])
                for m in model_names
            ]
            
            if is_higher_better:
                vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
            else:
                vals_sorted = sorted(vals, key=lambda x: x[1])
            
            if len(vals_sorted) > 1:
                second_best[metric] = vals_sorted[1][0]
        
        # 构建表格
        num_cols = 4 + len(metrics)  # Model, GFLOPs, Params, FPS + metrics
        
        latex = "\\begin{table}[ht]\n"
        latex += "\\centering\n"
        latex += "\\small\n"
        latex += f"\\begin{{tabular}}{{l|cccc{'|c'*len(metrics)}}}\n"
        latex += "\\toprule\n"
        
        # 表头
        header_line = "Model & GFLOPs & Params (M) & FPS "
        for metric in metrics:
            header_line += f" & {metric}"
        header_line += " \\\\\n"
        latex += header_line
        latex += "\\midrule\n"
        
        # 数据行
        for model_name in sorted(model_names):
            result = results[model_name]
            
            row_parts = [model_name]
            row_parts.append(f"{result['gflops']:.2f}")
            row_parts.append(f"{result['params']:.2f}")
            row_parts.append(f"{result['fps']:.2f}")
            
            for metric in metrics:
                if metric not in result['metrics']:
                    row_parts.append("-")
                    continue
                
                mean = result['metrics'][metric]['mean']
                std = result['metrics'][metric]['std']
                
                # 判断是否为最佳或次优
                is_best = best_models.get(metric) == model_name
                is_second = second_best.get(metric) == model_name
                
                value_str = f"{mean:.4f}±{std:.4f}"
                
                if is_best:
                    value_str = f"\\textbf{{{value_str}}}"
                elif is_second:
                    value_str = f"\\underline{{{value_str}}}"
                
                # 添加显著性标记
                if metric in ['PSNR', 'SSIM', 'R2']:
                    best_name = best_models.get(metric)
                    if best_name and best_name in significance and \
                       model_name in significance[best_name]:
                        sig = significance[best_name][model_name]['significance']
                        if sig:
                            value_str += f"$^{{{sig}}}$"
                
                row_parts.append(value_str)
            
            row_line = " & ".join(row_parts) + " \\\\\n"
            latex += row_line
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\caption{Comprehensive Model Comparison. "
        latex += "\\textbf{Bold} indicates best performance. "
        latex += "\\underline{Underlined} indicates second best. "
        latex += "$^*$p < 0.05, $^{**}$p < 0.01.}\n"
        latex += "\\label{tab:sota_comparison}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    @staticmethod
    def generate_csv_with_significance(results: Dict, 
                                      significance: Dict,
                                      output_path: Path) -> None:
        """
        生成包含显著性信息的 CSV
        
        Args:
            results: 评估结果
            significance: 显著性检验结果
            output_path: 保存路径
        """
        import csv
        
        model_names = list(results.keys())
        
        rows = []
        for model_name in model_names:
            result = results[model_name]
            
            row = {
                'Model': model_name,
                'GFLOPs': f"{result['gflops']:.2f}",
                'Params_M': f"{result['params']:.2f}",
                'FPS': f"{result['fps']:.2f}",
            }
            
            # 添加指标
            for metric, values in result['metrics'].items():
                row[f'{metric}_mean'] = f"{values['mean']:.4f}"
                row[f'{metric}_std'] = f"{values['std']:.4f}"
                
                # 添加显著性
                if model_name in significance:
                    for other_model, sig_info in significance[model_name].items():
                        if sig_info['significance']:
                            row[f'{metric}_vs_{other_model}'] = sig_info['significance']
            
            rows.append(row)
        
        # 写入 CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


class WorstCaseAnalyzer:
    """最差情况分析器"""
    
    @staticmethod
    def analyze_worst_cases(results: Dict, 
                           output_dir: Path,
                           top_k: int = 5) -> None:
        """
        分析最差情况并生成报告
        
        Args:
            results: 评估结果
            output_dir: 输出目录
            top_k: 显示前 k 个最差样本
        """
        worst_cases_analysis = {}
        
        for model_name, result in results.items():
            worst_cases = result.get('worst_cases', [])[:top_k]
            
            worst_cases_analysis[model_name] = {
                'total_samples': len(result.get('per_sample_losses', [])),
                'worst_cases': worst_cases,
                'avg_loss': float(np.mean(result.get('per_sample_losses', []))),
                'max_loss': float(np.max(result.get('per_sample_losses', []))) if result.get('per_sample_losses') else 0,
                'min_loss': float(np.min(result.get('per_sample_losses', []))) if result.get('per_sample_losses') else 0,
            }
        
        # 保存分析结果
        analysis_path = output_dir / 'worst_cases_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(worst_cases_analysis, f, indent=2)
        
        # 打印摘要
        print("\n" + "="*80)
        print("最差情况分析总结")
        print("="*80)
        for model_name, analysis in worst_cases_analysis.items():
            print(f"\n{model_name}:")
            print(f"  总样本数: {analysis['total_samples']}")
            print(f"  平均 Loss: {analysis['avg_loss']:.6f}")
            print(f"  最大 Loss: {analysis['max_loss']:.6f}")
            print(f"  最小 Loss: {analysis['min_loss']:.6f}")
            print(f"  最差 5 个样本: {[w['sample_id'] for w in analysis['worst_cases']]}")


class MetricsAggregator:
    """指标聚合器"""
    
    @staticmethod
    def aggregate_and_format(results: Dict) -> Dict:
        """
        聚合指标并格式化为论文要求的形式
        
        Args:
            results: 原始评估结果
        
        Returns:
            格式化的聚合结果
        """
        formatted = {}
        
        for model_name, result in results.items():
            formatted[model_name] = {
                'efficiency': {
                    'GFLOPs': f"{result['gflops']:.2f}",
                    'Params_M': f"{result['params']:.2f}",
                    'FPS': f"{result['fps']:.2f}",
                },
                'metrics': {}
            }
            
            for metric, values in result['metrics'].items():
                formatted[model_name]['metrics'][metric] = {
                    'mean': f"{values['mean']:.4f}",
                    'std': f"{values['std']:.4f}",
                    'full': f"{values['mean']:.4f} ± {values['std']:.4f}"
                }
        
        return formatted


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    
    # 创建示例数据
    results = {
        'Model_A': {
            'gflops': 10.5,
            'params': 25.3,
            'fps': 120.4,
            'metrics': {
                'PSNR': {
                    'mean': 28.45,
                    'std': 1.23,
                    'all_values': np.random.normal(28.45, 1.23, 100).tolist()
                },
                'SSIM': {
                    'mean': 0.92,
                    'std': 0.05,
                    'all_values': np.random.normal(0.92, 0.05, 100).tolist()
                },
                'RMSE': {
                    'mean': 5.67,
                    'std': 0.89,
                    'all_values': np.random.normal(5.67, 0.89, 100).tolist()
                }
            },
            'worst_cases': [
                {'sample_id': 5, 'loss': 15.23},
                {'sample_id': 12, 'loss': 14.56}
            ],
            'per_sample_losses': [1.2, 2.3, 3.4] * 34
        },
        'Model_B': {
            'gflops': 15.2,
            'params': 45.6,
            'fps': 95.3,
            'metrics': {
                'PSNR': {
                    'mean': 27.89,
                    'std': 1.45,
                    'all_values': np.random.normal(27.89, 1.45, 100).tolist()
                },
                'SSIM': {
                    'mean': 0.90,
                    'std': 0.06,
                    'all_values': np.random.normal(0.90, 0.06, 100).tolist()
                },
                'RMSE': {
                    'mean': 6.12,
                    'std': 0.95,
                    'all_values': np.random.normal(6.12, 0.95, 100).tolist()
                }
            },
            'worst_cases': [],
            'per_sample_losses': [1.5, 2.8, 3.9] * 34
        }
    }
    
    # 执行统计分析
    analyzer = AdvancedStatisticalAnalyzer()
    significance = analyzer.perform_pairwise_comparison(results, 'PSNR')
    
    # 生成 LaTeX 表格
    table_generator = PublicationTableGenerator()
    latex_code = table_generator.generate_latex_table_with_pvalues(
        results, significance
    )
    
    print("LaTeX 表格代码:")
    print(latex_code)
    
    # 分析最差情况
    worst_analyzer = WorstCaseAnalyzer()
    worst_analyzer.analyze_worst_cases(results, Path('./output'))
    
    # 聚合指标
    agg = MetricsAggregator()
    formatted = agg.aggregate_and_format(results)
    print("\n格式化后的指标:")
    print(json.dumps(formatted, indent=2))


if __name__ == '__main__':
    example_usage()
