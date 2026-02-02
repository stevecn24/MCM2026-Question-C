import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DWTSAnalyzer:
    def __init__(self, fan_votes_path, judge_scores_path):
        try:
            self.fan_votes = pd.read_excel(fan_votes_path)
            self.judge_scores = pd.read_excel(judge_scores_file)
            self.combined_data = None
        except Exception as e:
            print(f"文件读取错误: {e}")
            raise
    
    def preprocess_data(self):
        """预处理数据，合并粉丝投票和评委分数"""
        try:
            # 清洗粉丝投票数据
            # 假设第一列是Season
            self.fan_votes['Season'] = self.fan_votes.iloc[:, 0]
            self.fan_votes['Week'] = self.fan_votes['Week']
            self.fan_votes['Name'] = self.fan_votes['Name'].str.strip()
            
            # 确保Judge_Score列是数值类型
            if 'Judge_Score' in self.fan_votes.columns:
                self.fan_votes['Judge_Score'] = pd.to_numeric(self.fan_votes['Judge_Score'], errors='coerce')
            
            # 清洗评委分数数据
            # 转换所有评委分数列为数值类型
            judge_columns = [col for col in self.judge_scores.columns if 'judge' in col.lower() and 'score' in col.lower()]
            
            for col in judge_columns:
                self.judge_scores[col] = pd.to_numeric(self.judge_scores[col], errors='coerce')
            
            # 计算每周的评委总分
            self.judge_scores['Total_Judge_Score'] = self.judge_scores[judge_columns].sum(axis=1, skipna=True)
            
            # 标准化列名以便合并
            self.judge_scores['Name'] = self.judge_scores['celebrity_name'].str.strip()
            
            # 确保Season列是整数类型
            self.fan_votes['Season'] = pd.to_numeric(self.fan_votes['Season'], errors='coerce')
            self.judge_scores['season'] = pd.to_numeric(self.judge_scores['season'], errors='coerce')
            
            # 合并数据集 - 使用多列匹配
            self.combined_data = pd.merge(
                self.fan_votes,
                self.judge_scores[['Name', 'season', 'Total_Judge_Score', 'results']],
                left_on=['Name', 'Season'],
                right_on=['Name', 'season'],
                how='left'
            )
            
            # 重命名列以保持一致性
            self.combined_data.rename(columns={'Total_Judge_Score': 'Judge_Score_Combined'}, inplace=True)
            
            # 如果没有匹配到Judge_Score_Combined，使用原来的Judge_Score
            if 'Judge_Score' in self.combined_data.columns:
                mask = self.combined_data['Judge_Score_Combined'].isna()
                self.combined_data.loc[mask, 'Judge_Score_Combined'] = self.combined_data.loc[mask, 'Judge_Score']
            
            # 统一使用Judge_Score_Combined作为评委分数
            self.combined_data['Judge_Score'] = self.combined_data['Judge_Score_Combined']
            
            # 删除不需要的列
            columns_to_drop = ['Judge_Score_Combined', 'season']
            for col in columns_to_drop:
                if col in self.combined_data.columns:
                    self.combined_data.drop(columns=[col], inplace=True)
            
            print(f"数据合并完成，共 {len(self.combined_data)} 条记录")
            print(f"列名: {self.combined_data.columns.tolist()}")
            
            # 显示前几行数据以验证
            print("\n合并后数据示例:")
            print(self.combined_data[['Season', 'Week', 'Name', 'Fan_Vote_Mean', 'Judge_Score', 'Result', 'Method']].head())
            
        except Exception as e:
            print(f"数据预处理错误: {e}")
            import traceback
            traceback.print_exc()
            raise


class VotingSystemSimulator:
    @staticmethod
    def rank_method_simulation(week_data):
        """
        排名制模拟：评委排名 + 粉丝投票排名
        """
        week_data = week_data.copy()
        
        # 确保有必要的列
        required_cols = ['Judge_Score', 'Fan_Vote_Mean']
        for col in required_cols:
            if col not in week_data.columns:
                print(f"缺少列: {col}")
                week_data['Predicted_Eliminated'] = False
                return week_data
        
        try:
            # 计算评委排名（分数越高，排名数字越小）
            week_data['Judge_Rank'] = week_data['Judge_Score'].rank(ascending=False, method='min').astype(int)
            
            # 计算粉丝投票排名（投票比例越高，排名数字越小）
            week_data['Fan_Rank'] = week_data['Fan_Vote_Mean'].rank(ascending=False, method='min').astype(int)
            
            # 总排名 = 评委排名 + 粉丝排名
            week_data['Total_Rank'] = week_data['Judge_Rank'] + week_data['Fan_Rank']
            
            # 最可能被淘汰的是总排名数字最大的选手
            max_rank = week_data['Total_Rank'].max()
            week_data['Predicted_Eliminated'] = week_data['Total_Rank'] == max_rank
            
            return week_data
        except Exception as e:
            print(f"排名制模拟错误: {e}")
            week_data['Predicted_Eliminated'] = False
            return week_data
    
    @staticmethod
    def percent_method_simulation(week_data, judge_max=40):
        """
        百分比制模拟：评委分数百分比 + 粉丝投票百分比
        """
        week_data = week_data.copy()
        
        # 确保有必要的列
        required_cols = ['Judge_Score', 'Fan_Vote_Mean']
        for col in required_cols:
            if col not in week_data.columns:
                print(f"缺少列: {col}")
                week_data['Predicted_Eliminated'] = False
                return week_data
        
        try:
            # 计算评委百分比（标准化到0-1）
            if judge_max > 0 and week_data['Judge_Score'].max() > 0:
                week_data['Judge_Percent'] = week_data['Judge_Score'] / judge_max
            else:
                # 如果没有最大值信息，使用排名百分比
                week_data['Judge_Percent'] = week_data['Judge_Score'].rank(pct=True)
            
            # 粉丝投票比例已经是百分比形式
            week_data['Fan_Percent'] = week_data['Fan_Vote_Mean']
            
            # 综合得分 = 评委百分比 + 粉丝百分比
            week_data['Combined_Score'] = week_data['Judge_Percent'] + week_data['Fan_Percent']
            
            # 最可能被淘汰的是综合得分最低的选手
            min_score = week_data['Combined_Score'].min()
            week_data['Predicted_Eliminated'] = week_data['Combined_Score'] == min_score
            
            return week_data
        except Exception as e:
            print(f"百分比制模拟错误: {e}")
            week_data['Predicted_Eliminated'] = False
            return week_data


class AccuracyEvaluator:
    def __init__(self, simulation_results):
        self.results = simulation_results
        
    def calculate_accuracy(self):
        """计算模型预测淘汰的准确性"""
        try:
            # 确保结果列存在
            if 'Result' not in self.results.columns or 'Predicted_Eliminated' not in self.results.columns:
                print("缺少必要的列: 'Result' 或 'Predicted_Eliminated'")
                return {'overall_accuracy': 0, 'classification_report': {}}
            
            # 将结果转换为布尔值
            actual = (self.results['Result'] == 'Eliminated').astype(bool)
            predicted = self.results['Predicted_Eliminated'].astype(bool)
            
            accuracy = accuracy_score(actual, predicted)
            
            # 按投票方法分类统计
            rank_accuracy = np.nan
            percent_accuracy = np.nan
            
            if 'Method' in self.results.columns:
                # 排名制准确率
                rank_mask = self.results['Method'] == 'rank'
                if rank_mask.any():
                    rank_actual = actual[rank_mask]
                    rank_predicted = predicted[rank_mask]
                    if len(rank_actual) > 0:
                        rank_accuracy = accuracy_score(rank_actual, rank_predicted)
                
                # 百分比制准确率
                percent_mask = self.results['Method'] == 'percent'
                if percent_mask.any():
                    percent_actual = actual[percent_mask]
                    percent_predicted = predicted[percent_mask]
                    if len(percent_actual) > 0:
                        percent_accuracy = accuracy_score(percent_actual, percent_predicted)
            
            # 生成分类报告
            report = classification_report(actual, predicted, output_dict=True)
            
            return {
                'overall_accuracy': accuracy,
                'rank_method_accuracy': rank_accuracy,
                'percent_method_accuracy': percent_accuracy,
                'classification_report': report
            }
        except Exception as e:
            print(f"计算准确性错误: {e}")
            return {'overall_accuracy': 0, 'rank_method_accuracy': np.nan, 
                    'percent_method_accuracy': np.nan, 'classification_report': {}}
    
    def calculate_fan_influence_metric(self):
        """计算粉丝投票影响力指标"""
        try:
            # 粉丝投票与淘汰结果的相关性
            fan_correlations = []
            judge_correlations = []
            
            for week, week_data in self.results.groupby('Week'):
                if len(week_data) < 2:
                    continue
                    
                try:
                    # 计算粉丝投票相关性
                    fan_corr = np.corrcoef(
                        week_data['Fan_Vote_Mean'].fillna(0).values,
                        (week_data['Result'] == 'Eliminated').astype(int).values
                    )[0, 1]
                    if not np.isnan(fan_corr):
                        fan_correlations.append(fan_corr)
                except Exception as e:
                    pass
                    
                try:
                    # 计算评委分数相关性
                    judge_corr = np.corrcoef(
                        week_data['Judge_Score'].fillna(0).values,
                        (week_data['Result'] == 'Eliminated').astype(int).values
                    )[0, 1]
                    if not np.isnan(judge_corr):
                        judge_correlations.append(judge_corr)
                except Exception as e:
                    pass
            
            fan_correlation = np.mean(fan_correlations) if fan_correlations else 0
            judge_correlation = np.mean(judge_correlations) if judge_correlations else 0
            
            # 计算影响力比例
            total_influence = abs(fan_correlation) + abs(judge_correlation)
            fan_influence_ratio = abs(fan_correlation) / total_influence if total_influence > 0 else 0.5
            
            return {
                'fan_vote_correlation': fan_correlation,
                'judge_score_correlation': judge_correlation,
                'fan_influence_ratio': fan_influence_ratio,
                'fan_correlation_count': len(fan_correlations),
                'judge_correlation_count': len(judge_correlations)
            }
        except Exception as e:
            print(f"计算影响力指标错误: {e}")
            return {
                'fan_vote_correlation': 0,
                'judge_score_correlation': 0,
                'fan_influence_ratio': 0.5,
                'fan_correlation_count': 0,
                'judge_correlation_count': 0
            }


class ControversyAnalyzer:
    def __init__(self, full_data):
        self.data = full_data
        
    def identify_controversial_cases(self, threshold=0.3):
        """识别评委与粉丝意见分歧的案例"""
        controversial = []
        
        try:
            for season in self.data['Season'].unique():
                if pd.isna(season):
                    continue
                    
                season_data = self.data[self.data['Season'] == season]
                
                for name in season_data['Name'].unique():
                    if pd.isna(name):
                        continue
                        
                    contestant_data = season_data[season_data['Name'] == name]
                    
                    if len(contestant_data) > 1:
                        # 确保有足够的数据计算
                        if contestant_data['Judge_Score'].isna().all() or contestant_data['Fan_Vote_Mean'].isna().all():
                            continue
                        
                        # 计算平均评委排名百分位
                        judge_percentile = contestant_data['Judge_Score'].rank(pct=True, method='average').mean()
                        
                        # 计算平均粉丝投票百分位
                        fan_percentile = contestant_data['Fan_Vote_Mean'].rank(pct=True, method='average').mean()
                        
                        # 分歧度 = |评委百分位 - 粉丝百分位|
                        divergence = abs(judge_percentile - fan_percentile)
                        
                        # 获取最终结果
                        final_result = contestant_data['Result'].iloc[-1] if 'Result' in contestant_data.columns else 'Unknown'
                        
                        if (divergence > threshold and 
                            final_result not in ['Withdrew', 'Eliminated Week 1'] and
                            not pd.isna(divergence)):
                            controversial.append({
                                'Name': name,
                                'Season': season,
                                'Divergence_Score': divergence,
                                'Judge_Percentile': judge_percentile,
                                'Fan_Percentile': fan_percentile,
                                'Final_Result': final_result
                            })
        
            if controversial:
                df = pd.DataFrame(controversial)
                return df.sort_values('Divergence_Score', ascending=False)
            else:
                print("未找到符合条件的争议案例")
                return pd.DataFrame(columns=['Name', 'Season', 'Divergence_Score', 
                                            'Judge_Percentile', 'Fan_Percentile', 'Final_Result'])
        except Exception as e:
            print(f"识别争议案例错误: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def analyze_specific_cases(self, case_names):
        """分析特定争议案例"""
        case_analyses = []
        
        try:
            for case in case_names:
                # 使用模糊匹配查找选手
                case_data = self.data[self.data['Name'].str.contains(case, case=False, na=False)]
                if not case_data.empty:
                    analysis = {
                        'Case': case,
                        'Seasons': list(case_data['Season'].unique()),
                        'Avg_Judge_Score': case_data['Judge_Score'].mean(),
                        'Avg_Fan_Vote': case_data['Fan_Vote_Mean'].mean(),
                        'Judge_Rank_Percentile': case_data['Judge_Score'].rank(pct=True, method='average').mean(),
                        'Fan_Rank_Percentile': case_data['Fan_Vote_Mean'].rank(pct=True, method='average').mean(),
                        'Survival_Weeks': len(case_data),
                        'Final_Result': case_data['Result'].iloc[-1] if 'Result' in case_data.columns else 'Unknown'
                    }
                    case_analyses.append(analysis)
            
            return pd.DataFrame(case_analyses)
        except Exception as e:
            print(f"分析特定案例错误: {e}")
            return pd.DataFrame()


class JudgeSaveSimulator:
    def __init__(self, weekly_data):
        self.weekly_data = weekly_data
        
    def simulate_with_judge_save(self, selection_criteria='judge_score'):
        """
        模拟增设评委拯救环节
        selection_criteria: 'judge_score' 或 'fan_vote' 或 'combined'
        """
        simulation_results = []
        
        try:
            for week_num in self.weekly_data['Week'].unique():
                week_data = self.weekly_data[self.weekly_data['Week'] == week_num].copy()
                
                if len(week_data) < 2:
                    continue
                
                # 确定投票方法
                method = 'rank'
                if 'Method' in week_data.columns:
                    method = week_data['Method'].iloc[0]
                
                # 使用原始方法确定最危险的两名选手
                if method == 'rank':
                    week_data = VotingSystemSimulator.rank_method_simulation(week_data)
                    if 'Total_Rank' in week_data.columns:
                        bottom_two = week_data.nlargest(2, 'Total_Rank')
                    else:
                        continue
                else:
                    week_data = VotingSystemSimulator.percent_method_simulation(week_data)
                    if 'Combined_Score' in week_data.columns:
                        bottom_two = week_data.nsmallest(2, 'Combined_Score')
                    else:
                        continue
                
                # 评委选择淘汰其中一名
                if len(bottom_two) >= 2:
                    if selection_criteria == 'judge_score':
                        eliminated = bottom_two.loc[bottom_two['Judge_Score'].idxmin()]
                    elif selection_criteria == 'fan_vote':
                        eliminated = bottom_two.loc[bottom_two['Fan_Vote_Mean'].idxmin()]
                    else:  # combined
                        if method == 'rank':
                            eliminated = bottom_two.loc[bottom_two['Total_Rank'].idxmax()]
                        else:
                            eliminated = bottom_two.loc[bottom_two['Combined_Score'].idxmin()]
                    
                    week_data['Simulated_Eliminated'] = week_data['Name'] == eliminated['Name']
                    simulation_results.append(week_data)
            
            if simulation_results:
                return pd.concat(simulation_results)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"模拟评委拯救错误: {e}")
            return pd.DataFrame()
    
    def compare_simulation_results(self, original_results, judge_save_results):
        """比较原始结果与评委拯救结果"""
        try:
            if original_results.empty or judge_save_results.empty:
                return {
                    'total_changes': 0,
                    'change_percentage': 0,
                    'changed_cases': pd.DataFrame()
                }
            
            # 确保列存在
            required_cols = ['Season', 'Week', 'Name', 'Result']
            for col in required_cols:
                if col not in original_results.columns:
                    if col == 'Result':
                        original_results[col] = 'Safe'  # 默认值
                    else:
                        original_results[col] = np.nan
            
            comparison = pd.merge(
                original_results[['Season', 'Week', 'Name', 'Result']].dropna(subset=['Season', 'Week', 'Name']),
                judge_save_results[['Season', 'Week', 'Name', 'Simulated_Eliminated']].dropna(subset=['Season', 'Week', 'Name']),
                on=['Season', 'Week', 'Name'],
                how='inner'
            )
            
            if comparison.empty:
                return {
                    'total_changes': 0,
                    'change_percentage': 0,
                    'changed_cases': pd.DataFrame()
                }
            
            # 计算变化
            comparison['Result_Status'] = comparison['Result'].apply(
                lambda x: 'Eliminated' if str(x) == 'Eliminated' else 'Safe'
            )
            comparison['Simulated_Status'] = comparison['Simulated_Eliminated'].apply(
                lambda x: 'Eliminated' if x else 'Safe'
            )
            
            changes = comparison[comparison['Result_Status'] != comparison['Simulated_Status']]
            
            return {
                'total_changes': len(changes),
                'change_percentage': len(changes) / len(comparison) * 100 if len(comparison) > 0 else 0,
                'changed_cases': changes
            }
        except Exception as e:
            print(f"比较模拟结果错误: {e}")
            return {
                'total_changes': 0,
                'change_percentage': 0,
                'changed_cases': pd.DataFrame()
            }


def complete_analysis(fan_votes_path, judge_scores_path):
    """
    执行完整分析流程
    """
    print("开始分析...")
    
    # 1. 初始化分析器
    try:
        analyzer = DWTSAnalyzer(fan_votes_path, judge_scores_path)
        analyzer.preprocess_data()
        
        if analyzer.combined_data is None or analyzer.combined_data.empty:
            print("合并数据为空，无法进行分析")
            return None
    except Exception as e:
        print(f"初始化分析器错误: {e}")
        return None
    
    # 2. 模拟投票系统
    simulation_results = []
    print("模拟投票系统...")
    
    try:
        # 检查是否有必要的列
        required_cols = ['Season', 'Week', 'Name', 'Fan_Vote_Mean', 'Judge_Score', 'Result']
        missing_cols = [col for col in required_cols if col not in analyzer.combined_data.columns]
        if missing_cols:
            print(f"缺少必要的列: {missing_cols}")
            return None
        
        # 按Season和Week分组
        grouped_data = analyzer.combined_data.groupby(['Season', 'Week'])
        total_groups = len(grouped_data)
        
        for i, ((season, week), week_data) in enumerate(grouped_data, 1):
            if i % 20 == 0:
                print(f"处理中: {i}/{total_groups} 组 (Season {season}, Week {week})")
            
            # 确定投票方法
            method = 'rank'  # 默认值
            if 'Method' in week_data.columns:
                method = week_data['Method'].iloc[0]
            
            # 根据方法进行模拟
            if method == 'rank':
                simulated_week = VotingSystemSimulator.rank_method_simulation(week_data)
            else:
                simulated_week = VotingSystemSimulator.percent_method_simulation(week_data)
            
            simulation_results.append(simulated_week)
        
        if simulation_results:
            all_simulations = pd.concat(simulation_results, ignore_index=True)
            print(f"模拟完成，共 {len(all_simulations)} 条记录")
        else:
            print("模拟未生成任何结果")
            return None
    except Exception as e:
        print(f"模拟投票系统错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 3. 评估准确性
    print("评估准确性...")
    evaluator = AccuracyEvaluator(all_simulations)
    accuracy_metrics = evaluator.calculate_accuracy()
    influence_metrics = evaluator.calculate_fan_influence_metric()
    
    # 4. 识别争议案例
    print("识别争议案例...")
    controversy_analyzer = ControversyAnalyzer(analyzer.combined_data)
    controversial_cases = controversy_analyzer.identify_controversial_cases()
    
    # 特定案例分析
    print("分析特定案例...")
    specific_cases = ['Jerry Rice', 'Billy Ray Cyrus', 'Bristol Palin', 'Bobby Bones']
    case_analyses = controversy_analyzer.analyze_specific_cases(specific_cases)
    
    # 5. 模拟评委拯救环节
    print("模拟评委拯救环节...")
    judge_save_simulator = JudgeSaveSimulator(analyzer.combined_data)
    judge_save_results = judge_save_simulator.simulate_with_judge_save()
    simulation_comparison = judge_save_simulator.compare_simulation_results(
        analyzer.combined_data, judge_save_results
    )
    
    # 6. 生成报告
    print("生成报告...")
    report = {
        'accuracy_summary': accuracy_metrics,
        'influence_summary': influence_metrics,
        'controversial_cases_count': len(controversial_cases) if not controversial_cases.empty else 0,
        'controversial_cases_sample': controversial_cases.head(10).to_dict('records') if not controversial_cases.empty else [],
        'specific_case_analyses': case_analyses.to_dict('records') if not case_analyses.empty else [],
        'judge_save_impact': simulation_comparison,
        'recommendations': generate_recommendations(
            accuracy_metrics, influence_metrics, simulation_comparison
        ),
        'data_summary': {
            'total_records': len(analyzer.combined_data),
            'unique_seasons': analyzer.combined_data['Season'].nunique(),
            'unique_contestants': analyzer.combined_data['Name'].nunique(),
            'elimination_count': (analyzer.combined_data['Result'] == 'Eliminated').sum()
        }
    }
    
    return report

def generate_recommendations(accuracy_metrics, influence_metrics, judge_save_impact):
    """生成建议报告"""
    recommendations = []
    
    # 基于准确性的建议
    overall_acc = accuracy_metrics.get('overall_accuracy', 0)
    if overall_acc > 0.8:
        recommendations.append(f"粉丝投票估计模型准确性较高（准确率{overall_acc:.1%}），可以用于预测淘汰结果。")
    elif overall_acc > 0.6:
        recommendations.append(f"粉丝投票估计模型准确性一般（准确率{overall_acc:.1%}），需要进一步优化。")
    else:
        recommendations.append(f"粉丝投票估计模型准确性较低（准确率{overall_acc:.1%}），建议重新评估模型或数据质量。")
    
    # 基于投票方法的建议
    rank_acc = accuracy_metrics.get('rank_method_accuracy', 0)
    percent_acc = accuracy_metrics.get('percent_method_accuracy', 0)
    
    if not np.isnan(rank_acc) and not np.isnan(percent_acc):
        if rank_acc > percent_acc + 0.05:
            recommendations.append(f"排名制下模型预测更准确（排名制{rank_acc:.1%} vs 百分比制{percent_acc:.1%}），建议未来赛季继续使用排名制。")
        elif percent_acc > rank_acc + 0.05:
            recommendations.append(f"百分比制下模型预测更准确（百分比制{percent_acc:.1%} vs 排名制{rank_acc:.1%}），建议未来赛季使用百分比制。")
        else:
            recommendations.append(f"两种方法预测准确性相近（排名制{rank_acc:.1%} vs 百分比制{percent_acc:.1%}），均可考虑使用。")
    
    # 基于粉丝影响力的建议
    fan_influence = influence_metrics.get('fan_influence_ratio', 0.5)
    if fan_influence > 0.6:
        recommendations.append(f"粉丝投票对结果影响较大（影响力比例{fan_influence:.0%}），节目更偏向大众娱乐性。")
    elif fan_influence < 0.4:
        recommendations.append(f"评委评分对结果影响较大（影响力比例{fan_influence:.0%}），节目更偏向专业评审。")
    else:
        recommendations.append(f"评委和粉丝投票影响相对平衡（影响力比例{fan_influence:.0%}）。")
    
    # 基于评委拯救环节的建议
    change_pct = judge_save_impact.get('change_percentage', 0)
    if change_pct > 20:
        recommendations.append(
            f"增设评委拯救环节会显著改变结果（改变{change_pct:.1f}%），"
            "这可以减少争议但可能大幅降低粉丝参与感，建议谨慎引入。"
        )
    elif change_pct > 10:
        recommendations.append(
            f"增设评委拯救环节会适度改变结果（改变{change_pct:.1f}%），"
            "可以考虑引入以平衡专业性和娱乐性。"
        )
    elif change_pct > 0:
        recommendations.append(
            f"增设评委拯救环节影响较小（改变{change_pct:.1f}%），"
            "可以考虑引入作为争议情况的补充机制。"
        )
    else:
        recommendations.append("评委拯救环节对结果影响很小，可以根据节目需求决定是否引入。")
    
    # 最终建议
    if fan_influence > 0.7:
        recommendations.append(
            "综合建议：鉴于粉丝投票影响力较大，建议未来赛季保持现有投票系统，"
            "可考虑在极端争议情况下引入临时评委干预机制。"
        )
    elif fan_influence < 0.3:
        recommendations.append(
            "综合建议：鉴于评委评分影响力较大，建议未来赛季增强粉丝投票权重，"
            "或引入评委拯救环节增加专业评审的最终决定权。"
        )
    else:
        recommendations.append(
            "综合建议：未来赛季可采用平衡系统，平时使用当前投票方法，"
            "在决赛或半决赛引入评委拯救环节处理争议情况，以兼顾专业性和娱乐性。"
        )
    
    return recommendations


# 主执行程序
if __name__ == "__main__":
    # 文件路径（需要根据实际文件调整）
    fan_votes_file = "full_estimated_fan_votes.xlsx"
    judge_scores_file = "2026_MCM_Problem_C_Data.xlsx"
    
    try:
        # 执行完整分析
        analysis_report = complete_analysis(fan_votes_file, judge_scores_file)
        
        if analysis_report is None:
            print("分析失败，请检查数据文件")
            exit(1)
        
        # 输出关键结果
        print("\n" + "=" * 70)
        print("DWTS 投票系统分析报告")
        print("=" * 70)
        
        print(f"\n1. 数据概览:")
        print(f"   总记录数: {analysis_report['data_summary']['total_records']}")
        print(f"   赛季数量: {analysis_report['data_summary']['unique_seasons']}")
        print(f"   选手数量: {analysis_report['data_summary']['unique_contestants']}")
        print(f"   淘汰次数: {analysis_report['data_summary']['elimination_count']}")
        
        print(f"\n2. 模型准确性:")
        print(f"   整体准确率: {analysis_report['accuracy_summary']['overall_accuracy']:.2%}")
        
        rank_acc = analysis_report['accuracy_summary']['rank_method_accuracy']
        if not np.isnan(rank_acc):
            print(f"   排名制准确率: {rank_acc:.2%}")
        
        percent_acc = analysis_report['accuracy_summary']['percent_method_accuracy']
        if not np.isnan(percent_acc):
            print(f"   百分比制准确率: {percent_acc:.2%}")
        
        print(f"\n3. 投票影响力分析:")
        print(f"   粉丝投票与淘汰相关性: {analysis_report['influence_summary']['fan_vote_correlation']:.3f}")
        print(f"   评委评分与淘汰相关性: {analysis_report['influence_summary']['judge_score_correlation']:.3f}")
        print(f"   粉丝影响力比例: {analysis_report['influence_summary']['fan_influence_ratio']:.2%}")
        
        print(f"\n4. 争议案例识别:")
        print(f"   共发现 {analysis_report['controversial_cases_count']} 个争议案例")
        if analysis_report['controversial_cases_sample']:
            print("   前5名争议案例:")
            for i, case in enumerate(analysis_report['controversial_cases_sample'][:5], 1):
                print(f"   {i}. {case['Name']} (第{case['Season']}季): "
                      f"分歧度={case['Divergence_Score']:.3f}, 最终结果: {case['Final_Result']}")
        
        print(f"\n5. 特定案例分析:")
        if analysis_report['specific_case_analyses']:
            for case in analysis_report['specific_case_analyses']:
                print(f"   {case['Case']}: 平均评委分={case['Avg_Judge_Score']:.1f}, "
                      f"平均粉丝投票={case['Avg_Fan_Vote']:.3f}, 生存周数={case['Survival_Weeks']}, "
                      f"最终结果: {case.get('Final_Result', 'Unknown')}")
        else:
            print("   未找到指定的特定案例")
        
        print(f"\n6. 评委拯救环节影响:")
        print(f"   结果改变比例: {analysis_report['judge_save_impact']['change_percentage']:.1f}%")
        print(f"   总改变次数: {analysis_report['judge_save_impact']['total_changes']}")
        
        print(f"\n7. 推荐建议:")
        for i, rec in enumerate(analysis_report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 70)
        
        # 保存详细报告到文件
        try:
            import json
            with open("dwts_analysis_report.json", "w", encoding="utf-8") as f:
                json.dump(analysis_report, f, indent=2, ensure_ascii=False, default=str)
            print("详细报告已保存到 dwts_analysis_report.json")
        except Exception as e:
            print(f"保存报告文件错误: {e}")
        
        # 保存CSV格式的详细数据
        try:
            # 重新加载分析器以获取原始数据
            analyzer = DWTSAnalyzer(fan_votes_file, judge_scores_file)
            analyzer.preprocess_data()
            analyzer.combined_data.to_csv("dwts_combined_data.csv", index=False, encoding='utf-8-sig')
            print("合并数据已保存到 dwts_combined_data.csv")
        except Exception as e:
            print(f"保存数据文件错误: {e}")
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        print("请确保数据文件路径正确")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()