import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import json
import os

# ============================================
# 基础分析类 - 需要放在最前面
# ============================================
class DWTSAnalyzer:
    def __init__(self, fan_votes_path, judge_scores_path):
        try:
            self.fan_votes = pd.read_excel(fan_votes_path)
            self.judge_scores = pd.read_excel(judge_scores_path)  # 修正了变量名错误
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
            self.fan_votes['Season'] = pd.to_numeric(self.fan_votes['Season'], errors='coerce').astype('Int64')
            self.judge_scores['season'] = pd.to_numeric(self.judge_scores['season'], errors='coerce').astype('Int64')
            
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
            
            # 确保所有数值列都是Python原生类型
            for col in self.combined_data.select_dtypes(include=[np.int64, np.float64]).columns:
                self.combined_data[col] = self.combined_data[col].astype('float')
            
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
    
    def get_season_week_data(self, season, week):
        """获取指定赛季和周次的数据"""
        if self.combined_data is None:
            return pd.DataFrame()
        
        mask = (self.combined_data['Season'] == season) & (self.combined_data['Week'] == week)
        return self.combined_data[mask].copy()


class ExtendedDWTSAnalyzer(DWTSAnalyzer):
    """扩展的DWTS分析器，添加额外功能"""
    def __init__(self, fan_votes_path, judge_scores_path):
        super().__init__(fan_votes_path, judge_scores_path)
    
    def get_weekly_summary(self):
        """获取每周的统计摘要"""
        if self.combined_data is None:
            return pd.DataFrame()
        
        weekly_summary = self.combined_data.groupby(['Season', 'Week']).agg({
            'Name': 'count',
            'Judge_Score': ['mean', 'std', 'min', 'max'],
            'Fan_Vote_Mean': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # 扁平化列名
        weekly_summary.columns = ['_'.join(col).strip() for col in weekly_summary.columns.values]
        weekly_summary = weekly_summary.reset_index()
        
        return weekly_summary


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


class EnhancedJudgeSaveSimulator:
    def __init__(self, weekly_data):
        self.weekly_data = weekly_data
        
    def simulate_with_judge_save_detailed(self, selection_criteria='combined'):
        """
        详细模拟评委拯救环节，返回完整分析结果
        """
        detailed_results = []
        weekly_simulations = []
        
        try:
            # 按周处理
            for week_num in sorted(self.weekly_data['Week'].unique()):
                week_data = self.weekly_data[self.weekly_data['Week'] == week_num].copy()
                
                if len(week_data) < 2:
                    continue
                
                # 确定投票方法
                method = 'rank'
                if 'Method' in week_data.columns:
                    method = week_data['Method'].iloc[0]
                
                # 获取Season信息
                season = week_data['Season'].iloc[0] if 'Season' in week_data.columns else 'Unknown'
                
                # 原始模拟（无评委拯救）
                if method == 'rank':
                    original_week = VotingSystemSimulator.rank_method_simulation(week_data)
                    if 'Total_Rank' in original_week.columns:
                        # 找到最危险的两名选手
                        bottom_two_original = original_week.nlargest(2, 'Total_Rank')
                    else:
                        continue
                else:
                    original_week = VotingSystemSimulator.percent_method_simulation(week_data)
                    if 'Combined_Score' in original_week.columns:
                        bottom_two_original = original_week.nsmallest(2, 'Combined_Score')
                    else:
                        continue
                
                # 确定原始模拟的淘汰者（排名最差）
                if method == 'rank':
                    original_eliminated = bottom_two_original.loc[bottom_two_original['Total_Rank'].idxmax()]
                else:
                    original_eliminated = bottom_two_original.loc[bottom_two_original['Combined_Score'].idxmin()]
                
                # 评委拯救环节：从bottom_two中选出一名淘汰
                if len(bottom_two_original) >= 2:
                    bottom_two_names = bottom_two_original['Name'].tolist()
                    
                    # 根据标准选择淘汰者
                    if selection_criteria == 'judge_score':
                        eliminated = bottom_two_original.loc[bottom_two_original['Judge_Score'].idxmin()]
                    elif selection_criteria == 'fan_vote':
                        eliminated = bottom_two_original.loc[bottom_two_original['Fan_Vote_Mean'].idxmin()]
                    else:  # combined
                        if method == 'rank':
                            eliminated = bottom_two_original.loc[bottom_two_original['Total_Rank'].idxmax()]
                        else:
                            eliminated = bottom_two_original.loc[bottom_two_original['Combined_Score'].idxmin()]
                    
                    # 记录真实结果
                    actual_eliminated_names = week_data[week_data['Result'] == 'Eliminated']['Name'].tolist()
                    
                    # 准备本周详细结果
                    week_result = {
                        'Season': int(season) if not pd.isna(season) and isinstance(season, (np.integer, int)) else str(season),
                        'Week': int(week_num) if not pd.isna(week_num) and isinstance(week_num, (np.integer, int)) else str(week_num),
                        'Method': method,
                        'Bottom_Two': bottom_two_names,
                        'Original_Predicted': original_eliminated['Name'],
                        'Judge_Save_Predicted': eliminated['Name'],
                        'Actual_Eliminated': actual_eliminated_names,
                        'Bottom_Two_Details': []
                    }
                    
                    # 添加bottom two的详细信息
                    for _, contestant in bottom_two_original.iterrows():
                        contestant_info = {
                            'Name': contestant['Name'],
                            'Judge_Score': float(contestant['Judge_Score']) if not pd.isna(contestant['Judge_Score']) else None,
                            'Fan_Vote_Mean': float(contestant['Fan_Vote_Mean']) if not pd.isna(contestant['Fan_Vote_Mean']) else None,
                            'Judge_Percentile': None,
                            'Fan_Percentile': None
                        }
                        
                        # 计算百分位（如果数据足够）
                        if 'Judge_Score' in week_data.columns and not week_data['Judge_Score'].isna().all():
                            judge_percentile = (week_data['Judge_Score'].rank(pct=True, method='min')
                                               [week_data['Name'] == contestant['Name']])
                            if not judge_percentile.empty:
                                contestant_info['Judge_Percentile'] = float(judge_percentile.iloc[0])
                        
                        if 'Fan_Vote_Mean' in week_data.columns and not week_data['Fan_Vote_Mean'].isna().all():
                            fan_percentile = (week_data['Fan_Vote_Mean'].rank(pct=True, method='min')
                                             [week_data['Name'] == contestant['Name']])
                            if not fan_percentile.empty:
                                contestant_info['Fan_Percentile'] = float(fan_percentile.iloc[0])
                        
                        week_result['Bottom_Two_Details'].append(contestant_info)
                    
                    # 确定结果是否改变
                    original_correct = original_eliminated['Name'] in actual_eliminated_names
                    judge_save_correct = eliminated['Name'] in actual_eliminated_names
                    
                    week_result['Original_Correct'] = original_correct
                    week_result['Judge_Save_Correct'] = judge_save_correct
                    week_result['Result_Changed'] = original_eliminated['Name'] != eliminated['Name']
                    
                    # 计算分歧度（如果两个选手都有百分位数据）
                    if (week_result['Bottom_Two_Details'][0]['Judge_Percentile'] is not None and 
                        week_result['Bottom_Two_Details'][0]['Fan_Percentile'] is not None and
                        week_result['Bottom_Two_Details'][1]['Judge_Percentile'] is not None and 
                        week_result['Bottom_Two_Details'][1]['Fan_Percentile'] is not None):
                        
                        # 分歧度计算：评委与粉丝对bottom two选手的相对偏好差异
                        contestant1_divergence = abs(
                            week_result['Bottom_Two_Details'][0]['Judge_Percentile'] - 
                            week_result['Bottom_Two_Details'][0]['Fan_Percentile']
                        )
                        contestant2_divergence = abs(
                            week_result['Bottom_Two_Details'][1]['Judge_Percentile'] - 
                            week_result['Bottom_Two_Details'][1]['Fan_Percentile']
                        )
                        
                        week_result['Divergence_Score'] = float(max(contestant1_divergence, contestant2_divergence))
                    else:
                        week_result['Divergence_Score'] = None
                    
                    detailed_results.append(week_result)
                    weekly_simulations.append(week_data)
            
            return {
                'weekly_simulations': pd.concat(weekly_simulations) if weekly_simulations else pd.DataFrame(),
                'detailed_results': detailed_results
            }
        except Exception as e:
            print(f"详细模拟评委拯救错误: {e}")
            import traceback
            traceback.print_exc()
            return {'weekly_simulations': pd.DataFrame(), 'detailed_results': []}
    
    def analyze_judge_save_impact(self, detailed_results, original_data):
        """
        分析评委拯救环节的详细影响
        """
        try:
            if not detailed_results:
                return {
                    'total_weeks': 0,
                    'changed_weeks': 0,
                    'change_percentage': 0,
                    'improved_accuracy': 0,
                    'worsened_accuracy': 0,
                    'unchanged_accuracy': 0,
                    'change_details': [],
                    'season_breakdown': {},
                    'method_breakdown': {}
                }
            
            total_weeks = len(detailed_results)
            changed_weeks = 0
            improved_weeks = 0
            worsened_weeks = 0
            unchanged_weeks = 0
            
            change_details = []
            season_stats = {}
            method_stats = {}
            
            for week_result in detailed_results:
                season = week_result['Season']
                method = week_result['Method']
                
                # 确保season是字符串类型（JSON兼容）
                season_key = str(season)
                
                # 初始化统计
                if season_key not in season_stats:
                    season_stats[season_key] = {'total': 0, 'changed': 0, 'improved': 0, 'worsened': 0}
                if method not in method_stats:
                    method_stats[method] = {'total': 0, 'changed': 0, 'improved': 0, 'worsened': 0}
                
                season_stats[season_key]['total'] += 1
                method_stats[method]['total'] += 1
                
                if week_result['Result_Changed']:
                    changed_weeks += 1
                    season_stats[season_key]['changed'] += 1
                    method_stats[method]['changed'] += 1
                    
                    # 检查准确率变化
                    if week_result['Judge_Save_Correct'] and not week_result['Original_Correct']:
                        improved_weeks += 1
                        season_stats[season_key]['improved'] += 1
                        method_stats[method]['improved'] += 1
                        accuracy_change = '改善'
                    elif not week_result['Judge_Save_Correct'] and week_result['Original_Correct']:
                        worsened_weeks += 1
                        season_stats[season_key]['worsened'] += 1
                        method_stats[method]['worsened'] += 1
                        accuracy_change = '恶化'
                    else:
                        unchanged_weeks += 1
                        accuracy_change = '持平'
                    
                    # 记录详细变化
                    change_detail = {
                        'Season': season,
                        'Week': week_result['Week'],
                        'Method': method,
                        'Bottom_Two': week_result['Bottom_Two'],
                        'Original_Predicted': week_result['Original_Predicted'],
                        'Judge_Save_Predicted': week_result['Judge_Save_Predicted'],
                        'Actual_Eliminated': week_result['Actual_Eliminated'],
                        'Accuracy_Change': accuracy_change,
                        'Divergence_Score': week_result.get('Divergence_Score', None),
                        'Judge_Preference': None,
                        'Fan_Preference': None
                    }
                    
                    # 确定评委和粉丝的偏好
                    bottom_two_details = week_result['Bottom_Two_Details']
                    if len(bottom_two_details) == 2:
                        # 评委偏好（分数更高的）
                        if (bottom_two_details[0]['Judge_Percentile'] is not None and 
                            bottom_two_details[1]['Judge_Percentile'] is not None):
                            if bottom_two_details[0]['Judge_Percentile'] > bottom_two_details[1]['Judge_Percentile']:
                                change_detail['Judge_Preference'] = bottom_two_details[0]['Name']
                            else:
                                change_detail['Judge_Preference'] = bottom_two_details[1]['Name']
                        
                        # 粉丝偏好（投票比例更高的）
                        if (bottom_two_details[0]['Fan_Percentile'] is not None and 
                            bottom_two_details[1]['Fan_Percentile'] is not None):
                            if bottom_two_details[0]['Fan_Percentile'] > bottom_two_details[1]['Fan_Percentile']:
                                change_detail['Fan_Preference'] = bottom_two_details[0]['Name']
                            else:
                                change_detail['Fan_Preference'] = bottom_two_details[1]['Name']
                    
                    change_details.append(change_detail)
                else:
                    unchanged_weeks += 1
            
            # 计算百分比
            change_percentage = float((changed_weeks / total_weeks * 100) if total_weeks > 0 else 0)
            improved_percentage = float((improved_weeks / changed_weeks * 100) if changed_weeks > 0 else 0)
            worsened_percentage = float((worsened_weeks / changed_weeks * 100) if changed_weeks > 0 else 0)
            
            # 添加百分比到分项统计
            for season_key in season_stats:
                if season_stats[season_key]['total'] > 0:
                    season_stats[season_key]['change_percentage'] = float(
                        season_stats[season_key]['changed'] / season_stats[season_key]['total'] * 100
                    )
            
            for method_key in method_stats:
                if method_stats[method_key]['total'] > 0:
                    method_stats[method_key]['change_percentage'] = float(
                        method_stats[method_key]['changed'] / method_stats[method_key]['total'] * 100
                    )
            
            return {
                'total_weeks': total_weeks,
                'changed_weeks': changed_weeks,
                'change_percentage': change_percentage,
                'improved_weeks': improved_weeks,
                'worsened_weeks': worsened_weeks,
                'unchanged_weeks': unchanged_weeks,
                'improved_percentage': improved_percentage,
                'worsened_percentage': worsened_percentage,
                'change_details': change_details,
                'season_breakdown': season_stats,
                'method_breakdown': method_stats
            }
        except Exception as e:
            print(f"分析评委拯救影响错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                'total_weeks': 0,
                'changed_weeks': 0,
                'change_percentage': 0,
                'improved_accuracy': 0,
                'worsened_accuracy': 0,
                'unchanged_accuracy': 0,
                'change_details': [],
                'season_breakdown': {},
                'method_breakdown': {}
            }
    
    def generate_judge_save_report(self, impact_analysis, selection_criteria='combined'):
        """
        生成评委拯救环节详细报告
        """
        # 确保所有数值都是JSON兼容的类型
        report = {
            'summary': {
                'selection_criteria': selection_criteria,
                'total_weeks_analyzed': int(impact_analysis['total_weeks']) if not pd.isna(impact_analysis['total_weeks']) else 0,
                'weeks_with_changes': int(impact_analysis['changed_weeks']) if not pd.isna(impact_analysis['changed_weeks']) else 0,
                'change_rate_percentage': float(impact_analysis['change_percentage']) if not pd.isna(impact_analysis['change_percentage']) else 0.0,
                'accuracy_improved_weeks': int(impact_analysis['improved_weeks']) if not pd.isna(impact_analysis['improved_weeks']) else 0,
                'accuracy_worsened_weeks': int(impact_analysis['worsened_weeks']) if not pd.isna(impact_analysis['worsened_weeks']) else 0,
                'accuracy_improved_percentage': float(impact_analysis['improved_percentage']) if not pd.isna(impact_analysis['improved_percentage']) else 0.0,
                'accuracy_worsened_percentage': float(impact_analysis['worsened_percentage']) if not pd.isna(impact_analysis['worsened_percentage']) else 0.0
            },
            'season_analysis': impact_analysis['season_breakdown'],
            'method_analysis': impact_analysis['method_breakdown'],
            'detailed_changes': impact_analysis['change_details'],
            'recommendations': []
        }
        
        # 生成建议
        change_rate = impact_analysis['change_percentage']
        improved_rate = impact_analysis['improved_percentage']
        
        if change_rate > 20:
            report['recommendations'].append(
                f"评委拯救环节将显著改变{change_rate:.1f}%的周次结果，可能大幅改变比赛走向。"
            )
        elif change_rate > 10:
            report['recommendations'].append(
                f"评委拯救环节将适度改变{change_rate:.1f}%的周次结果，可作为平衡机制引入。"
            )
        else:
            report['recommendations'].append(
                f"评委拯救环节仅改变{change_rate:.1f}%的周次结果，影响有限但可作为补充机制。"
            )
        
        if improved_rate > 60:
            report['recommendations'].append(
                f"在结果改变的周次中，{improved_rate:.1f}%的情况下评委拯救提高了预测准确性，表明评委判断更符合实际。"
            )
        elif improved_rate > 40:
            report['recommendations'].append(
                f"评委拯救在{improved_rate:.1f}%的改变情况下提高了准确性，效果中等。"
            )
        else:
            report['recommendations'].append(
                f"评委拯救仅在{improved_rate:.1f}%的改变情况下提高了准确性，可能需要调整选择标准。"
            )
        
        # 分析各赛季差异
        if impact_analysis['season_breakdown']:
            high_change_seasons = []
            low_change_seasons = []
            
            for season_key, stats in impact_analysis['season_breakdown'].items():
                change_pct = stats.get('change_percentage', 0)
                if change_pct > 20:
                    high_change_seasons.append(f"第{season_key}季({change_pct:.1f}%)")
                elif change_pct < 5:
                    low_change_seasons.append(f"第{season_key}季({change_pct:.1f}%)")
            
            if high_change_seasons:
                report['recommendations'].append(
                    f"以下赛季受影响较大: {', '.join(high_change_seasons)}，在这些赛季引入评委拯救可能带来较大变化。"
                )
            
            if low_change_seasons:
                report['recommendations'].append(
                    f"以下赛季受影响较小: {', '.join(low_change_seasons)}，在这些赛季引入评委拯救变化有限。"
                )
        
        return report


# ============================================
# 自定义JSON编码器
# ============================================
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理numpy数据类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif pd.isna(obj):
            return None
        # 让基类处理其他类型
        return super().default(obj)


# ============================================
# 主分析函数
# ============================================
def export_detailed_judge_save_analysis(analyzer, output_dir='judge_save_analysis'):
    """
    导出详细的评委拯救环节分析
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("评委拯救环节详细分析")
    print("=" * 80)
    
    # 使用增强的评委拯救模拟器
    enhanced_simulator = EnhancedJudgeSaveSimulator(analyzer.combined_data)
    
    # 测试不同的选择标准
    selection_criteria_list = ['combined', 'judge_score', 'fan_vote']
    
    all_results = {}
    
    for criteria in selection_criteria_list:
        print(f"\n模拟评委拯救环节 (选择标准: {criteria})")
        print("-" * 40)
        
        # 运行详细模拟
        simulation_result = enhanced_simulator.simulate_with_judge_save_detailed(
            selection_criteria=criteria
        )
        
        # 分析影响
        impact_analysis = enhanced_simulator.analyze_judge_save_impact(
            simulation_result['detailed_results'],
            analyzer.combined_data
        )
        
        # 生成报告
        report = enhanced_simulator.generate_judge_save_report(impact_analysis, criteria)
        
        all_results[criteria] = {
            'impact_analysis': impact_analysis,
            'report': report,
            'detailed_results': simulation_result['detailed_results']
        }
        
        # 输出摘要
        print(f"分析周次总数: {impact_analysis['total_weeks']}")
        print(f"结果改变周次: {impact_analysis['changed_weeks']} ({impact_analysis['change_percentage']:.1f}%)")
        print(f"准确率提高周次: {impact_analysis['improved_weeks']} ({impact_analysis['improved_percentage']:.1f}%)")
        print(f"准确率降低周次: {impact_analysis['worsened_weeks']} ({impact_analysis['worsened_percentage']:.1f}%)")
        
        # 保存详细结果到文件
        output_file = os.path.join(output_dir, f'judge_save_{criteria}_analysis.json')
        
        # 准备要保存的数据，确保所有数据都是JSON兼容的
        save_data = {
            'report': report,
            'impact_analysis': impact_analysis,
            'detailed_results': simulation_result['detailed_results'][:50]  # 只保存前50条详细记录
        }
        
        # 使用自定义编码器保存JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        print(f"详细结果已保存到: {output_file}")
    
    # 比较不同标准的效果
    print("\n" + "=" * 80)
    print("不同选择标准比较")
    print("=" * 80)
    
    comparison_table = []
    for criteria in selection_criteria_list:
        analysis = all_results[criteria]['impact_analysis']
        comparison_table.append({
            '选择标准': criteria,
            '改变比例': f"{analysis['change_percentage']:.1f}%",
            '准确率提高比例': f"{analysis['improved_percentage']:.1f}%",
            '准确率降低比例': f"{analysis['worsened_percentage']:.1f}%",
            '分析周次': analysis['total_weeks']
        })
    
    comparison_df = pd.DataFrame(comparison_table)
    print(comparison_df.to_string(index=False))
    
    # 找出改变最大的案例
    print("\n" + "=" * 80)
    print("评委拯救改变最大的10个案例")
    print("=" * 80)
    
    all_change_details = []
    for criteria in selection_criteria_list:
        for change in all_results[criteria]['impact_analysis']['change_details']:
            change['Selection_Criteria'] = criteria
            all_change_details.append(change)
    
    if all_change_details:
        # 按分歧度排序（如果有）
        sorted_changes = sorted(
            [c for c in all_change_details if c.get('Divergence_Score') is not None],
            key=lambda x: x.get('Divergence_Score', 0),
            reverse=True
        )[:10]
        
        for i, change in enumerate(sorted_changes, 1):
            print(f"\n{i}. 第{change['Season']}季 第{change['Week']}周")
            print(f"   投票方法: {change['Method']}")
            print(f"   危险组: {', '.join(change['Bottom_Two'])}")
            print(f"   原预测淘汰: {change['Original_Predicted']}")
            print(f"   评委拯救后淘汰: {change['Judge_Save_Predicted']}")
            print(f"   实际淘汰: {', '.join(change['Actual_Eliminated'])}")
            print(f"   准确率变化: {change['Accuracy_Change']}")
            if change.get('Divergence_Score'):
                print(f"   分歧度: {change['Divergence_Score']:.3f}")
            if change.get('Judge_Preference'):
                print(f"   评委偏好: {change['Judge_Preference']}")
            if change.get('Fan_Preference'):
                print(f"   粉丝偏好: {change['Fan_Preference']}")
            print(f"   选择标准: {change['Selection_Criteria']}")
    
    # 生成综合建议
    print("\n" + "=" * 80)
    print("综合建议")
    print("=" * 80)
    
    # 找出最佳选择标准
    best_criteria = None
    best_score = -1
    
    for criteria in selection_criteria_list:
        analysis = all_results[criteria]['impact_analysis']
        # 评分：改变比例适中(20-40%) + 准确率提高比例高
        change_score = min(max(analysis['change_percentage'], 0), 100)
        accuracy_score = analysis['improved_percentage'] - analysis['worsened_percentage']
        
        # 综合评分（改变比例30%左右最佳，准确率提高越多越好）
        ideal_change = 30
        change_penalty = abs(change_score - ideal_change) * 0.5
        total_score = accuracy_score - change_penalty
        
        if total_score > best_score:
            best_score = total_score
            best_criteria = criteria
    
    print(f"推荐的选择标准: {best_criteria}")
    print(f"理由:")
    
    best_analysis = all_results[best_criteria]['impact_analysis']
    print(f"  1. 改变比例适中 ({best_analysis['change_percentage']:.1f}%)，既不会完全颠覆比赛，又能提供有效干预")
    print(f"  2. 准确率提高比例 ({best_analysis['improved_percentage']:.1f}%) 高于降低比例 ({best_analysis['worsened_percentage']:.1f}%)")
    
    # 保存综合报告
    summary_report = {
        'comparison_of_criteria': comparison_table,
        'recommended_criteria': best_criteria,
        'best_criteria_analysis': all_results[best_criteria]['report'],
        'top_changed_cases': sorted_changes if 'sorted_changes' in locals() else []
    }
    
    summary_file = os.path.join(output_dir, 'judge_save_summary_report.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
    
    print(f"\n综合分析报告已保存到: {summary_file}")
    
    return all_results


def complete_analysis_with_judge_save_details(fan_votes_path, judge_scores_path):
    """
    执行包含详细评委拯救分析的完整分析流程
    """
    print("开始详细分析...")
    
    # 1. 初始化分析器
    try:
        analyzer = ExtendedDWTSAnalyzer(fan_votes_path, judge_scores_path)
        analyzer.preprocess_data()
        
        if analyzer.combined_data is None or analyzer.combined_data.empty:
            print("合并数据为空，无法进行分析")
            return None
        
        print(f"数据加载完成：共 {len(analyzer.combined_data)} 条记录")
        print(f"数据列：{analyzer.combined_data.columns.tolist()}")
    except Exception as e:
        print(f"初始化分析器错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 2. 执行详细评委拯救分析
    print("\n" + "=" * 80)
    print("执行评委拯救环节详细分析")
    print("=" * 80)
    
    judge_save_results = export_detailed_judge_save_analysis(analyzer)
    
    # 3. 数据分析摘要
    print("\n" + "=" * 80)
    print("数据摘要")
    print("=" * 80)
    
    data_summary = {
        'total_records': len(analyzer.combined_data),
        'unique_seasons': int(analyzer.combined_data['Season'].nunique()),
        'unique_contestants': int(analyzer.combined_data['Name'].nunique()),
        'unique_weeks': int(analyzer.combined_data['Week'].nunique()),
        'elimination_count': int((analyzer.combined_data['Result'] == 'Eliminated').sum()) if 'Result' in analyzer.combined_data.columns else 0,
        'rank_method_count': int((analyzer.combined_data['Method'] == 'rank').sum()) if 'Method' in analyzer.combined_data.columns else 0,
        'percent_method_count': int((analyzer.combined_data['Method'] == 'percent').sum()) if 'Method' in analyzer.combined_data.columns else 0
    }
    
    print(f"总记录数: {data_summary['total_records']}")
    print(f"赛季数量: {data_summary['unique_seasons']}")
    print(f"选手数量: {data_summary['unique_contestants']}")
    print(f"周次数量: {data_summary['unique_weeks']}")
    print(f"淘汰次数: {data_summary['elimination_count']}")
    print(f"排名制周次: {data_summary['rank_method_count']}")
    print(f"百分比制周次: {data_summary['percent_method_count']}")
    
    # 4. 保存数据到CSV
    try:
        analyzer.combined_data.to_csv("dwts_combined_data.csv", index=False, encoding='utf-8-sig')
        print("\n合并数据已保存到 dwts_combined_data.csv")
    except Exception as e:
        print(f"保存数据文件错误: {e}")
    
    # 5. 生成综合报告
    report = {
        'data_summary': data_summary,
        'judge_save_analysis': judge_save_results,
        'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return report


# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    # 文件路径
    fan_votes_file = "full_estimated_fan_votes.xlsx"
    judge_scores_file = "2026_MCM_Problem_C_Data.xlsx"
    
    print("DWTS投票系统分析程序")
    print("=" * 60)
    print(f"粉丝投票文件: {fan_votes_file}")
    print(f"评委分数文件: {judge_scores_file}")
    print("=" * 60)
    
    try:
        # 执行包含详细评委拯救的分析
        analysis_report = complete_analysis_with_judge_save_details(fan_votes_file, judge_scores_file)
        
        if analysis_report is None:
            print("分析失败，请检查数据文件")
            exit(1)
        
        # 保存完整报告
        with open("complete_dwts_analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        print("\n" + "=" * 60)
        print("完整分析完成！")
        print("详细报告已保存到:")
        print("  - complete_dwts_analysis_report.json (综合报告)")
        print("  - judge_save_analysis/ 目录下的多个文件 (评委拯救详细分析)")
        print("  - dwts_combined_data.csv (合并数据)")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        print("请确保数据文件路径正确")
        print("请将数据文件放在当前目录下，或修改文件路径")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()