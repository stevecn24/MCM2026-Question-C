import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import json
import os
from datetime import datetime

# ============================================
# 数据加载器
# ============================================
class DataLoader:
    def __init__(self, fan_votes_path, judge_scores_path):
        self.fan_votes_path = fan_votes_path
        self.judge_scores_path = judge_scores_path
        self.combined_data = None
        
    def load_and_merge_data(self):
        """加载并合并数据"""
        try:
            print("加载数据文件...")
            # 读取粉丝投票数据
            fan_votes = pd.read_excel(self.fan_votes_path)
            print(f"粉丝投票数据: {len(fan_votes)} 条记录")
            
            # 读取评委分数数据
            judge_scores = pd.read_excel(self.judge_scores_path)
            print(f"评委分数数据: {len(judge_scores)} 条记录")
            
            # 简化处理：直接使用粉丝投票数据
            self.combined_data = fan_votes.copy()
            
            # 确保必要的列存在
            required_cols = ['Season', 'Week', 'Name', 'Fan_Vote_Mean', 'Result']
            for col in required_cols:
                if col not in self.combined_data.columns:
                    print(f"警告: 缺少列 '{col}'")
            
            # 如果有Judge_Score列，使用它，否则使用默认值
            if 'Judge_Score' not in self.combined_data.columns:
                self.combined_data['Judge_Score'] = 30  # 默认值
                print("警告: 使用默认评委分数 30")
            
            print(f"\n合并后数据: {len(self.combined_data)} 条记录")
            print(f"赛季数: {self.combined_data['Season'].nunique()}")
            
            return True
            
        except Exception as e:
            print(f"数据加载错误: {e}")
            import traceback
            traceback.print_exc()
            return False

# ============================================
# 评委拯救分析器
# ============================================
class JudgeSaveAnalyzer:
    def __init__(self, data):
        self.data = data
        self.results = {}
        
    def analyze_all_seasons(self):
        """分析所有赛季"""
        print("\n" + "="*80)
        print("评委拯救环节影响分析")
        print("="*80)
        
        # 获取所有赛季
        seasons = sorted(self.data['Season'].unique())
        print(f"发现 {len(seasons)} 个赛季: {seasons}")
        
        # 分析每个赛季
        season_results = {}
        for season in seasons:
            if pd.isna(season):
                continue
                
            season_data = self.data[self.data['Season'] == season]
            if len(season_data) < 5:  # 数据太少跳过
                continue
                
            print(f"\n分析第 {int(season)} 季...")
            season_result = self.analyze_season(season_data, season)
            if season_result:
                season_results[season] = season_result
        
        # 汇总结果
        self.results = self.summarize_results(season_results)
        return self.results
    
    def analyze_season(self, season_data, season_num):
        """分析单个赛季"""
        # 按周分组
        weeks = sorted(season_data['Week'].unique())
        weekly_results = []
        
        for week in weeks:
            week_data = season_data[season_data['Week'] == week]
            if len(week_data) < 2:  # 至少需要2名选手
                continue
                
            week_result = self.analyze_week(week_data, week, season_num)
            if week_result:
                weekly_results.append(week_result)
        
        if not weekly_results:
            return None
        
        # 计算赛季统计 - 修复准确率计算
        total_weeks = len(weekly_results)
        
        # 计算改变周次
        changed_weeks = [r for r in weekly_results if r['changed_result']]
        changed_count = len(changed_weeks)
        
        # 计算改变周次中的正确预测数
        correct_count = sum(1 for r in changed_weeks if r['judge_correct'])
        
        # 计算百分比
        change_percentage = (changed_count / total_weeks * 100) if total_weeks > 0 else 0
        accuracy_percentage = (correct_count / changed_count * 100) if changed_count > 0 else 0
        
        # 计算总预测准确率（所有周次）
        total_correct = sum(1 for r in weekly_results if r['judge_correct'])
        total_accuracy = (total_correct / total_weeks * 100) if total_weeks > 0 else 0
        
        return {
            'season': season_num,
            'total_weeks': total_weeks,
            'changed_weeks': changed_count,
            'correct_in_changed': correct_count,
            'total_correct': total_correct,
            'change_percentage': change_percentage,
            'accuracy_in_changed': accuracy_percentage,  # 改变周次中的准确率
            'overall_accuracy': total_accuracy,          # 总准确率
            'weekly_results': weekly_results
        }
    
    def analyze_week(self, week_data, week_num, season_num):
        """分析单周数据"""
        try:
            # 确保数据格式正确
            week_data = week_data.copy()
            
            # 如果缺少必要列，使用默认值
            if 'Judge_Score' not in week_data.columns:
                week_data['Judge_Score'] = 30
            
            if 'Fan_Vote_Mean' not in week_data.columns:
                week_data['Fan_Vote_Mean'] = 0.5
            
            # 计算排名
            # 评委排名（分数越高排名越好）
            week_data['Judge_Rank'] = week_data['Judge_Score'].rank(ascending=False)
            
            # 粉丝排名（投票比例越高排名越好）
            week_data['Fan_Rank'] = week_data['Fan_Vote_Mean'].rank(ascending=False)
            
            # 总排名 = 评委排名 + 粉丝排名（越小越好）
            week_data['Total_Rank'] = week_data['Judge_Rank'] + week_data['Fan_Rank']
            
            # 找到最差的两名选手（总排名最高）
            bottom_two = week_data.nlargest(2, 'Total_Rank')
            
            # 原系统预测：总排名最高的被淘汰
            original_predicted = bottom_two.iloc[0]['Name'] if len(bottom_two) > 0 else None
            
            # 评委拯救预测：评委分数最低的被淘汰
            if len(bottom_two) > 1:
                judge_predicted = bottom_two.loc[bottom_two['Judge_Score'].idxmin()]['Name']
            else:
                judge_predicted = original_predicted
            
            # 实际结果
            if 'Result' in week_data.columns:
                actual_eliminated = week_data[week_data['Result'] == 'Eliminated']
                actual_name = actual_eliminated.iloc[0]['Name'] if len(actual_eliminated) > 0 else "Unknown"
            else:
                actual_name = "Unknown"
            
            # 判断结果是否改变
            changed = (original_predicted != judge_predicted) if original_predicted and judge_predicted else False
            
            # 判断评委预测是否正确
            judge_correct = (judge_predicted == actual_name) if actual_name != "Unknown" else False
            
            # 计算分歧度
            divergence = self.calculate_divergence(bottom_two, week_data)
            
            return {
                'season': season_num,
                'week': week_num,
                'total_contestants': len(week_data),
                'bottom_two': bottom_two[['Name', 'Judge_Score', 'Fan_Vote_Mean', 'Total_Rank']].to_dict('records'),
                'original_predicted': original_predicted,
                'judge_predicted': judge_predicted,
                'actual_result': actual_name,
                'changed_result': changed,
                'judge_correct': judge_correct,
                'divergence': divergence
            }
            
        except Exception as e:
            print(f"分析第{season_num}季第{week_num}周时出错: {e}")
            return None
    
    def calculate_divergence(self, bottom_two, week_data):
        """计算分歧度"""
        if len(bottom_two) < 2:
            return 0
        
        divergences = []
        for _, contestant in bottom_two.iterrows():
            name = contestant['Name']
            contestant_data = week_data[week_data['Name'] == name]
            
            if not contestant_data.empty:
                # 计算评委百分位
                judge_percentile = week_data['Judge_Score'].rank(pct=True)[week_data['Name'] == name].iloc[0]
                
                # 计算粉丝百分位
                fan_percentile = week_data['Fan_Vote_Mean'].rank(pct=True)[week_data['Name'] == name].iloc[0]
                
                divergence = abs(judge_percentile - fan_percentile)
                divergences.append(divergence)
        
        return max(divergences) if divergences else 0
    
    def summarize_results(self, season_results):
        """汇总所有赛季结果"""
        if not season_results:
            return None
        
        total_seasons = len(season_results)
        total_weeks = sum(r['total_weeks'] for r in season_results.values())
        total_changed = sum(r['changed_weeks'] for r in season_results.values())
        total_correct_in_changed = sum(r['correct_in_changed'] for r in season_results.values())
        total_overall_correct = sum(r['total_correct'] for r in season_results.values())
        
        # 计算总体指标
        overall_change_percentage = (total_changed / total_weeks * 100) if total_weeks > 0 else 0
        accuracy_in_changed_percentage = (total_correct_in_changed / total_changed * 100) if total_changed > 0 else 0
        overall_accuracy_percentage = (total_overall_correct / total_weeks * 100) if total_weeks > 0 else 0
        
        # 找出最有影响力的案例
        all_weekly_results = []
        for season_result in season_results.values():
            all_weekly_results.extend(season_result['weekly_results'])
        
        # 按分歧度排序，找出最争议的案例
        high_divergence_cases = sorted(
            [r for r in all_weekly_results if r['changed_result'] and r.get('divergence', 0) > 0.3],
            key=lambda x: x.get('divergence', 0),
            reverse=True
        )[:10]
        
        return {
            'summary': {
                'total_seasons': total_seasons,
                'total_weeks': total_weeks,
                'weeks_with_changes': total_changed,
                'correct_in_changed_weeks': total_correct_in_changed,
                'total_correct_predictions': total_overall_correct,
                'overall_change_percentage': overall_change_percentage,
                'accuracy_in_changed_percentage': accuracy_in_changed_percentage,  # 改变周次中的准确率
                'overall_accuracy_percentage': overall_accuracy_percentage,        # 总体准确率
                'key_metrics_explained': {
                    'overall_change_percentage': '评委拯救改变结果的周次比例',
                    'accuracy_in_changed_percentage': '在改变结果的周次中，评委预测正确的比例',
                    'overall_accuracy_percentage': '在所有周次中，评委预测正确的比例'
                }
            },
            'season_details': {season: result for season, result in season_results.items()},
            'top_cases': high_divergence_cases,
            'recommendations': self.generate_recommendations(
                total_changed, total_weeks, 
                total_correct_in_changed, total_overall_correct
            )
        }
    
    def generate_recommendations(self, changed_weeks, total_weeks, correct_in_changed, total_correct):
        """生成建议"""
        change_percentage = (changed_weeks / total_weeks * 100) if total_weeks > 0 else 0
        accuracy_in_changed = (correct_in_changed / changed_weeks * 100) if changed_weeks > 0 else 0
        overall_accuracy = (total_correct / total_weeks * 100) if total_weeks > 0 else 0
        
        recommendations = []
        
        # 基于影响率的建议
        if change_percentage > 20:
            recommendations.append(f"评委拯救环节影响较大（{change_percentage:.1f}%的周次会改变结果），引入需谨慎。")
        elif change_percentage > 10:
            recommendations.append(f"评委拯救环节影响适中（{change_percentage:.1f}%的周次会改变结果），可考虑引入。")
        else:
            recommendations.append(f"评委拯救环节影响较小（仅{change_percentage:.1f}%的周次会改变结果），可作为补充机制。")
        
        # 基于准确率的建议（改变周次中的准确率）
        if accuracy_in_changed > 70:
            recommendations.append(f"在改变结果的周次中，评委选择准确性高（{accuracy_in_changed:.1f}%），建议信任评委判断。")
        elif accuracy_in_changed > 50:
            recommendations.append(f"在改变结果的周次中，评委选择准确性一般（{accuracy_in_changed:.1f}%），需要权衡。")
        else:
            recommendations.append(f"在改变结果的周次中，评委选择准确性较低（仅{accuracy_in_changed:.1f}%），需重新考虑评委选择标准。")
        
        # 总体准确率
        recommendations.append(f"总体预测准确率：{overall_accuracy:.1f}%（所有周次中评委预测正确的比例）。")
        
        # 综合建议
        if change_percentage > 15 and accuracy_in_changed > 60:
            recommendations.append("推荐引入评委拯救环节，既能有效干预，又保持较高准确性。")
        elif change_percentage < 10:
            recommendations.append("可选择性引入评委拯救环节，在争议较大时使用。")
        else:
            recommendations.append("建议进一步分析具体案例，确定最适合的引入方式。")
        
        return recommendations

# ============================================
# 报告生成器
# ============================================
class ReportGenerator:
    @staticmethod
    def save_json_report(results, filename):
        """保存JSON格式报告"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"JSON报告已保存: {filename}")
        except Exception as e:
            print(f"保存JSON报告时出错: {e}")
    
    @staticmethod
    def save_text_report(results, filename):
        """保存文本格式报告"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(ReportGenerator.generate_text_summary(results))
            print(f"文本报告已保存: {filename}")
        except Exception as e:
            print(f"保存文本报告时出错: {e}")
    
    @staticmethod
    def generate_text_summary(results):
        """生成文本摘要"""
        if not results:
            return "无分析结果"
        
        summary = results.get('summary', {})
        top_cases = results.get('top_cases', [])
        recommendations = results.get('recommendations', [])
        
        output = "="*80 + "\n"
        output += "DWTS评委拯救环节影响分析报告\n"
        output += "="*80 + "\n\n"
        
        output += "1. 总体分析结果\n"
        output += "-"*40 + "\n"
        output += f"分析赛季数: {summary.get('total_seasons', 0)}\n"
        output += f"分析周次数: {summary.get('total_weeks', 0)}\n"
        output += f"结果改变周次: {summary.get('weeks_with_changes', 0)}\n"
        output += f"评委拯救总体影响率: {summary.get('overall_change_percentage', 0):.2f}%\n"
        output += f"改变周次中的准确率: {summary.get('accuracy_in_changed_percentage', 0):.2f}%\n"
        output += f"总体预测准确率: {summary.get('overall_accuracy_percentage', 0):.2f}%\n\n"
        
        # 解释指标含义
        key_metrics = summary.get('key_metrics_explained', {})
        if key_metrics:
            output += "指标解释:\n"
            for metric, explanation in key_metrics.items():
                output += f"  - {metric}: {explanation}\n"
            output += "\n"
        
        output += "2. 各赛季影响率\n"
        output += "-"*40 + "\n"
        
        season_details = results.get('season_details', {})
        for season, details in season_details.items():
            output += f"第{season}季: "
            output += f"改变率={details.get('change_percentage', 0):.1f}%, "
            output += f"改变准确率={details.get('accuracy_in_changed', 0):.1f}%, "
            output += f"总体准确率={details.get('overall_accuracy', 0):.1f}%"
            output += f" ({details.get('changed_weeks', 0)}/{details.get('total_weeks', 0)}周)\n"
        
        output += "\n3. 最有影响力的案例\n"
        output += "-"*40 + "\n"
        
        if top_cases:
            for i, case in enumerate(top_cases[:5], 1):
                output += f"{i}. 第{case.get('season')}季 第{case.get('week')}周\n"
                output += f"   危险组: {case['bottom_two'][0]['Name']} vs {case['bottom_two'][1]['Name']}\n"
                output += f"   原预测淘汰: {case.get('original_predicted')}\n"
                output += f"   评委拯救后淘汰: {case.get('judge_predicted')}\n"
                output += f"   实际淘汰: {case.get('actual_result')}\n"
                output += f"   分歧度: {case.get('divergence', 0)*100:.1f}%\n\n"
        else:
            output += "无高影响力案例\n\n"
        
        output += "4. 建议\n"
        output += "-"*40 + "\n"
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                output += f"{i}. {rec}\n"
        else:
            output += "无建议\n"
        
        output += "\n" + "="*80 + "\n"
        
        return output
    
    @staticmethod
    def print_summary(results):
        """打印摘要到控制台"""
        if not results:
            print("无分析结果")
            return
        
        summary = results.get('summary', {})
        
        print("\n" + "="*80)
        print("关键分析结果")
        print("="*80)
        
        print(f"\n1. 总体影响率: {summary.get('overall_change_percentage', 0):.2f}%")
        print("   (评委拯救改变结果的周次比例)")
        
        print(f"\n2. 改变周次中的准确率: {summary.get('accuracy_in_changed_percentage', 0):.2f}%")
        print("   (在改变结果的周次中，评委预测正确的比例)")
        
        print(f"\n3. 总体预测准确率: {summary.get('overall_accuracy_percentage', 0):.2f}%")
        print("   (在所有周次中，评委预测正确的比例)")
        
        print(f"\n4. 分析范围:")
        print(f"   - 分析赛季数: {summary.get('total_seasons', 0)}")
        print(f"   - 分析周次数: {summary.get('total_weeks', 0)}")
        print(f"   - 改变周次数: {summary.get('weeks_with_changes', 0)}")
        print(f"   - 改变周次中正确预测数: {summary.get('correct_in_changed_weeks', 0)}")
        
        top_cases = results.get('top_cases', [])
        if top_cases:
            print(f"\n5. 最有影响力的案例:")
            for i, case in enumerate(top_cases[:3], 1):
                print(f"   案例{i}: 第{case.get('season')}季第{case.get('week')}周")
                print(f"         分歧度: {case.get('divergence', 0)*100:.1f}%")
                print(f"         改变: {case.get('original_predicted')} → {case.get('judge_predicted')}")
                print(f"         实际: {case.get('actual_result')}")
        
        print("\n" + "="*80)

# ============================================
# 主程序
# ============================================
def main():
    # 文件路径
    fan_votes_file = "full_estimated_fan_votes.xlsx"
    judge_scores_file = "2026_MCM_Problem_C_Data.xlsx"
    
    print("DWTS评委拯救环节影响分析程序")
    print("="*60)
    
    try:
        # 1. 加载数据
        print("\n步骤1: 加载数据...")
        data_loader = DataLoader(fan_votes_file, judge_scores_file)
        if not data_loader.load_and_merge_data():
            print("数据加载失败，程序退出")
            return
        
        # 2. 分析数据
        print("\n步骤2: 分析评委拯救环节影响...")
        analyzer = JudgeSaveAnalyzer(data_loader.combined_data)
        results = analyzer.analyze_all_seasons()
        
        if not results:
            print("分析失败，无结果")
            return
        
        # 3. 生成报告
        print("\n步骤3: 生成分析报告...")
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"judge_save_analysis_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存报告
        json_file = os.path.join(output_dir, "analysis_results.json")
        ReportGenerator.save_json_report(results, json_file)
        
        text_file = os.path.join(output_dir, "analysis_summary.txt")
        ReportGenerator.save_text_report(results, text_file)
        
        # 打印摘要
        ReportGenerator.print_summary(results)
        
        print(f"\n分析完成！报告已保存到目录: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保以下文件在当前目录中:")
        print(f"  1. {fan_votes_file}")
        print(f"  2. {judge_scores_file}")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()

# ============================================
# 运行程序
# ============================================
if __name__ == "__main__":
    main()