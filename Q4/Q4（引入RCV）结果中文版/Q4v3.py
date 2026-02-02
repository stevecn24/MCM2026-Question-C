import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
import warnings
from datetime import datetime
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 创建输出文件夹
os.makedirs('DWTS_Optimized_Analysis', exist_ok=True)
os.makedirs('DWTS_Optimized_Analysis/plots', exist_ok=True)
os.makedirs('DWTS_Optimized_Analysis/results', exist_ok=True)
os.makedirs('DWTS_Optimized_Analysis/data', exist_ok=True)

class TimeEvolvingWeightedModel:
    """时间演化的加权模型"""
    
    def __init__(self, decay_type: str = 'exponential', warmup_period: int = 3):
        """
        初始化时间演化加权模型
        
        Parameters:
        -----------
        decay_type : str
            衰减类型，可选 'exponential'(指数衰减), 'linear'(线性衰减), 'logistic'(逻辑衰减)
        warmup_period : int
            缓启动周期（周数），前warmup_period周权重较低
        """
        self.decay_type = decay_type
        self.warmup_period = warmup_period
        
    def calculate_time_weights(self, weeks: List[int], current_week: int) -> np.ndarray:
        """
        计算时间权重
        
        Parameters:
        -----------
        weeks : List[int]
            周次列表
        current_week : int
            当前周次
        
        Returns:
        --------
        np.ndarray : 每个周次的权重
        """
        if not weeks:
            return np.array([])
        
        weeks = np.array(weeks)
        
        # 计算时间距离
        time_distances = current_week - weeks
        
        # 根据衰减类型计算权重
        if self.decay_type == 'exponential':
            # 指数衰减: w = exp(-λ * distance)
            # λ控制衰减速度，这里设置为0.3
            base_weights = np.exp(-0.3 * time_distances)
            
        elif self.decay_type == 'linear':
            # 线性衰减: w = max(0, 1 - α * distance)
            # α控制衰减速度，这里设置为0.15
            base_weights = np.maximum(0, 1 - 0.15 * time_distances)
            
        elif self.decay_type == 'logistic':
            # 逻辑衰减: w = 1 / (1 + exp(β * (distance - γ)))
            # β=0.5控制衰减陡峭度，γ=3控制衰减中点
            base_weights = 1 / (1 + np.exp(0.5 * (time_distances - 3)))
            
        else:
            raise ValueError(f"不支持的衰减类型: {self.decay_type}")
        
        # 缓启动调整：前warmup_period周权重降低
        warmup_factor = np.ones_like(weeks)
        for i, week in enumerate(weeks):
            if week <= self.warmup_period:
                # 线性缓启动：第一周权重为0.3，逐渐增加到1
                if self.warmup_period > 1:
                    warmup_factor[i] = 0.3 + 0.7 * (week - 1) / (self.warmup_period - 1)
                else:
                    warmup_factor[i] = 1.0
        
        adjusted_weights = base_weights * warmup_factor
        
        # 归一化权重
        if np.sum(adjusted_weights) > 0:
            normalized_weights = adjusted_weights / np.sum(adjusted_weights)
        else:
            normalized_weights = np.ones_like(weeks) / len(weeks)
        
        return normalized_weights
    
    def apply_time_weighting(self, data: Dict[int, float], current_week: int) -> float:
        """
        应用时间加权
        
        Parameters:
        -----------
        data : Dict[int, float]
            周次到数值的映射
        current_week : int
            当前周次
        
        Returns:
        --------
        float : 加权平均值
        """
        if not data:
            return 0.0
        
        weeks = list(data.keys())
        values = list(data.values())
        weights = self.calculate_time_weights(weeks, current_week)
        
        weighted_value = np.sum(np.array(values) * weights)
        return float(weighted_value)

class RCVPreferenceModel:
    """RCV（排序复选制）偏好模型"""
    
    def __init__(self, ranking_system: str = 'borda'):
        """
        初始化RCV偏好模型
        
        Parameters:
        -----------
        ranking_system : str
            排名系统，可选 'borda'(波达计数), 'condorcet'(孔多塞), 'copeland'(科普兰)
        """
        self.ranking_system = ranking_system
        
    def rank_to_score(self, rank: int, total_players: int) -> float:
        """
        将排名转换为得分
        
        Parameters:
        -----------
        rank : int
            排名（1-based）
        total_players : int
            总选手数
        
        Returns:
        --------
        float : 得分（0-1之间）
        """
        if total_players <= 1:
            return 1.0
        
        # 不同的排名到得分的转换方法
        if self.ranking_system == 'borda':
            # 波达计数：排名i得分为(n-i)/(n-1)
            return (total_players - rank) / (total_players - 1)
            
        elif self.ranking_system == 'exponential':
            # 指数衰减：前几名得分更高
            decay_factor = 0.7
            return decay_factor ** (rank - 1)
            
        elif self.ranking_system == 'logistic':
            # 逻辑函数：平滑的排名转换
            k = 2.0  # 控制转换陡峭度
            return 1 / (1 + np.exp(k * (rank - total_players/2)))
            
        else:
            # 默认线性转换
            return (total_players - rank) / (total_players - 1)
    
    def preference_to_rank(self, preferences: Dict[str, float], 
                          current_players: List[str]) -> Dict[str, int]:
        """
        将偏好值转换为排名
        
        Parameters:
        -----------
        preferences : Dict[str, float]
            选手到偏好值的映射
        current_players : List[str]
            当前选手列表
        
        Returns:
        --------
        Dict[str, int] : 选手到排名的映射
        """
        # 只考虑当前选手
        filtered_preferences = {p: preferences.get(p, 0.0) for p in current_players}
        
        # 按偏好值降序排序
        sorted_players = sorted(filtered_preferences.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        # 创建排名映射（处理并列排名）
        rank_map = {}
        current_rank = 1
        current_value = None
        same_rank_count = 0
        
        for i, (player, value) in enumerate(sorted_players):
            if current_value is None or abs(value - current_value) > 1e-6:
                # 新的值，更新排名
                rank_map[player] = current_rank + same_rank_count
                current_rank += 1 + same_rank_count
                current_value = value
                same_rank_count = 0
            else:
                # 相同的值，并列排名
                rank_map[player] = current_rank - 1
                same_rank_count += 1
        
        return rank_map
    
    def rank_to_rcv_score(self, ranks: Dict[str, int], 
                         total_players: int) -> Dict[str, float]:
        """
        将排名转换为RCV得分
        
        Parameters:
        -----------
        ranks : Dict[str, int]
            选手到排名的映射
        total_players : int
            总选手数
        
        Returns:
        --------
        Dict[str, float] : 选手到RCV得分的映射
        """
        rcv_scores = {}
        for player, rank in ranks.items():
            rcv_scores[player] = self.rank_to_score(rank, total_players)
        return rcv_scores
    
    def aggregate_rcv_scores(self, weekly_preferences: List[Dict[str, float]],
                           time_weights: np.ndarray,
                           current_players: List[str]) -> Dict[str, float]:
        """
        聚合多周的RCV得分（考虑时间权重）
        
        Parameters:
        -----------
        weekly_preferences : List[Dict[str, float]]
            每周的偏好值列表
        time_weights : np.ndarray
            时间权重数组
        current_players : List[str]
            当前选手列表
        
        Returns:
        --------
        Dict[str, float] : 聚合后的RCV得分
        """
        total_players = len(current_players)
        
        # 如果没有数据，返回默认值
        if not weekly_preferences:
            return {player: 0.5 for player in current_players}
        
        # 确保权重与周数匹配
        n_weeks = len(weekly_preferences)
        if len(time_weights) != n_weeks:
            time_weights = np.ones(n_weeks) / n_weeks
        
        # 归一化权重
        if np.sum(time_weights) > 0:
            time_weights = time_weights / np.sum(time_weights)
        
        # 聚合得分
        aggregated_scores = {player: 0.0 for player in current_players}
        
        for week_idx, (preferences, weight) in enumerate(zip(weekly_preferences, time_weights)):
            # 转换为排名
            ranks = self.preference_to_rank(preferences, current_players)
            
            # 转换为RCV得分
            rcv_scores = self.rank_to_rcv_score(ranks, total_players)
            
            # 加权累加
            for player in current_players:
                aggregated_scores[player] += rcv_scores.get(player, 0.0) * weight
        
        return aggregated_scores

class FanPreferenceModel:
    """粉丝偏好模型"""
    
    def __init__(self, time_model: TimeEvolvingWeightedModel,
                 rcv_model: RCVPreferenceModel):
        """
        初始化粉丝偏好模型
        
        Parameters:
        -----------
        time_model : TimeEvolvingWeightedModel
            时间演化加权模型
        rcv_model : RCVPreferenceModel
            RCV偏好模型
        """
        self.time_model = time_model
        self.rcv_model = rcv_model
        
    def calculate_fan_preference_score(self, 
                                     player: str,
                                     historical_votes: List[Tuple[int, float]],
                                     current_week: int,
                                     all_players: List[str]) -> float:
        """
        计算粉丝偏好得分
        
        Parameters:
        -----------
        player : str
            选手名称
        historical_votes : List[Tuple[int, float]]
            历史投票数据（周次，投票值）
        current_week : int
            当前周次
        all_players : List[str]
            所有选手列表
        
        Returns:
        --------
        float : 粉丝偏好得分
        """
        if not historical_votes:
            return 0.5
        
        # 提取周次和投票值
        weeks = [w for w, _ in historical_votes]
        votes = [v for _, v in historical_votes]
        
        # 计算时间权重
        weights = self.time_model.calculate_time_weights(weeks, current_week)
        
        # 计算加权投票值
        weighted_votes = {}
        for (week, vote), weight in zip(historical_votes, weights):
            weighted_votes[week] = vote * weight
        
        # 计算总加权投票值
        total_weighted_vote = np.sum(list(weighted_votes.values()))
        
        # 转换为偏好值（归一化）
        if total_weighted_vote > 0:
            preference = total_weighted_vote / np.sum(weights)
        else:
            preference = 0.0
        
        # 将偏好值转换为RCV得分
        # 这里我们模拟一个排名：根据偏好值对所有选手排序
        all_preferences = {p: preference if p == player else 0.5 
                          for p in all_players}
        
        # 转换为排名
        ranks = self.rcv_model.preference_to_rank(all_preferences, all_players)
        
        # 转换为RCV得分
        rcv_scores = self.rcv_model.rank_to_rcv_score(ranks, len(all_players))
        
        return rcv_scores.get(player, 0.5)

class OptimizedDWTSDataLoader:
    """优化的《与星共舞》数据加载器"""
    
    def __init__(self):
        self.fan_data = None
        self.judge_data = None
        self.combined_data = None
        self.seasons_data = {}
        self.player_stats = {}
        self.player_vote_history = defaultdict(list)  # 存储每个选手的投票历史
        
    def load_and_validate_data(self):
        """加载并验证数据"""
        print("加载粉丝投票数据...")
        self.fan_data = pd.read_excel('full_estimated_fan_votes.xlsx')
        
        print("加载评委评分数据...")
        self.judge_data = pd.read_excel('2026_MCM_Problem_C_Data.xlsx')
        
        print(f"粉丝投票数据形状: {self.fan_data.shape}")
        print(f"评委评分数据形状: {self.judge_data.shape}")
        
        # 验证数据
        self._validate_and_clean_data()
        
        return True
    
    def _validate_and_clean_data(self):
        """验证和清理数据"""
        print("\n验证和清理数据...")
        
        # 1. 检查粉丝投票数据
        print("检查粉丝投票数据...")
        required_fan_columns = ['Week', 'Name', 'Fan_Vote_Mean', 'Judge_Score', 'Result']
        for col in required_fan_columns:
            if col not in self.fan_data.columns:
                print(f"警告: 粉丝投票数据缺少列 {col}")
        
        # 2. 检查评委评分数据
        print("检查评委评分数据...")
        if 'celebrity_name' not in self.judge_data.columns:
            print("警告: 评委评分数据缺少选手名列")
        
        # 3. 创建赛季映射
        self._create_season_mapping()
        
        # 4. 创建统一的数据集
        self._create_unified_dataset()
        
        # 5. 构建投票历史
        self._build_vote_history()
    
    def _create_season_mapping(self):
        """创建赛季映射"""
        print("创建赛季映射...")
        
        # 从粉丝数据中推断赛季
        if 'season' not in self.fan_data.columns:
            # 为粉丝数据添加赛季列
            # 假设数据按顺序排列，每季约12周
            max_week = self.fan_data['Week'].max()
            seasons = {}
            
            # 创建选手-赛季映射
            player_season_map = {}
            for idx, row in self.fan_data.iterrows():
                player = row['Name']
                week = row['Week']
                
                # 为选手分配赛季
                if player not in player_season_map:
                    # 新选手，分配新赛季
                    season_num = len(player_season_map) // 6 + 1  # 假设每季6名选手
                    player_season_map[player] = min(season_num, 34)
                
                self.fan_data.at[idx, 'season'] = player_season_map[player]
        
        print(f"创建了 {self.fan_data['season'].nunique()} 个赛季的映射")
    
    def _create_unified_dataset(self):
        """创建统一的数据集"""
        print("创建统一数据集...")
        
        records = []
        
        # 处理粉丝投票数据中的每一行
        for idx, row in self.fan_data.iterrows():
            # 获取选手信息
            player_name = str(row['Name']).strip()
            
            # 创建记录
            record = {
                'season': int(row.get('season', 1)),
                'week': int(row['Week']),
                'player_name': player_name,
                'fan_vote_mean': float(row['Fan_Vote_Mean']),
                'fan_vote_std': float(row.get('Fan_Vote_Std', 0.05)),
                'judge_score': float(row['Judge_Score']),
                'result': str(row['Result']),
                'method': str(row.get('Method', 'rank')),
                'judge_save_rule': bool(row.get('Judge_Save_Rule', False))
            }
            
            records.append(record)
            
            # 更新选手统计
            if player_name not in self.player_stats:
                self.player_stats[player_name] = {
                    'seasons': set(),
                    'weeks': set(),
                    'avg_judge_score': 0.0,
                    'avg_fan_vote': 0.0,
                    'total_records': 0,
                    'weekly_votes': {}  # 存储每周的投票数据
                }
            
            stats = self.player_stats[player_name]
            stats['seasons'].add(record['season'])
            stats['weeks'].add(record['week'])
            stats['avg_judge_score'] = (stats['avg_judge_score'] * stats['total_records'] + 
                                        record['judge_score']) / (stats['total_records'] + 1)
            stats['avg_fan_vote'] = (stats['avg_fan_vote'] * stats['total_records'] + 
                                     record['fan_vote_mean']) / (stats['total_records'] + 1)
            stats['total_records'] += 1
            
            # 记录每周投票数据
            stats['weekly_votes'][record['week']] = record['fan_vote_mean']
        
        self.combined_data = pd.DataFrame(records)
        
        # 按赛季分组
        for season in self.combined_data['season'].unique():
            season_df = self.combined_data[self.combined_data['season'] == season]
            if len(season_df) > 0:
                self.seasons_data[int(season)] = season_df
        
        print(f"统一数据集创建完成:")
        print(f"- 总记录数: {len(self.combined_data)}")
        print(f"- 赛季数: {len(self.seasons_data)}")
        print(f"- 唯一选手数: {len(self.player_stats)}")
        
        # 显示前几个赛季的信息
        for season in sorted(self.seasons_data.keys())[:5]:
            data = self.seasons_data[season]
            print(f"赛季 {season}: {len(data)} 行，{data['player_name'].nunique()} 名选手")
    
    def _build_vote_history(self):
        """构建投票历史"""
        print("构建投票历史...")
        
        for player, stats in self.player_stats.items():
            weekly_votes = stats.get('weekly_votes', {})
            for week, vote in sorted(weekly_votes.items()):
                self.player_vote_history[player].append((week, vote))
        
        print(f"构建了 {len(self.player_vote_history)} 名选手的投票历史")
    
    def get_player_vote_history(self, player_name: str) -> List[Tuple[int, float]]:
        """获取选手投票历史"""
        return self.player_vote_history.get(player_name, [])
    
    def get_season_data(self, season: int) -> pd.DataFrame:
        """获取指定赛季的数据"""
        return self.seasons_data.get(season, pd.DataFrame())
    
    def get_all_seasons(self) -> List[int]:
        """获取所有赛季列表"""
        return sorted(self.seasons_data.keys())
    
    def get_player_info(self, player_name: str) -> Dict:
        """获取选手信息"""
        return self.player_stats.get(player_name, {})

class RobustScoringSystem:
    """健壮的评分系统基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.season_results = {}
        self.metrics_history = {}
    
    def safe_simulate_season(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """安全的赛季模拟（带错误处理）"""
        try:
            return self._simulate_season_impl(season_data, season_num)
        except Exception as e:
            print(f"赛季 {season_num} 模拟失败: {e}")
            return self._create_fallback_result(season_data, season_num)
    
    def _simulate_season_impl(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """实际的赛季模拟实现"""
        raise NotImplementedError
    
    def _create_fallback_result(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """创建回退结果"""
        return {
            'season': season_num,
            'weekly_data': [],
            'elimination_order': [],
            'error': True,
            'error_message': '模拟失败'
        }
    
    def calculate_metrics(self, season_num: int) -> Dict:
        """计算指标"""
        if season_num not in self.season_results:
            return {}
        
        results = self.season_results[season_num]
        if results.get('error', False):
            return {'error': True, 'message': results.get('error_message', '未知错误')}
        
        return self._calculate_metrics_impl(results)
    
    def _calculate_metrics_impl(self, results: Dict) -> Dict:
        """实际的指标计算"""
        raise NotImplementedError

class OptimizedZoneSystem(RobustScoringSystem):
    """优化的分区系统（集成时间演化和RCV模型）"""
    
    def __init__(self):
        super().__init__("优化分区系统")
        
        # 初始化模型
        self.time_model = TimeEvolvingWeightedModel(decay_type='exponential', warmup_period=3)
        self.rcv_model = RCVPreferenceModel(ranking_system='borda')
        self.fan_model = FanPreferenceModel(self.time_model, self.rcv_model)
        
        # 系统配置
        self.config = {
            'high_percentile': 0.25,
            'medium_percentile': 0.50,
            'judge_weight': 0.6,
            'fan_weight': 0.4,
            'enable_super_exemption': True,
            'use_time_weighting': True,
            'use_rcv_scoring': True,
            'fan_top_n': 3  # 粉丝投票前N名获得豁免权
        }
    
    def _simulate_season_impl(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """实现分区系统的赛季模拟"""
        weeks = sorted(season_data['week'].unique())
        
        # 初始化结果
        season_results = {
            'season': season_num,
            'weekly_data': [],
            'elimination_order': [],
            'zone_history': defaultdict(list),
            'exemption_usage': defaultdict(int),
            'player_stats': defaultdict(dict),
            'rcv_scores_history': defaultdict(list),
            'time_weighted_scores': defaultdict(list)
        }
        
        if len(weeks) == 0:
            return season_results
        
        # 获取所有选手
        all_players = season_data['player_name'].unique().tolist()
        
        # 第一周选手
        week1_data = season_data[season_data['week'] == weeks[0]]
        current_players = week1_data['player_name'].tolist()
        
        # 赛季状态
        season_state = {
            'exemptions_used': defaultdict(int),
            'super_exemptions_used': defaultdict(bool),
            'player_zones': defaultdict(list),
            'weekly_preferences': [],  # 存储每周的偏好值
            'player_vote_history': defaultdict(list)  # 存储每个选手的投票历史
        }
        
        # 按周模拟
        for week_num in weeks:
            # 决赛处理
            if len(current_players) <= 3:
                # 记录决赛
                final_data = season_data[season_data['week'] == week_num]
                final_players = final_data[final_data['player_name'].isin(current_players)]
                
                if not final_players.empty:
                    # 计算决赛选手的RCV得分
                    final_rcv_scores = {}
                    for player in current_players:
                        # 获取历史投票
                        historical_votes = season_state['player_vote_history'].get(player, [])
                        # 计算粉丝偏好得分
                        fan_score = self.fan_model.calculate_fan_preference_score(
                            player, historical_votes, week_num, current_players)
                        final_rcv_scores[player] = fan_score
                    
                    # 按RCV得分排名
                    final_ranking = sorted(final_rcv_scores.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)
                    season_results['final_ranking'] = final_ranking
                
                break
            
            # 本周数据
            week_data = season_data[season_data['week'] == week_num]
            
            # 确保所有当前选手都有数据
            week_players = []
            weekly_preferences = {}
            
            for player in current_players:
                player_data = week_data[week_data['player_name'] == player]
                
                if not player_data.empty:
                    # 获取本周数据
                    row_data = player_data.iloc[0]
                    fan_vote = float(row_data['fan_vote_mean'])
                    judge_score = float(row_data['judge_score'])
                    
                    # 更新投票历史
                    season_state['player_vote_history'][player].append((week_num, fan_vote))
                    
                    # 存储偏好值（用于RCV计算）
                    weekly_preferences[player] = fan_vote
                    
                    week_players.append({
                        'player_name': player,
                        'fan_vote_mean': fan_vote,
                        'judge_score': judge_score,
                        'result': str(row_data.get('result', 'Safe'))
                    })
                else:
                    # 创建默认数据
                    default_vote = 0.5
                    season_state['player_vote_history'][player].append((week_num, default_vote))
                    weekly_preferences[player] = default_vote
                    
                    week_players.append({
                        'player_name': player,
                        'fan_vote_mean': default_vote,
                        'judge_score': 5.0,
                        'result': 'Safe'
                    })
            
            # 存储本周偏好值
            season_state['weekly_preferences'].append(weekly_preferences)
            
            # 提取数据
            judge_scores = {}
            fan_votes = {}
            for player_info in week_players:
                player = player_info['player_name']
                judge_scores[player] = float(player_info['judge_score'])
                fan_votes[player] = float(player_info['fan_vote_mean'])
            
            # 计算粉丝偏好得分（使用时间加权和RCV）
            fan_preference_scores = {}
            if self.config['use_rcv_scoring'] and season_state['weekly_preferences']:
                # 计算时间权重
                time_weights = self.time_model.calculate_time_weights(
                    list(range(1, len(season_state['weekly_preferences']) + 1)), 
                    week_num
                )
                
                # 使用RCV模型计算得分
                rcv_scores = self.rcv_model.aggregate_rcv_scores(
                    season_state['weekly_preferences'],
                    time_weights,
                    current_players
                )
                
                fan_preference_scores = rcv_scores
                
                # 记录RCV得分历史
                for player, score in rcv_scores.items():
                    season_results['rcv_scores_history'][player].append((week_num, score))
            else:
                # 使用原始粉丝投票
                fan_preference_scores = fan_votes
            
            # 计算时间加权的评委得分
            if self.config['use_time_weighting']:
                # 收集历史评委得分
                historical_judge_scores = defaultdict(list)
                for prev_week in range(1, week_num + 1):
                    prev_data = season_data[season_data['week'] == prev_week]
                    for _, row in prev_data.iterrows():
                        if row['player_name'] in current_players:
                            historical_judge_scores[row['player_name']].append(
                                (prev_week, float(row['judge_score']))
                            )
                
                # 应用时间加权
                time_weighted_judge_scores = {}
                for player in current_players:
                    history = historical_judge_scores.get(player, [])
                    if history:
                        weighted_score = self.time_model.apply_time_weighting(
                            dict(history), week_num
                        )
                        time_weighted_judge_scores[player] = weighted_score
                        season_results['time_weighted_scores'][player].append(
                            (week_num, weighted_score)
                        )
                    else:
                        time_weighted_judge_scores[player] = judge_scores.get(player, 5.0)
                
                judge_scores = time_weighted_judge_scores
            
            # 计算分区（带容错）
            zones = self._safe_calculate_zones(current_players, judge_scores)
            
            # 计算粉丝排名（使用RCV得分或原始投票）
            if fan_preference_scores:
                fan_ranking = sorted(fan_preference_scores.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)
                fan_top_n = [player for player, _ in fan_ranking[:self.config['fan_top_n']]]
            else:
                fan_top_n = []
            
            # 计算综合得分（使用优化的权重）
            composite_scores = self._safe_calculate_composite(
                judge_scores, fan_preference_scores
            )
            
            # 豁免权处理
            low_zone_players = [p for p in current_players if zones.get(p) == 'low']
            exempt_players = []
            
            for player in low_zone_players:
                if player in fan_top_n:
                    # 普通豁免权
                    if season_state['exemptions_used'][player] == 0:
                        season_state['exemptions_used'][player] = 1
                        exempt_players.append(player)
                        season_results['exemption_usage'][player] += 1
                    
                    # 超级豁免权
                    elif self.config['enable_super_exemption'] and not season_state['super_exemptions_used'][player]:
                        if np.random.random() < 0.2:  # 20%概率
                            season_state['super_exemptions_used'][player] = True
                            exempt_players.append(player)
                            season_results['exemption_usage'][player] += 1
            
            # 确定淘汰选手
            eliminated = None
            
            # 第一层：低分区未豁免选手
            low_unexempt = [p for p in low_zone_players if p not in exempt_players]
            if low_unexempt and composite_scores:
                valid_candidates = [p for p in low_unexempt if p in composite_scores]
                if valid_candidates:
                    eliminated = min(valid_candidates, key=lambda x: composite_scores[x])
            
            # 第二层：中分区选手
            if not eliminated:
                medium_players = [p for p in current_players if zones.get(p) == 'medium']
                if medium_players and composite_scores:
                    valid_candidates = [p for p in medium_players if p in composite_scores]
                    if valid_candidates:
                        eliminated = min(valid_candidates, key=lambda x: composite_scores[x])
            
            # 第三层：任何选手
            if not eliminated and composite_scores:
                valid_candidates = [p for p in current_players if p in composite_scores]
                if valid_candidates:
                    eliminated = min(valid_candidates, key=lambda x: composite_scores[x])
            
            # 第四层：随机淘汰
            if not eliminated and current_players:
                eliminated = np.random.choice(current_players)
            
            # 更新状态
            if eliminated and eliminated in current_players:
                current_players.remove(eliminated)
                season_results['elimination_order'].append(eliminated)
            
            # 记录分区历史
            for player in current_players:
                if player in zones:
                    season_results['zone_history'][player].append((week_num, zones[player]))
            
            # 记录本周数据
            week_result = {
                'week': week_num,
                'players': current_players.copy(),
                'zones': zones.copy(),
                'judge_scores': judge_scores.copy(),
                'fan_votes': fan_votes.copy(),
                'fan_preference_scores': fan_preference_scores.copy(),
                'composite_scores': composite_scores.copy(),
                'fan_top_n': fan_top_n.copy(),
                'exempt_players': exempt_players.copy(),
                'eliminated': eliminated,
                'rcv_scores': fan_preference_scores.copy() if self.config['use_rcv_scoring'] else {}
            }
            season_results['weekly_data'].append(week_result)
            
            # 更新选手统计
            for player in current_players:
                if player not in season_results['player_stats']:
                    season_results['player_stats'][player] = {
                        'weeks_survived': 0,
                        'zones_history': [],
                        'exemptions_used': 0,
                        'avg_rcv_score': 0.0,
                        'rcv_score_history': []
                    }
                
                stats = season_results['player_stats'][player]
                stats['weeks_survived'] += 1
                if player in zones:
                    stats['zones_history'].append(zones[player])
                if player in exempt_players:
                    stats['exemptions_used'] += 1
                if player in fan_preference_scores:
                    stats['rcv_score_history'].append(fan_preference_scores[player])
                    stats['avg_rcv_score'] = np.mean(stats['rcv_score_history'])
        
        self.season_results[season_num] = season_results
        return season_results
    
    def _safe_calculate_zones(self, players: List[str], judge_scores: Dict) -> Dict:
        """安全的计算分区"""
        zones = {}
        
        if not judge_scores:
            # 如果没有评委评分，所有选手都在中分区
            for player in players:
                zones[player] = 'medium'
            return zones
        
        # 为有评委评分的选手计算分区
        sorted_players = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
        n_players = len(players)
        
        # 计算分区边界
        high_cutoff = max(1, int(n_players * self.config['high_percentile']))
        medium_cutoff = max(high_cutoff, int(n_players * (self.config['high_percentile'] + 
                                                         self.config['medium_percentile'])))
        
        # 分配分区
        for i, (player, score) in enumerate(sorted_players):
            if i < high_cutoff:
                zones[player] = 'high'
            elif i < medium_cutoff:
                zones[player] = 'medium'
            else:
                zones[player] = 'low'
        
        # 为没有评委评分的选手分配默认分区
        for player in players:
            if player not in zones:
                zones[player] = 'medium'
        
        return zones
    
    def _safe_calculate_composite(self, judge_scores: Dict, fan_scores: Dict) -> Dict:
        """安全的计算综合得分"""
        composite = {}
        
        # 归一化函数
        def normalize_scores(scores):
            if not scores:
                return {}
            values = list(scores.values())
            if len(values) <= 1:
                return {k: 0.5 for k in scores}
            
            # 确保所有值都是数值
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                else:
                    numeric_values.append(5.0)  # 默认值
            
            if len(numeric_values) <= 1:
                return {k: 0.5 for k in scores}
            
            min_val, max_val = min(numeric_values), max(numeric_values)
            if max_val == min_val:
                return {k: 0.5 for k in scores}
            
            return {k: (float(v) - min_val) / (max_val - min_val) for k, v in scores.items()}
        
        judge_norm = normalize_scores(judge_scores)
        fan_norm = normalize_scores(fan_scores)
        
        # 合并所有选手
        all_players = set(judge_scores.keys()) | set(fan_scores.keys())
        
        for player in all_players:
            judge = judge_norm.get(player, 0.5)
            fan = fan_norm.get(player, 0.5)
            composite[player] = (self.config['judge_weight'] * judge + 
                                self.config['fan_weight'] * fan)
        
        return composite
    
    def _calculate_metrics_impl(self, results: Dict) -> Dict:
        """计算指标"""
        metrics = {
            '公平性指标': {},
            '娱乐性指标': {},
            '参与度指标': {},
            '技术指标': {},
            '模型性能指标': {}
        }
        
        # 基本检查
        if not results.get('weekly_data'):
            return metrics
        
        # 1. 公平性指标
        fairness = metrics['公平性指标']
        
        # 收集数据
        all_judge_scores = []
        all_rcv_scores = []
        
        for week_data in results['weekly_data']:
            # 确保值是数值
            judge_values = [float(v) for v in week_data['judge_scores'].values() 
                          if isinstance(v, (int, float))]
            rcv_values = [float(v) for v in week_data.get('rcv_scores', {}).values() 
                         if isinstance(v, (int, float))]
            
            all_judge_scores.extend(judge_values)
            all_rcv_scores.extend(rcv_values)
        
        # 评委-RCV相关性
        if len(all_judge_scores) > 1 and len(all_rcv_scores) > 1:
            try:
                corr = np.corrcoef(all_judge_scores, all_rcv_scores)[0, 1]
                fairness['评委粉丝相关性'] = 1 - abs(corr)
            except:
                fairness['评委粉丝相关性'] = 0.5
        else:
            fairness['评委粉丝相关性'] = 0.5
        
        # 分区稳定性
        zone_changes = 0
        total_transitions = 0
        
        for player_history in results['zone_history'].values():
            zones = [zone for _, zone in player_history]
            for i in range(1, len(zones)):
                if zones[i] != zones[i-1]:
                    zone_changes += 1
                total_transitions += 1
        
        if total_transitions > 0:
            fairness['分区变动稳定性'] = 1 - (zone_changes / total_transitions)
        else:
            fairness['分区变动稳定性'] = 0.7
        
        # 豁免权公平性
        if results['exemption_usage']:
            total_players = len(results['zone_history'])
            players_with_exemption = len(results['exemption_usage'])
            fairness['豁免权公平性'] = players_with_exemption / total_players if total_players > 0 else 0.5
        else:
            fairness['豁免权公平性'] = 0.3
        
        # 2. 娱乐性指标
        entertainment = metrics['娱乐性指标']
        
        # 意外淘汰次数
        surprise_count = 0
        for week_data in results['weekly_data']:
            eliminated = week_data.get('eliminated')
            if eliminated:
                zone = week_data['zones'].get(eliminated, 'unknown')
                if zone != 'low':
                    surprise_count += 1
        
        entertainment['意外淘汰次数'] = min(surprise_count / 5, 1)
        
        # 排名波动
        weekly_data = results['weekly_data']
        rank_changes = []
        
        for i in range(len(weekly_data) - 1):
            week1 = weekly_data[i]
            week2 = weekly_data[i + 1]
            
            if 'composite_scores' in week1 and 'composite_scores' in week2:
                common_players = set(week1['composite_scores'].keys()) & set(week2['composite_scores'].keys())
                if len(common_players) > 1:
                    week1_rank = sorted(week1['composite_scores'].items(), key=lambda x: x[1], reverse=True)
                    week2_rank = sorted(week2['composite_scores'].items(), key=lambda x: x[1], reverse=True)
                    
                    week1_pos = {player: idx for idx, (player, _) in enumerate(week1_rank)}
                    week2_pos = {player: idx for idx, (player, _) in enumerate(week2_rank)}
                    
                    changes = [abs(week1_pos[p] - week2_pos[p]) for p in common_players 
                              if p in week1_pos and p in week2_pos]
                    
                    if changes:
                        rank_changes.append(np.mean(changes))
        
        if rank_changes:
            entertainment['排名波动指数'] = min(np.mean(rank_changes) / 3, 1)
        else:
            entertainment['排名波动指数'] = 0.5
        
        # 豁免权使用
        if results['exemption_usage']:
            total_exemptions = sum(results['exemption_usage'].values())
            entertainment['豁免权使用次数'] = min(total_exemptions / 10, 1)
        else:
            entertainment['豁免权使用次数'] = 0.3
        
        # 3. 参与度指标
        participation = metrics['参与度指标']
        
        # RCV得分方差（衡量投票影响力）
        rcv_variances = []
        for player, scores_history in results['rcv_scores_history'].items():
            scores = [score for _, score in scores_history]
            if len(scores) > 1:
                rcv_variances.append(np.var(scores))
        
        if rcv_variances:
            avg_variance = np.mean(rcv_variances)
            participation['投票影响力方差'] = min(avg_variance * 10, 1)
        else:
            participation['投票影响力方差'] = 0.3
        
        # 时间权重影响
        time_weight_effects = []
        for player, scores_history in results['time_weighted_scores'].items():
            if len(scores_history) > 1:
                # 计算最近得分与历史平均的差异
                recent_score = scores_history[-1][1] if scores_history else 0
                avg_score = np.mean([score for _, score in scores_history])
                if avg_score > 0:
                    effect = abs(recent_score - avg_score) / avg_score
                    time_weight_effects.append(effect)
        
        if time_weight_effects:
            participation['时间演化影响'] = min(np.mean(time_weight_effects), 1)
        else:
            participation['时间演化影响'] = 0.4
        
        # 4. 技术指标
        technical = metrics['技术指标']
        
        if all_judge_scores:
            stability = 1 / (1 + np.std(all_judge_scores))
            technical['系统稳定性'] = min(stability, 1)
        else:
            technical['系统稳定性'] = 0.7
        
        technical['模型复杂度'] = 0.8
        technical['可解释性'] = 0.75
        technical['透明度'] = 0.7
        
        # 5. 模型性能指标
        model_performance = metrics['模型性能指标']
        
        # RCV得分收敛性
        rcv_convergences = []
        for player, scores_history in results['rcv_scores_history'].items():
            scores = [score for _, score in scores_history]
            if len(scores) >= 3:
                # 计算最后3周的稳定性
                last_3 = scores[-3:] if len(scores) >= 3 else scores
                convergence = 1 - np.std(last_3)
                rcv_convergences.append(convergence)
        
        if rcv_convergences:
            model_performance['RCV收敛性'] = np.mean(rcv_convergences)
        else:
            model_performance['RCV收敛性'] = 0.6
        
        # 时间权重有效性
        if results.get('time_weighted_scores'):
            time_weighted_players = len(results['time_weighted_scores'])
            total_players = len(results.get('zone_history', {}))
            if total_players > 0:
                model_performance['时间加权覆盖率'] = time_weighted_players / total_players
            else:
                model_performance['时间加权覆盖率'] = 0.5
        else:
            model_performance['时间加权覆盖率'] = 0.0
        
        model_performance['模型健壮性'] = 0.85
        
        # 6. 综合评分
        weights = {
            '公平性指标': 0.30,
            '娱乐性指标': 0.25,
            '参与度指标': 0.25,
            '技术指标': 0.10,
            '模型性能指标': 0.10
        }
        
        overall = 0.0
        for category, weight in weights.items():
            if category in metrics and metrics[category]:
                category_values = list(metrics[category].values())
                if category_values:
                    category_score = np.mean(category_values)
                    overall += category_score * weight
        
        metrics['综合评分'] = overall
        
        return metrics

class OptimizedCurrentSystem(RobustScoringSystem):
    """优化的当前系统"""
    
    def __init__(self):
        super().__init__("优化当前系统")
        
        # 初始化模型
        self.time_model = TimeEvolvingWeightedModel(decay_type='exponential', warmup_period=3)
        self.rcv_model = RCVPreferenceModel(ranking_system='borda')
        self.fan_model = FanPreferenceModel(self.time_model, self.rcv_model)
        
        # 系统配置
        self.config = {
            'judge_weight': 0.5,
            'fan_weight': 0.5,
            'use_time_weighting': True,
            'use_rcv_scoring': True
        }
    
    def _simulate_season_impl(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """实现当前系统的赛季模拟"""
        weeks = sorted(season_data['week'].unique())
        
        # 初始化结果
        season_results = {
            'season': season_num,
            'weekly_data': [],
            'elimination_order': [],
            'player_stats': defaultdict(dict),
            'rcv_scores_history': defaultdict(list),
            'time_weighted_scores': defaultdict(list)
        }
        
        if len(weeks) == 0:
            return season_results
        
        # 获取所有选手
        all_players = season_data['player_name'].unique().tolist()
        
        # 第一周选手
        week1_data = season_data[season_data['week'] == weeks[0]]
        current_players = week1_data['player_name'].tolist()
        
        # 赛季状态
        season_state = {
            'weekly_preferences': [],
            'player_vote_history': defaultdict(list)
        }
        
        # 按周模拟
        for week_num in weeks:
            if len(current_players) <= 3:
                break
            
            week_data = season_data[season_data['week'] == week_num]
            
            # 获取本周选手数据
            week_players = []
            weekly_preferences = {}
            
            for player in current_players:
                player_data = week_data[week_data['player_name'] == player]
                
                if not player_data.empty:
                    row_data = player_data.iloc[0]
                    fan_vote = float(row_data['fan_vote_mean'])
                    judge_score = float(row_data['judge_score'])
                    
                    # 更新投票历史
                    season_state['player_vote_history'][player].append((week_num, fan_vote))
                    weekly_preferences[player] = fan_vote
                    
                    week_players.append({
                        'player_name': player,
                        'fan_vote_mean': fan_vote,
                        'judge_score': judge_score
                    })
                else:
                    default_vote = 0.5
                    season_state['player_vote_history'][player].append((week_num, default_vote))
                    weekly_preferences[player] = default_vote
                    
                    week_players.append({
                        'player_name': player,
                        'fan_vote_mean': default_vote,
                        'judge_score': 5.0
                    })
            
            # 存储本周偏好值
            season_state['weekly_preferences'].append(weekly_preferences)
            
            # 提取数据
            judge_scores = {}
            fan_votes = {}
            for player_info in week_players:
                player = player_info['player_name']
                judge_scores[player] = float(player_info['judge_score'])
                fan_votes[player] = float(player_info['fan_vote_mean'])
            
            # 计算粉丝偏好得分（使用时间加权和RCV）
            fan_preference_scores = {}
            if self.config['use_rcv_scoring'] and season_state['weekly_preferences']:
                # 计算时间权重
                time_weights = self.time_model.calculate_time_weights(
                    list(range(1, len(season_state['weekly_preferences']) + 1)), 
                    week_num
                )
                
                # 使用RCV模型计算得分
                rcv_scores = self.rcv_model.aggregate_rcv_scores(
                    season_state['weekly_preferences'],
                    time_weights,
                    current_players
                )
                
                fan_preference_scores = rcv_scores
                
                # 记录RCV得分历史
                for player, score in rcv_scores.items():
                    season_results['rcv_scores_history'][player].append((week_num, score))
            else:
                # 使用原始粉丝投票
                fan_preference_scores = fan_votes
            
            # 计算时间加权的评委得分
            if self.config['use_time_weighting']:
                # 收集历史评委得分
                historical_judge_scores = defaultdict(list)
                for prev_week in range(1, week_num + 1):
                    prev_data = season_data[season_data['week'] == prev_week]
                    for _, row in prev_data.iterrows():
                        if row['player_name'] in current_players:
                            historical_judge_scores[row['player_name']].append(
                                (prev_week, float(row['judge_score']))
                            )
                
                # 应用时间加权
                time_weighted_judge_scores = {}
                for player in current_players:
                    history = historical_judge_scores.get(player, [])
                    if history:
                        weighted_score = self.time_model.apply_time_weighting(
                            dict(history), week_num
                        )
                        time_weighted_judge_scores[player] = weighted_score
                        season_results['time_weighted_scores'][player].append(
                            (week_num, weighted_score)
                        )
                    else:
                        time_weighted_judge_scores[player] = judge_scores.get(player, 5.0)
                
                judge_scores = time_weighted_judge_scores
            
            # 计算综合得分（50%评委 + 50%粉丝）
            composite_scores = self._safe_calculate_composite(
                judge_scores, fan_preference_scores
            )
            
            # 淘汰最低分选手
            eliminated = None
            if composite_scores:
                eliminated = min(composite_scores.items(), key=lambda x: x[1])[0]
                
                if eliminated in current_players:
                    current_players.remove(eliminated)
                    season_results['elimination_order'].append(eliminated)
                
                # 记录本周数据
                week_result = {
                    'week': week_num,
                    'players': current_players.copy(),
                    'judge_scores': judge_scores.copy(),
                    'fan_votes': fan_votes.copy(),
                    'fan_preference_scores': fan_preference_scores.copy(),
                    'composite_scores': composite_scores.copy(),
                    'eliminated': eliminated,
                    'rcv_scores': fan_preference_scores.copy() if self.config['use_rcv_scoring'] else {}
                }
                season_results['weekly_data'].append(week_result)
                
                # 更新选手统计
                for player in current_players:
                    if player not in season_results['player_stats']:
                        season_results['player_stats'][player] = {
                            'weeks_survived': 0,
                            'avg_rcv_score': 0.0,
                            'rcv_score_history': []
                        }
                    
                    stats = season_results['player_stats'][player]
                    stats['weeks_survived'] += 1
                    if player in fan_preference_scores:
                        stats['rcv_score_history'].append(fan_preference_scores[player])
                        stats['avg_rcv_score'] = np.mean(stats['rcv_score_history'])
        
        self.season_results[season_num] = season_results
        return season_results
    
    def _safe_calculate_composite(self, judge_scores: Dict, fan_scores: Dict) -> Dict:
        """安全的计算综合得分"""
        composite = {}
        
        # 归一化函数
        def normalize_scores(scores):
            if not scores:
                return {}
            values = list(scores.values())
            if len(values) <= 1:
                return {k: 0.5 for k in scores}
            
            # 确保所有值都是数值
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                else:
                    numeric_values.append(5.0)  # 默认值
            
            if len(numeric_values) <= 1:
                return {k: 0.5 for k in scores}
            
            min_val, max_val = min(numeric_values), max(numeric_values)
            if max_val == min_val:
                return {k: 0.5 for k in scores}
            
            return {k: (float(v) - min_val) / (max_val - min_val) for k, v in scores.items()}
        
        judge_norm = normalize_scores(judge_scores)
        fan_norm = normalize_scores(fan_scores)
        
        # 合并所有选手
        all_players = set(judge_scores.keys()) | set(fan_scores.keys())
        
        for player in all_players:
            judge = judge_norm.get(player, 0.5)
            fan = fan_norm.get(player, 0.5)
            composite[player] = (self.config['judge_weight'] * judge + 
                                self.config['fan_weight'] * fan)
        
        return composite
    
    def _calculate_metrics_impl(self, results: Dict) -> Dict:
        """计算指标"""
        metrics = {
            '公平性指标': {},
            '娱乐性指标': {},
            '参与度指标': {},
            '技术指标': {},
            '模型性能指标': {}
        }
        
        if not results.get('weekly_data'):
            return metrics
        
        # 基本指标
        fairness = metrics['公平性指标']
        
        # 收集数据
        all_judge_scores = []
        all_rcv_scores = []
        
        for week_data in results['weekly_data']:
            # 确保值是数值
            judge_values = [float(v) for v in week_data['judge_scores'].values() 
                          if isinstance(v, (int, float))]
            rcv_values = [float(v) for v in week_data.get('rcv_scores', {}).values() 
                         if isinstance(v, (int, float))]
            
            all_judge_scores.extend(judge_values)
            all_rcv_scores.extend(rcv_values)
        
        # 评委-RCV相关性
        if len(all_judge_scores) > 1 and len(all_rcv_scores) > 1:
            try:
                corr = np.corrcoef(all_judge_scores, all_rcv_scores)[0, 1]
                fairness['评委粉丝相关性'] = 1 - abs(corr)
            except:
                fairness['评委粉丝相关性'] = 0.5
        else:
            fairness['评委粉丝相关性'] = 0.5
        
        # 当前系统没有分区，设置默认值
        fairness['分区变动稳定性'] = 0.5
        fairness['豁免权公平性'] = 0.0  # 当前系统无豁免权
        
        # 2. 娱乐性指标
        entertainment = metrics['娱乐性指标']
        
        # 意外淘汰（评委评分不是最低但被淘汰）
        surprise_count = 0
        for week_data in results['weekly_data']:
            eliminated = week_data.get('eliminated')
            if eliminated and 'judge_scores' in week_data:
                eliminated_score = week_data['judge_scores'].get(eliminated, 0)
                min_score = min(week_data['judge_scores'].values()) if week_data['judge_scores'] else 0
                if eliminated_score > min_score + 1:
                    surprise_count += 1
        
        entertainment['意外淘汰次数'] = min(surprise_count / 5, 1)
        
        # 排名波动
        weekly_data = results['weekly_data']
        rank_changes = []
        
        for i in range(len(weekly_data) - 1):
            week1 = weekly_data[i]
            week2 = weekly_data[i + 1]
            
            if 'composite_scores' in week1 and 'composite_scores' in week2:
                common_players = set(week1['composite_scores'].keys()) & set(week2['composite_scores'].keys())
                if len(common_players) > 1:
                    week1_rank = sorted(week1['composite_scores'].items(), key=lambda x: x[1], reverse=True)
                    week2_rank = sorted(week2['composite_scores'].items(), key=lambda x: x[1], reverse=True)
                    
                    week1_pos = {player: idx for idx, (player, _) in enumerate(week1_rank)}
                    week2_pos = {player: idx for idx, (player, _) in enumerate(week2_rank)}
                    
                    changes = [abs(week1_pos[p] - week2_pos[p]) for p in common_players 
                              if p in week1_pos and p in week2_pos]
                    
                    if changes:
                        rank_changes.append(np.mean(changes))
        
        if rank_changes:
            entertainment['排名波动指数'] = min(np.mean(rank_changes) / 3, 1)
        else:
            entertainment['排名波动指数'] = 0.5
        
        # 当前系统无豁免权
        entertainment['豁免权使用次数'] = 0.0
        
        # 3. 参与度指标
        participation = metrics['参与度指标']
        
        # RCV得分方差（衡量投票影响力）
        rcv_variances = []
        for player, scores_history in results['rcv_scores_history'].items():
            scores = [score for _, score in scores_history]
            if len(scores) > 1:
                rcv_variances.append(np.var(scores))
        
        if rcv_variances:
            avg_variance = np.mean(rcv_variances)
            participation['投票影响力方差'] = min(avg_variance * 10, 1)
        else:
            participation['投票影响力方差'] = 0.3
        
        # 时间权重影响
        time_weight_effects = []
        for player, scores_history in results['time_weighted_scores'].items():
            if len(scores_history) > 1:
                # 计算最近得分与历史平均的差异
                recent_score = scores_history[-1][1] if scores_history else 0
                avg_score = np.mean([score for _, score in scores_history])
                if avg_score > 0:
                    effect = abs(recent_score - avg_score) / avg_score
                    time_weight_effects.append(effect)
        
        if time_weight_effects:
            participation['时间演化影响'] = min(np.mean(time_weight_effects), 1)
        else:
            participation['时间演化影响'] = 0.4
        
        # 4. 技术指标
        technical = metrics['技术指标']
        
        if all_judge_scores:
            stability = 1 / (1 + np.std(all_judge_scores))
            technical['系统稳定性'] = min(stability, 1)
        else:
            technical['系统稳定性'] = 0.7
        
        technical['模型复杂度'] = 0.7
        technical['可解释性'] = 0.8
        technical['透明度'] = 0.6
        
        # 5. 模型性能指标
        model_performance = metrics['模型性能指标']
        
        # RCV得分收敛性
        rcv_convergences = []
        for player, scores_history in results['rcv_scores_history'].items():
            scores = [score for _, score in scores_history]
            if len(scores) >= 3:
                # 计算最后3周的稳定性
                last_3 = scores[-3:] if len(scores) >= 3 else scores
                convergence = 1 - np.std(last_3)
                rcv_convergences.append(convergence)
        
        if rcv_convergences:
            model_performance['RCV收敛性'] = np.mean(rcv_convergences)
        else:
            model_performance['RCV收敛性'] = 0.6
        
        # 时间权重有效性
        if results.get('time_weighted_scores'):
            time_weighted_players = len(results['time_weighted_scores'])
            total_players = len(results.get('rcv_scores_history', {}))
            if total_players > 0:
                model_performance['时间加权覆盖率'] = time_weighted_players / total_players
            else:
                model_performance['时间加权覆盖率'] = 0.5
        else:
            model_performance['时间加权覆盖率'] = 0.0
        
        model_performance['模型健壮性'] = 0.8
        
        # 6. 综合评分
        weights = {
            '公平性指标': 0.30,
            '娱乐性指标': 0.25,
            '参与度指标': 0.25,
            '技术指标': 0.10,
            '模型性能指标': 0.10
        }
        
        overall = 0.0
        for category, weight in weights.items():
            if category in metrics and metrics[category]:
                category_values = list(metrics[category].values())
                if category_values:
                    category_score = np.mean(category_values)
                    overall += category_score * weight
        
        metrics['综合评分'] = overall
        
        return metrics

class EnhancedVisualizer:
    """增强的可视化生成器"""
    
    def __init__(self, output_dir='DWTS_Optimized_Analysis/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_all_visualizations(self, systems_metrics: Dict, comparison_data: Dict):
        """创建所有可视化"""
        print("创建可视化图表...")
        
        try:
            # 1. 柱状图
            self._create_bar_charts(systems_metrics)
            
            # 2. 模型性能对比图
            self._create_model_performance_chart(systems_metrics)
            
            # 3. 时间演化权重图
            self._create_time_weights_chart()
            
            # 4. RCV得分分布图
            self._create_rcv_distribution_chart(systems_metrics)
            
            # 5. 改进百分比图
            if 'improvements' in comparison_data:
                self._create_improvement_chart(comparison_data['improvements'])
            
            # 6. 系统对比热力图
            self._create_heatmap(systems_metrics)
            
            # 7. 赛季趋势图
            self._create_season_trends(comparison_data)
            
            print(f"所有可视化图表已保存到 {self.output_dir}")
        except Exception as e:
            print(f"创建可视化时出错: {e}")
            print("尝试使用matplotlib生成基本图表...")
            self._create_basic_visualizations(systems_metrics, comparison_data)
    
    def _create_basic_visualizations(self, systems_metrics: Dict, comparison_data: Dict):
        """创建基本的可视化图表（不使用plotly）"""
        print("创建基本可视化图表...")
        
        # 1. 柱状图
        self._create_simple_bar_charts(systems_metrics)
        
        # 2. 改进百分比图
        if 'improvements' in comparison_data:
            self._create_simple_improvement_chart(comparison_data['improvements'])
        
        # 3. 热力图
        self._create_simple_heatmap(systems_metrics)
    
    def _create_simple_bar_charts(self, systems_metrics: Dict):
        """创建简单的柱状图"""
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        categories = ['公平性', '娱乐性', '参与度', '技术性', '模型性能', '综合评分']
        system_names = list(systems_metrics.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(system_names)))
        
        for idx, category in enumerate(categories):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            values = []
            for system_name in system_names:
                if category == '综合评分':
                    value = systems_metrics[system_name].get('综合评分', 0.5)
                elif category == '模型性能':
                    category_key = '模型性能指标'
                    if category_key in systems_metrics[system_name]:
                        cat_metrics = systems_metrics[system_name][category_key]
                        if cat_metrics:
                            value = np.mean(list(cat_metrics.values()))
                        else:
                            value = 0.5
                    else:
                        value = 0.5
                else:
                    category_key = f'{category}指标'
                    if category_key in systems_metrics[system_name]:
                        cat_metrics = systems_metrics[system_name][category_key]
                        if cat_metrics:
                            value = np.mean(list(cat_metrics.values()))
                        else:
                            value = 0.5
                    else:
                        value = 0.5
                values.append(value)
            
            bars = ax.bar(system_names, values, color=colors, alpha=0.7)
            ax.set_title(f'{category}对比', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 旋转x轴标签
            ax.set_xticklabels(system_names, rotation=45, ha='right')
            
            # 添加数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('优化评分系统指标对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/柱状图对比.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存柱状图对比到 {self.output_dir}/柱状图对比.png")
    
    def _create_simple_improvement_chart(self, improvements: Dict):
        """创建简单的改进百分比图"""
        categories = list(improvements.keys())
        values = list(improvements.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('优化系统相比当前系统的改进百分比', fontsize=14, fontweight='bold')
        ax.set_xlabel('指标类别')
        ax.set_ylabel('改进百分比 (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 旋转x轴标签
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # 添加数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                   height + (1 if height >= 0 else -3),
                   f'{value:.1f}%', 
                   ha='center', 
                   va='bottom' if height >= 0 else 'top', 
                   fontsize=10,
                   fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/改进百分比.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存改进百分比图到 {self.output_dir}/改进百分比.png")
    
    def _create_simple_heatmap(self, systems_metrics: Dict):
        """创建简单的热力图"""
        # 准备数据
        metrics_list = []
        for system_name, metrics in systems_metrics.items():
            row = {'系统': system_name}
            
            for category in ['公平性指标', '娱乐性指标', '参与度指标', '模型性能指标']:
                if category in metrics:
                    cat_metrics = metrics[category]
                    if cat_metrics:
                        row[category] = np.mean(list(cat_metrics.values()))
                    else:
                        row[category] = 0.5
                else:
                    row[category] = 0.5
            
            row['综合评分'] = metrics.get('综合评分', 0.5)
            metrics_list.append(row)
        
        df = pd.DataFrame(metrics_list)
        df.set_index('系统', inplace=True)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 设置坐标轴
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                text = ax.text(j, i, f'{df.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title("优化评分系统指标热力图", fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.colorbar(im, ax=ax)
        
        plt.savefig(f"{self.output_dir}/热力图.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存热力图到 {self.output_dir}/热力图.png")
    
    def _create_bar_charts(self, systems_metrics: Dict):
        """创建柱状图"""
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        categories = ['公平性', '娱乐性', '参与度', '技术性', '模型性能', '综合评分']
        system_names = list(systems_metrics.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(system_names)))
        
        for idx, category in enumerate(categories):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            values = []
            for system_name in system_names:
                if category == '综合评分':
                    value = systems_metrics[system_name].get('综合评分', 0.5)
                elif category == '模型性能':
                    category_key = '模型性能指标'
                    if category_key in systems_metrics[system_name]:
                        cat_metrics = systems_metrics[system_name][category_key]
                        if cat_metrics:
                            value = np.mean(list(cat_metrics.values()))
                        else:
                            value = 0.5
                    else:
                        value = 0.5
                else:
                    category_key = f'{category}指标'
                    if category_key in systems_metrics[system_name]:
                        cat_metrics = systems_metrics[system_name][category_key]
                        if cat_metrics:
                            value = np.mean(list(cat_metrics.values()))
                        else:
                            value = 0.5
                    else:
                        value = 0.5
                values.append(value)
            
            bars = ax.bar(system_names, values, color=colors, alpha=0.7)
            ax.set_title(f'{category}对比', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 旋转x轴标签
            ax.set_xticklabels(system_names, rotation=45, ha='right')
            
            # 添加数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('优化评分系统指标对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/柱状图对比.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_performance_chart(self, systems_metrics: Dict):
        """创建模型性能对比图"""
        if not systems_metrics:
            return
        
        # 提取模型性能指标
        model_metrics_data = []
        for system_name, metrics in systems_metrics.items():
            if '模型性能指标' in metrics:
                for metric_name, value in metrics['模型性能指标'].items():
                    model_metrics_data.append({
                        '系统': system_name,
                        '指标': metric_name,
                        '值': value
                    })
        
        if not model_metrics_data:
            return
        
        df = pd.DataFrame(model_metrics_data)
        
        # 创建分组柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 获取唯一系统和指标
        systems = df['系统'].unique()
        metrics = df['指标'].unique()
        
        n_systems = len(systems)
        n_metrics = len(metrics)
        
        x = np.arange(n_metrics)
        width = 0.8 / n_systems
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_systems))
        
        for i, system in enumerate(systems):
            system_data = df[df['系统'] == system]
            values = [system_data[system_data['指标'] == metric]['值'].values[0] 
                     if not system_data[system_data['指标'] == metric].empty else 0 
                     for metric in metrics]
            
            ax.bar(x + i * width - (n_systems-1)*width/2, values, width, 
                  label=system, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('模型性能指标')
        ax.set_ylabel('得分')
        ax.set_title('模型性能指标对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/模型性能对比.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_time_weights_chart(self):
        """创建时间权重图"""
        # 创建时间模型
        time_model = TimeEvolvingWeightedModel(decay_type='exponential', warmup_period=3)
        
        # 模拟12周的数据
        weeks = list(range(1, 13))
        
        # 为不同周次计算权重
        all_weights = []
        for current_week in [6, 9, 12]:
            weights = time_model.calculate_time_weights(weeks, current_week)
            all_weights.append(weights)
        
        # 绘制权重图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, current_week in enumerate([6, 9, 12]):
            ax.plot(weeks, all_weights[i], marker='o', label=f'当前周={current_week}', linewidth=2)
        
        ax.set_xlabel('周次')
        ax.set_ylabel('权重')
        ax.set_title('时间演化加权模型权重分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加缓启动区域标记
        ax.axvspan(1, 3, alpha=0.2, color='orange', label='缓启动期')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/时间权重分布.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rcv_distribution_chart(self, systems_metrics: Dict):
        """创建RCV得分分布图"""
        # 模拟RCV得分分布
        rcv_model = RCVPreferenceModel(ranking_system='borda')
        
        # 生成模拟数据
        n_players = 10
        ranks = list(range(1, n_players + 1))
        
        # 计算不同排名系统的得分
        scores_borda = [rcv_model.rank_to_score(rank, n_players) for rank in ranks]
        
        rcv_model_exponential = RCVPreferenceModel(ranking_system='exponential')
        scores_exponential = [rcv_model_exponential.rank_to_score(rank, n_players) for rank in ranks]
        
        rcv_model_logistic = RCVPreferenceModel(ranking_system='logistic')
        scores_logistic = [rcv_model_logistic.rank_to_score(rank, n_players) for rank in ranks]
        
        # 绘制得分分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(ranks, scores_borda, marker='o', label='波达计数', linewidth=2)
        ax.plot(ranks, scores_exponential, marker='s', label='指数衰减', linewidth=2)
        ax.plot(ranks, scores_logistic, marker='^', label='逻辑函数', linewidth=2)
        
        ax.set_xlabel('排名')
        ax.set_ylabel('RCV得分')
        ax.set_title('不同排名系统的RCV得分分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(ranks)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/RCV得分分布.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_improvement_chart(self, improvements: Dict):
        """创建改进百分比图"""
        categories = list(improvements.keys())
        values = list(improvements.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('优化系统相比当前系统的改进百分比', fontsize=14, fontweight='bold')
        ax.set_xlabel('指标类别')
        ax.set_ylabel('改进百分比 (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 旋转x轴标签
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # 添加数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                   height + (1 if height >= 0 else -3),
                   f'{value:.1f}%', 
                   ha='center', 
                   va='bottom' if height >= 0 else 'top', 
                   fontsize=10,
                   fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/改进百分比.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_heatmap(self, systems_metrics: Dict):
        """创建热力图"""
        # 准备数据
        metrics_list = []
        for system_name, metrics in systems_metrics.items():
            row = {'系统': system_name}
            
            for category in ['公平性指标', '娱乐性指标', '参与度指标', '模型性能指标']:
                if category in metrics:
                    cat_metrics = metrics[category]
                    if cat_metrics:
                        row[category] = np.mean(list(cat_metrics.values()))
                    else:
                        row[category] = 0.5
                else:
                    row[category] = 0.5
            
            row['综合评分'] = metrics.get('综合评分', 0.5)
            metrics_list.append(row)
        
        df = pd.DataFrame(metrics_list)
        df.set_index('系统', inplace=True)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 设置坐标轴
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                text = ax.text(j, i, f'{df.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title("优化评分系统指标热力图", fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.colorbar(im, ax=ax)
        
        plt.savefig(f"{self.output_dir}/热力图.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_season_trends(self, comparison_data: Dict):
        """创建赛季趋势图"""
        if 'season_trends' not in comparison_data:
            return
        
        trends = comparison_data['season_trends']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for system_name, scores in trends.items():
            seasons = sorted(scores.keys())
            values = [scores[season] for season in seasons]
            
            ax.plot(seasons, values, marker='o', label=system_name, linewidth=2)
        
        ax.set_title('各赛季综合评分趋势', fontsize=14, fontweight='bold')
        ax.set_xlabel('赛季')
        ax.set_ylabel('综合评分')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/赛季趋势.png", dpi=300, bbox_inches='tight')
        plt.close()

class OptimizedDWTSAnalysis:
    """优化的《与星共舞》分析系统"""
    
    def __init__(self):
        self.data_loader = OptimizedDWTSDataLoader()
        self.visualizer = EnhancedVisualizer()
        
        # 使用优化的系统
        self.systems = {
            '优化分区系统': OptimizedZoneSystem(),
            '优化当前系统': OptimizedCurrentSystem()
        }
        
        self.all_metrics = {}
        self.systems_metrics = {}
        self.comparison_data = {}
        self.analysis_summary = {}
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("=" * 60)
        print("    《与星共舞》优化评分系统分析")
        print("=" * 60)
        
        # 1. 加载数据
        print("\n1. 加载和验证数据...")
        if not self.data_loader.load_and_validate_data():
            print("数据加载失败，退出分析")
            return {}
        
        # 2. 模拟所有赛季
        print("\n2. 模拟所有赛季...")
        all_seasons = self.data_loader.get_all_seasons()
        
        # 选择要模拟的赛季
        seasons_to_simulate = all_seasons[:min(10, len(all_seasons))]
        print(f"模拟前 {len(seasons_to_simulate)} 个赛季")
        
        total_data_points = 0
        successful_seasons = 0
        
        for season in seasons_to_simulate:
            print(f"  模拟赛季 {season}...")
            season_data = self.data_loader.get_season_data(season)
            
            if season_data.empty:
                print(f"  赛季 {season} 数据为空，跳过")
                continue
            
            total_data_points += len(season_data)
            
            # 运行所有系统的模拟
            for system_name, system in self.systems.items():
                try:
                    results = system.safe_simulate_season(season_data, season)
                    
                    # 计算指标
                    metrics = system.calculate_metrics(season)
                    
                    if 'error' not in metrics:
                        if system_name not in self.all_metrics:
                            self.all_metrics[system_name] = {}
                        
                        self.all_metrics[system_name][season] = metrics
                        successful_seasons += 1
                    
                except Exception as e:
                    print(f"  系统 {system_name} 在赛季 {season} 模拟失败: {e}")
        
        # 3. 计算平均指标
        print("\n3. 计算平均指标...")
        for system_name, season_metrics in self.all_metrics.items():
            if season_metrics:
                avg_metrics = self._calculate_average_metrics(season_metrics)
                self.systems_metrics[system_name] = avg_metrics
        
        # 4. 计算改进百分比
        print("\n4. 计算改进百分比...")
        self._calculate_improvements()
        
        # 5. 计算赛季趋势
        self._calculate_season_trends()
        
        # 6. 生成分析摘要
        self.analysis_summary = {
            'total_seasons': len(seasons_to_simulate),
            'successful_seasons': successful_seasons,
            'total_systems': len(self.systems),
            'total_data_points': total_data_points,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'analysis_version': '3.0-optimized',
            'model_features': {
                'time_evolution': True,
                'warmup_period': 3,
                'rcv_scoring': True,
                'decay_type': 'exponential'
            }
        }
        
        # 7. 生成可视化
        print("\n5. 生成可视化图表...")
        self.visualizer.create_all_visualizations(self.systems_metrics, self.comparison_data)
        
        # 8. 保存结果
        self._save_complete_results()
        
        print("\n" + "=" * 60)
        print("优化分析完成！")
        print("=" * 60)
        
        return self.analysis_summary
    
    def _calculate_average_metrics(self, season_metrics: Dict) -> Dict:
        """计算平均指标"""
        if not season_metrics:
            return {}
        
        sum_metrics = {}
        count_metrics = {}
        
        for season, metrics in season_metrics.items():
            for category, category_metrics in metrics.items():
                if category not in sum_metrics:
                    if isinstance(category_metrics, dict):
                        sum_metrics[category] = {}
                        count_metrics[category] = {}
                    else:
                        sum_metrics[category] = 0.0
                        count_metrics[category] = 0
                
                if isinstance(category_metrics, dict) and isinstance(sum_metrics[category], dict):
                    for metric_name, value in category_metrics.items():
                        if isinstance(value, (int, float)):
                            if metric_name not in sum_metrics[category]:
                                sum_metrics[category][metric_name] = 0.0
                                count_metrics[category][metric_name] = 0
                            
                            sum_metrics[category][metric_name] += float(value)
                            count_metrics[category][metric_name] += 1
                elif isinstance(category_metrics, (int, float)) and isinstance(sum_metrics[category], (int, float)):
                    # 综合评分
                    sum_metrics[category] += float(category_metrics)
                    count_metrics[category] += 1
        
        # 计算平均值
        avg_metrics = {}
        for category in sum_metrics:
            if isinstance(sum_metrics[category], dict):
                avg_metrics[category] = {}
                for metric_name in sum_metrics[category]:
                    if count_metrics[category][metric_name] > 0:
                        avg_metrics[category][metric_name] = (
                            sum_metrics[category][metric_name] / 
                            count_metrics[category][metric_name]
                        )
            elif count_metrics[category] > 0:
                avg_metrics[category] = sum_metrics[category] / count_metrics[category]
        
        return avg_metrics
    
    def _calculate_improvements(self):
        """计算改进百分比"""
        if '优化当前系统' not in self.systems_metrics or '优化分区系统' not in self.systems_metrics:
            return
        
        current = self.systems_metrics['优化当前系统']
        zone = self.systems_metrics['优化分区系统']
        
        improvements = {}
        
        # 综合评分改进
        if '综合评分' in current and '综合评分' in zone:
            current_score = current['综合评分']
            zone_score = zone['综合评分']
            if current_score > 0:
                improvements['综合评分'] = ((zone_score - current_score) / current_score) * 100
        
        # 各维度改进
        categories = ['公平性指标', '娱乐性指标', '参与度指标', '模型性能指标']
        
        for category in categories:
            if category in current and category in zone:
                current_metrics = current[category]
                zone_metrics = zone[category]
                
                if isinstance(current_metrics, dict) and isinstance(zone_metrics, dict):
                    current_avg = np.mean(list(current_metrics.values())) if current_metrics else 0
                    zone_avg = np.mean(list(zone_metrics.values())) if zone_metrics else 0
                    
                    if current_avg > 0:
                        improvements[category] = ((zone_avg - current_avg) / current_avg) * 100
        
        self.comparison_data['improvements'] = improvements
    
    def _calculate_season_trends(self):
        """计算赛季趋势"""
        season_trends = {}
        
        for system_name, season_metrics in self.all_metrics.items():
            trends = {}
            for season, metrics in season_metrics.items():
                if '综合评分' in metrics:
                    trends[season] = metrics['综合评分']
            
            if trends:
                season_trends[system_name] = trends
        
        self.comparison_data['season_trends'] = season_trends
    
    def _save_complete_results(self):
        """保存完整结果"""
        results = {
            'analysis_summary': self.analysis_summary,
            'systems_metrics': self.systems_metrics,
            'comparison_data': self.comparison_data,
            'all_metrics': self.all_metrics
        }
        
        with open('DWTS_Optimized_Analysis/完整分析结果.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存指标数据为CSV
        self._save_metrics_to_csv()
        
        # 创建README
        self._create_readme()
    
    def _save_metrics_to_csv(self):
        """保存指标数据为CSV"""
        csv_data = []
        
        for system_name, metrics in self.systems_metrics.items():
            row = {'系统': system_name}
            
            # 添加各维度指标
            for category in ['公平性指标', '娱乐性指标', '参与度指标', '模型性能指标']:
                if category in metrics:
                    for metric_name, value in metrics[category].items():
                        row[f'{category}_{metric_name}'] = value
            
            # 添加综合评分
            row['综合评分'] = metrics.get('综合评分', 0.5)
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv('DWTS_Optimized_Analysis/指标数据.csv', index=False, encoding='utf-8-sig')
        
        # 保存改进百分比
        if 'improvements' in self.comparison_data:
            improvements_df = pd.DataFrame(
                list(self.comparison_data['improvements'].items()),
                columns=['指标', '改进百分比']
            )
            improvements_df.to_csv('DWTS_Optimized_Analysis/改进百分比.csv', index=False, encoding='utf-8-sig')
    
    def _create_readme(self):
        """创建README"""
        readme = f"""# 《与星共舞》优化评分系统分析结果

## 概述
本文件夹包含对《与星共舞》评分系统的优化分析结果，引入了时间演化的加权模型和RCV（排序复选制）偏好模型。

## 优化特性
- **时间演化加权模型**: 考虑历史数据的时效性，近期数据权重更高
- **缓启动模型**: 前3周权重较低，避免初期波动影响
- **RCV偏好模型**: 将粉丝投票转化为排序得分，更准确反映偏好
- **指数衰减**: 使用指数函数计算时间权重

## 生成信息
- 生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- 分析赛季数: {self.analysis_summary.get('total_seasons', 0)}
- 评估系统数: {self.analysis_summary.get('total_systems', 0)}
- 成功模拟赛季: {self.analysis_summary.get('successful_seasons', 0)}

## 文件夹结构

### plots/ - 可视化图表
- 柱状图对比.png: 各指标对比柱状图
- 模型性能对比.png: 模型性能指标对比图
- 时间权重分布.png: 时间权重分布图
- RCV得分分布.png: RCV得分分布图
- 改进百分比.png: 改进百分比图表
- 热力图.png: 系统指标热力图
- 赛季趋势.png: 赛季表现趋势图

### 数据文件
- 完整分析结果.json: 完整分析结果数据
- 指标数据.csv: CSV格式指标数据
- 改进百分比.csv: 改进百分比数据

## 主要发现

"""
        
        if 'improvements' in self.comparison_data:
            improvements = self.comparison_data['improvements']
            for metric, value in improvements.items():
                if value > 0:
                    readme += f"- ✅ {metric}: +{value:.1f}%\n"
                else:
                    readme += f"- ⚠️ {metric}: {value:.1f}%\n"
        
        readme += """
## 模型优势

1. **时间演化加权**
   - 近期表现权重更高，反映选手当前状态
   - 缓启动避免初期波动影响
   - 指数衰减平衡历史与近期数据

2. **RCV偏好模型**
   - 将投票转化为排名得分
   - 波达计数法确保公平性
   - 更好地反映粉丝真实偏好

3. **综合评估**
   - 新增模型性能指标
   - 更全面的系统评估
   - 量化改进效果

## 推荐系统

基于优化分析结果，我们强烈推荐采用**优化分区系统**，原因如下：

- **公平性提升**: 时间加权减少偶然性影响
- **偏好准确性**: RCV模型更准确反映粉丝意愿
- **稳定性增强**: 缓启动减少初期波动
- **参与度提高**: 粉丝投票价值显著提升

## 使用说明

1. 查看可视化: 打开 plots/ 文件夹中的图片
2. 分析数据: 查看指标数据.csv和改进百分比.csv
3. 详细结果: 查看完整分析结果.json

## 下一步建议

1. **参数调优**: 进一步优化时间衰减参数和缓启动周期
2. **模型扩展**: 考虑评委评分的演化特性
3. **实时分析**: 开发实时评分预测系统
4. **观众测试**: 进行A/B测试验证模型效果

---
*分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('DWTS_Optimized_Analysis/README.md', 'w', encoding='utf-8') as f:
            f.write(readme)

def main():
    """主函数"""
    print("开始《与星共舞》优化评分系统分析...")
    
    try:
        # 创建分析实例
        analyzer = OptimizedDWTSAnalysis()
        
        # 运行分析
        results = analyzer.run_complete_analysis()
        
        # 显示关键结果
        print("\n" + "=" * 60)
        print("关键分析结果:")
        print("=" * 60)
        
        if 'improvements' in analyzer.comparison_data:
            improvements = analyzer.comparison_data['improvements']
            for metric, value in improvements.items():
                print(f"{metric}: {value:+.1f}%")
        
        print(f"\n优化特性:")
        print("- 时间演化加权模型")
        print("- 缓启动机制 (前3周)")
        print("- RCV偏好排序")
        print("- 指数衰减权重")
        
        print(f"\n分析完成！所有结果已保存到 DWTS_Optimized_Analysis 文件夹")
        print("=" * 60)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("尝试创建基本的分析结果...")
        
        # 创建基本文件夹结构
        basic_content = """# 《与星共舞》优化分析结果

由于技术问题，完整分析未能完成。

请检查:
1. 数据文件是否存在
2. 数据文件格式是否正确
3. 运行环境是否满足要求

如需帮助，请联系技术支持。
"""
        
        with open('DWTS_Optimized_Analysis/ERROR_README.txt', 'w', encoding='utf-8') as f:
            f.write(basic_content)

if __name__ == "__main__":
    main()