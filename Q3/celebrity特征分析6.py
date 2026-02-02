# ============================================================================
# Dancing with the Stars: Enhanced Visualization Analysis System (Optimized)
# Comprehensive Analysis with Real Fan Vote Data
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 导入更多可视化库
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================================
# Helper Functions (Must be defined before use)
# ============================================================================

def categorize_industry(industry):
    """Enhanced industry categorization with more categories"""
    if pd.isna(industry):
        return 'Other'
    
    industry_str = str(industry).lower()
    
    category_mapping = {
        'athlete': 'Athletes',
        'sport': 'Athletes',
        'football': 'Athletes',
        'basketball': 'Athletes',
        'baseball': 'Athletes',
        'hockey': 'Athletes',
        'tennis': 'Athletes',
        'golf': 'Athletes',
        'swimming': 'Athletes',
        'gymnast': 'Athletes',
        'skater': 'Athletes',
        'actor': 'Actors',
        'actress': 'Actors',
        'singer': 'Singers',
        'musician': 'Singers',
        'rapper': 'Singers',
        'band': 'Singers',
        'model': 'Models',
        'tv': 'TV Personalities',
        'television': 'TV Personalities',
        'host': 'TV Personalities',
        'news': 'Journalists',
        'journalist': 'Journalists',
        'anchor': 'Journalists',
        'politician': 'Politicians',
        'dancer': 'Dancers',
        'comedian': 'Comedians',
        'radio': 'Radio Personalities',
        'social media': 'Social Media',
        'influencer': 'Social Media',
        'fashion': 'Fashion',
        'business': 'Business',
        'entrepreneur': 'Business',
        'doctor': 'Professionals',
        'lawyer': 'Professionals',
        'military': 'Professionals',
        'chef': 'Professionals'
    }
    
    for key, value in category_mapping.items():
        if key in industry_str:
            return value
    
    return 'Other'

def get_industry_detail(industry):
    """Get more detailed industry categorization"""
    if pd.isna(industry):
        return 'Unknown'
    
    industry_str = str(industry).lower()
    
    detail_mapping = {
        'nfl': 'Football Player',
        'nba': 'Basketball Player',
        'mlb': 'Baseball Player',
        'nhl': 'Hockey Player',
        'olymp': 'Olympic Athlete',
        'dancer': 'Professional Dancer',
        'choreographer': 'Choreographer',
        'reality tv': 'Reality TV Star',
        'talk show': 'Talk Show Host',
        'supermodel': 'Supermodel',
        'pop star': 'Pop Singer',
        'country singer': 'Country Singer',
        'rock star': 'Rock Singer',
        'broadway': 'Broadway Performer'
    }
    
    for key, value in detail_mapping.items():
        if key in industry_str:
            return value
    
    # Return original if no specific detail found
    return industry_str.title()

def categorize_region(country):
    """Enhanced region categorization"""
    if pd.isna(country):
        return 'Unknown'
    
    country_str = str(country)
    
    region_mapping = {
        'United States': 'North America',
        'USA': 'North America',
        'US': 'North America',
        'Canada': 'North America',
        'England': 'Europe',
        'UK': 'Europe',
        'Scotland': 'Europe',
        'Wales': 'Europe',
        'Ireland': 'Europe',
        'France': 'Europe',
        'Germany': 'Europe',
        'Italy': 'Europe',
        'Spain': 'Europe',
        'Australia': 'Oceania',
        'New Zealand': 'Oceania',
        'Mexico': 'Latin America',
        'Brazil': 'Latin America',
        'Chile': 'Latin America',
        'Cuba': 'Latin America',
        'China': 'Asia',
        'Japan': 'Asia',
        'Korea': 'Asia',
        'Philippines': 'Asia',
        'India': 'Asia'
    }
    
    for key, value in region_mapping.items():
        if key in country_str:
            return value
    
    return 'Other'

def estimate_partner_experience(partner):
    """Estimate partner experience level based on historical performance"""
    if pd.isna(partner) or partner == 'Unknown':
        return 'Unknown'
    
    # Top tier partners (multiple wins, high average scores)
    top_partners = ['Cheryl Burke', 'Derek Hough', 'Mark Ballas', 'Julianne Hough', 
                   'Maksim Chmerkovskiy', 'Tony Dovolani', 'Karina Smirnoff', 'Peta Murgatroyd']
    
    # Mid tier partners (consistent performers)
    mid_partners = ['Edyta Sliwinska', 'Kym Johnson', 'Anna Trebunskaya', 'Louis van Amstel',
                   'Valentin Chmerkovskiy', 'Witney Carson', 'Lindsay Arnold', 'Sharna Burgess']
    
    if partner in top_partners:
        return 'Top Tier'
    elif partner in mid_partners:
        return 'Mid Tier'
    else:
        return 'Other'

def classify_performance_type(row):
    """Classify performance type based on score characteristics"""
    if row['total_weeks'] < 3:
        return 'Insufficient Data'
    
    consistency = row['consistency']
    improvement = row['improvement_trend']
    max_score = row['max_score']
    min_score = row['min_score']
    
    # More detailed classification
    if consistency < 0.05 and improvement > 0.3:
        return 'Consistent Improver'
    elif consistency < 0.05 and improvement < -0.3:
        return 'Consistent Decliner'
    elif consistency < 0.05:
        return 'Consistent Performer'
    elif consistency > 0.2 and improvement > 0.3:
        return 'Volatile Improver'
    elif consistency > 0.2 and improvement < -0.3:
        return 'Volatile Decliner'
    elif consistency > 0.2:
        return 'Volatile Performer'
    elif max_score - min_score > 5:
        return 'Peaky Performer'
    else:
        return 'Average Performer'

# ============================================================================
# Configuration: Enhanced Visualization Setup
# ============================================================================

def setup_enhanced_visualization():
    """Setup enhanced visualization parameters"""
    # Use default English fonts
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set DPI for high-resolution output
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    
    # Enhanced styling
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set color palettes
    sns.set_palette("husl")
    
    # Set figure aesthetics
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    print("Enhanced visualization setup complete")

def create_output_directory():
    """Create output directory for all results"""
    # 创建带时间戳的唯一目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"DWTS_RealFanVote_Analysis_Results_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
    
    # Create comprehensive subdirectory structure
    subdirs = [
        'figures/radar_charts',
        'figures/heatmaps',
        'figures/network_graphs',
        'figures/3d_plots',
        'figures/interactive',
        'figures/comparison_charts',
        'figures/time_series',
        'figures/distribution',
        'figures/correlation',
        'figures/impact_analysis',
        'tables',
        'reports',
        'data_exports'
    ]
    
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
    
    return output_dir

# ============================================================================
# Enhanced Data Loading and Preprocessing with Real Fan Vote Data
# ============================================================================

def load_all_data():
    """Load all required data files with real fan vote data"""
    print("="*80)
    print("ENHANCED DATA LOADING WITH REAL FAN VOTE DATA")
    print("="*80)
    
    # 1. Load main data
    try:
        df_main = pd.read_csv('2026_MCM_Problem_C_Data.csv')
        print(f"✓ Main data loaded: {len(df_main)} rows, {len(df_main.columns)} columns")
    except FileNotFoundError:
        try:
            df_main = pd.read_excel('2026_MCM_Problem_C_Data.xlsx')
            print(f"✓ Main data loaded from Excel: {len(df_main)} rows")
        except:
            print("✗ ERROR: Could not load main data")
            return None, None
    
    # 2. Load real fan vote estimates
    try:
        fan_votes_df = pd.read_excel('full_estimated_fan_votes.xlsx', sheet_name=0)
        print(f"✓ Real fan vote data loaded: {len(fan_votes_df)} rows, {len(fan_votes_df.columns)} columns")
        
        # Clean and aggregate fan vote data
        fan_votes_agg = fan_votes_df.groupby('Name').agg({
            'Fan_Vote_Mean': 'mean',  # Average fan vote across weeks
            'Fan_Vote_Std': 'mean'    # Average std across weeks
        }).reset_index()
        
        print(f"✓ Fan vote data aggregated for {len(fan_votes_agg)} unique celebrities")
        
    except Exception as e:
        print(f"✗ ERROR: Could not load fan vote data: {e}")
        return None, None
    
    return df_main, fan_votes_agg

def preprocess_enhanced_data(df_main, fan_votes_df):
    """Enhanced preprocessing with real fan vote data and advanced features"""
    print("\nEnhanced data preprocessing with real fan vote data...")
    
    df = df_main.copy()
    
    # 1. Basic performance metrics
    df['final_placement'] = df['placement']
    df['is_winner'] = df['final_placement'] == 1
    df['is_top_3'] = df['final_placement'] <= 3
    df['is_top_5'] = df['final_placement'] <= 5
    
    # 2. Enhanced judge score analysis
    score_columns = [col for col in df.columns if 'judge' in col.lower() and 'score' in col.lower()]
    
    def calculate_enhanced_score_stats(row):
        scores = []
        weeks = []
        judge_scores_dict = {}
        
        for col in score_columns:
            if pd.notna(row[col]) and row[col] > 0:
                scores.append(row[col])
                week_num = int(col.split('_')[0].replace('week', ''))
                weeks.append(week_num)
                
                # Extract judge number
                judge_num = col.split('_')[2] if len(col.split('_')) > 2 else '1'
                judge_scores_dict[f'judge_{judge_num}_score'] = row[col]
        
        if scores:
            avg_score = np.mean(scores)
            score_std = np.std(scores) if len(scores) > 1 else 0
            
            # Calculate improvement trend
            if len(scores) >= 3:
                try:
                    trend = np.polyfit(weeks, scores, 1)[0]
                except:
                    trend = 0
            else:
                trend = 0
            
            # Calculate consistency metrics
            consistency = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
            
            # Calculate peak performance
            peak_performance = np.max(scores) if scores else 0
            
            return {
                'avg_score': avg_score,
                'score_std': score_std,
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'total_weeks': len(scores),
                'improvement_trend': trend,
                'consistency': consistency,
                'peak_performance': peak_performance,
                'score_range': np.max(scores) - np.min(scores) if scores else 0
            }
        else:
            return {'avg_score': 0, 'score_std': 0, 'max_score': 0, 'min_score': 0, 
                   'total_weeks': 0, 'improvement_trend': 0, 'consistency': 0, 
                   'peak_performance': 0, 'score_range': 0}
    
    score_stats = df.apply(calculate_enhanced_score_stats, axis=1, result_type='expand')
    
    for col in score_stats.columns:
        df[col] = score_stats[col]
    
    # Rename for clarity
    df['avg_judge_score'] = df['avg_score']
    df['judge_score_std'] = df['score_std']
    df['judge_score_range'] = df['score_range']
    
    # 3. Enhanced celebrity features
    # Industry categorization with more detail
    df['industry'] = df['celebrity_industry'].fillna('Unknown')
    df['industry_group'] = df['industry'].apply(categorize_industry)
    
    # Create industry subcategories for detailed analysis
    df['industry_detail'] = df['industry'].apply(get_industry_detail)
    
    # Age features with more granular groups
    df['age'] = df['celebrity_age_during_season']
    df['age_group'] = pd.cut(
        df['age'].fillna(df['age'].median()),
        bins=[0, 20, 25, 30, 35, 40, 45, 50, 60, 100],
        labels=['<20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-59', '60+']
    )
    
    # Age-performance metrics
    df['age_performance_score'] = df['avg_judge_score'] / df['age']
    df['age_group_performance'] = df.groupby('age_group')['avg_judge_score'].transform('mean')
    
    # Location features
    df['country'] = df['celebrity_homecountry/region'].fillna('Unknown')
    df['region'] = df['country'].apply(categorize_region)
    df['homestate'] = df['celebrity_homestate'].fillna('Unknown')
    df['is_usa'] = df['country'].str.contains('United States', na=False)
    
    # Professional partner analysis
    df['partner'] = df['ballroom_partner'].fillna('Unknown')
    df['partner_experience'] = df['partner'].apply(estimate_partner_experience)
    
    # Calculate partner success metrics
    partner_stats = df.groupby('partner').agg({
        'avg_judge_score': 'mean',
        'final_placement': 'mean',
        'celebrity_name': 'count'
    }).round(3)
    
    df['partner_avg_score'] = df['partner'].map(partner_stats['avg_judge_score'])
    df['partner_avg_placement'] = df['partner'].map(partner_stats['final_placement'])
    df['partner_experience_count'] = df['partner'].map(partner_stats['celebrity_name'])
    
    # Season features
    df['season_group'] = pd.cut(
        df['season'],
        bins=[0, 5, 10, 15, 20, 25, 30, 35],
        labels=['S1-5', 'S6-10', 'S11-15', 'S16-20', 'S21-25', 'S26-30', 'S31+']
    )
    
    # Performance characteristics
    df['performance_type'] = df.apply(classify_performance_type, axis=1)
    
    # 4. Merge real fan vote data
    print("Merging real fan vote data...")
    
    # Create normalized name for merging
    df['name_normalized'] = df['celebrity_name'].str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)
    fan_votes_df['name_normalized'] = fan_votes_df['Name'].str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)
    
    # Merge fan vote data
    df_merged = pd.merge(
        df, 
        fan_votes_df[['name_normalized', 'Fan_Vote_Mean', 'Fan_Vote_Std']],
        on='name_normalized',
        how='left'
    )
    
    # Handle missing fan vote data
    missing_fan_votes = df_merged['Fan_Vote_Mean'].isna().sum()
    if missing_fan_votes > 0:
        print(f"⚠ {missing_fan_votes} contestants missing fan vote data")
        # Fill missing values with industry averages
        industry_fan_avg = df_merged.groupby('industry_group')['Fan_Vote_Mean'].transform('mean')
        df_merged['Fan_Vote_Mean'] = df_merged['Fan_Vote_Mean'].fillna(industry_fan_avg)
        df_merged['Fan_Vote_Std'] = df_merged['Fan_Vote_Std'].fillna(df_merged['Fan_Vote_Std'].mean())
    
    # 5. Calculate advanced metrics with real fan votes
    df_merged['judge_fan_alignment'] = (df_merged['avg_judge_score'] / 10) - df_merged['Fan_Vote_Mean']
    df_merged['judge_fan_alignment_abs'] = abs(df_merged['judge_fan_alignment'])
    
    # Categorize alignment
    df_merged['judge_fan_correlation_group'] = pd.qcut(
        df_merged['judge_fan_alignment_abs'], 
        3, 
        labels=['High Alignment', 'Medium Alignment', 'Low Alignment']
    )
    
    # Calculate popularity-performance ratio
    df_merged['popularity_performance_ratio'] = df_merged['Fan_Vote_Mean'] / (df_merged['avg_judge_score'] / 10)
    
    # Create success score combining judge and fan scores
    df_merged['combined_success_score'] = (
        df_merged['avg_judge_score'] * 0.6 +  # Judge weight: 60%
        df_merged['Fan_Vote_Mean'] * 10 * 0.4  # Fan weight: 40% (scaled to match judge scores)
    )
    
    # Calculate fan vote consistency
    df_merged['fan_vote_consistency'] = 1 / (df_merged['Fan_Vote_Std'] + 0.001)  # Inverse of std
    
    # Create performance categories based on both judge and fan scores
    def categorize_performance(row):
        judge_percentile = row['avg_judge_score'] / 10
        fan_percentile = row['Fan_Vote_Mean']
        
        if judge_percentile > 0.8 and fan_percentile > 0.8:
            return 'Elite (High Judge & High Fan)'
        elif judge_percentile > 0.8 and fan_percentile <= 0.8:
            return 'Judge Favorite (High Judge, Low Fan)'
        elif judge_percentile <= 0.8 and fan_percentile > 0.8:
            return 'Fan Favorite (Low Judge, High Fan)'
        elif judge_percentile > 0.6 and fan_percentile > 0.6:
            return 'Solid Performer'
        else:
            return 'Underperformer'
    
    df_merged['performance_category'] = df_merged.apply(categorize_performance, axis=1)
    
    print(f"Enhanced preprocessing complete. {len(df_merged)} contestants analyzed")
    print(f"Real fan vote data integrated for {len(fan_votes_df)} celebrities")
    
    return df_merged

# ============================================================================
# Advanced Analysis Functions with Real Fan Vote Data
# ============================================================================

def analyze_feature_impact(df, output_dir):
    """Analyze the impact of celebrity features on judge scores and fan votes"""
    print("\nAnalyzing feature impact on judge scores and fan votes...")
    
    # Define features to analyze
    categorical_features = ['industry_group', 'age_group', 'region', 'partner_experience', 
                          'performance_type', 'performance_category']
    
    numerical_features = ['age', 'total_weeks', 'partner_avg_score']
    
    # Create impact analysis results
    impact_results = []
    
    for feature in categorical_features:
        # Skip if feature has too many unique values or missing data
        if df[feature].nunique() > 20 or df[feature].isna().all():
            continue
        
        # Calculate mean judge score and fan vote by category
        impact_stats = df.groupby(feature).agg({
            'avg_judge_score': ['mean', 'std', 'count'],
            'Fan_Vote_Mean': ['mean', 'std'],
            'judge_fan_alignment': 'mean',
            'combined_success_score': 'mean'
        }).round(3)
        
        # Flatten column names
        impact_stats.columns = ['_'.join(col).strip() for col in impact_stats.columns.values]
        
        # Calculate impact metrics
        impact_stats['judge_score_rank'] = impact_stats['avg_judge_score_mean'].rank(ascending=False)
        impact_stats['fan_vote_rank'] = impact_stats['Fan_Vote_Mean_mean'].rank(ascending=False)
        impact_stats['success_score_rank'] = impact_stats['combined_success_score_mean'].rank(ascending=False)
        
        # Calculate performance gap (difference between judge and fan ranks)
        impact_stats['judge_fan_rank_gap'] = abs(impact_stats['judge_score_rank'] - impact_stats['fan_vote_rank'])
        
        # Add to results
        for category, stats in impact_stats.iterrows():
            impact_results.append({
                'Feature': feature,
                'Category': category,
                'Avg_Judge_Score': stats['avg_judge_score_mean'],
                'Avg_Fan_Vote': stats['Fan_Vote_Mean_mean'],
                'Judge_Fan_Alignment': stats['judge_fan_alignment_mean'],
                'Success_Score': stats['combined_success_score_mean'],
                'Judge_Rank': int(stats['judge_score_rank']),
                'Fan_Rank': int(stats['fan_vote_rank']),
                'Success_Rank': int(stats['success_score_rank']),
                'Rank_Gap': int(stats['judge_fan_rank_gap']),
                'Sample_Size': int(stats['avg_judge_score_count'])
            })
    
    # Convert to DataFrame
    impact_df = pd.DataFrame(impact_results)
    
    # Save impact analysis results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    impact_path = os.path.join(output_dir, 'data_exports', f'feature_impact_analysis_{timestamp}.csv')
    
    try:
        impact_df.to_csv(impact_path, index=False)
        print(f"✓ Feature impact analysis saved: {impact_path}")
    except PermissionError as e:
        print(f"✗ Permission denied for {impact_path}. Trying alternative location...")
        # 尝试在当前目录保存
        alt_path = f'feature_impact_analysis_{timestamp}.csv'
        impact_df.to_csv(alt_path, index=False)
        print(f"✓ Feature impact analysis saved to current directory: {alt_path}")
    except Exception as e:
        print(f"✗ Error saving feature impact analysis: {e}")
    
    # Create visualization of feature impact
    create_feature_impact_visualizations(impact_df, output_dir)
    
    return impact_df

def create_feature_impact_visualizations(impact_df, output_dir):
    """Create visualizations showing feature impact on judge scores and fan votes"""
    print("Creating feature impact visualizations...")
    
    # 1. Feature Impact Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Pivot tables for heatmaps
    pivot_judge = impact_df.pivot_table(index='Feature', columns='Category', 
                                       values='Avg_Judge_Score', aggfunc='mean')
    pivot_fan = impact_df.pivot_table(index='Feature', columns='Category', 
                                     values='Avg_Fan_Vote', aggfunc='mean')
    pivot_alignment = impact_df.pivot_table(index='Feature', columns='Category', 
                                          values='Judge_Fan_Alignment', aggfunc='mean')
    pivot_success = impact_df.pivot_table(index='Feature', columns='Category', 
                                        values='Success_Score', aggfunc='mean')
    
    # Plot heatmaps
    heatmaps = [pivot_judge, pivot_fan, pivot_alignment, pivot_success]
    titles = ['Judge Score Impact', 'Fan Vote Impact', 'Judge-Fan Alignment', 'Combined Success Score']
    cmaps = ['RdYlGn', 'RdYlBu', 'RdBu_r', 'viridis']
    
    for idx, (ax, data, title, cmap) in enumerate(zip(axes.flat, heatmaps, titles, cmaps)):
        if not data.empty:
            im = ax.imshow(data.values, aspect='auto', cmap=cmap)
            
            # Add text annotations
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if not np.isnan(data.iloc[i, j]):
                        text = ax.text(j, i, f'{data.iloc[i, j]:.2f}',
                                     ha="center", va="center", 
                                     color="white" if abs(data.iloc[i, j] - data.values.mean()) > data.values.std() else "black")
            
            ax.set_xticks(range(data.shape[1]))
            ax.set_yticks(range(data.shape[0]))
            ax.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(data.index, fontsize=10)
            ax.set_title(f'{title}', fontsize=12, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
    
    plt.suptitle('Feature Impact Analysis: Judge Scores vs Fan Votes', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图片
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    heatmap_path = os.path.join(output_dir, 'figures', 'impact_analysis', f'feature_impact_heatmaps_{timestamp}.png')
    try:
        plt.savefig(heatmap_path, dpi=600, bbox_inches='tight')
        print(f"✓ Feature impact heatmaps saved: {heatmap_path}")
    except Exception as e:
        print(f"✗ Error saving feature impact heatmaps: {e}")
    finally:
        plt.close()
    
    # 2. Top Performing Categories Radar Chart
    fig = plt.figure(figsize=(14, 10))
    
    # Select top categories for each feature
    top_categories = impact_df.sort_values('Success_Score', ascending=False).head(10)
    
    # Create radar chart for top categories
    categories = top_categories['Category'].tolist()
    judge_scores = top_categories['Avg_Judge_Score'].tolist()
    fan_votes = top_categories['Avg_Fan_Vote'].tolist()
    success_scores = top_categories['Success_Score'].tolist()
    
    # Normalize scores for radar chart
    judge_norm = [(x - min(judge_scores)) / (max(judge_scores) - min(judge_scores)) for x in judge_scores]
    fan_norm = [(x - min(fan_votes)) / (max(fan_votes) - min(fan_votes)) for x in fan_votes]
    success_norm = [(x - min(success_scores)) / (max(success_scores) - min(success_scores)) for x in success_scores]
    
    # Create radar chart
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    judge_norm += judge_norm[:1]
    fan_norm += fan_norm[:1]
    success_norm += success_norm[:1]
    angles += angles[:1]
    
    ax.plot(angles, judge_norm, 'o-', linewidth=2, label='Judge Score', color='blue')
    ax.fill(angles, judge_norm, alpha=0.25, color='blue')
    
    ax.plot(angles, fan_norm, 'o-', linewidth=2, label='Fan Vote', color='green')
    ax.fill(angles, fan_norm, alpha=0.25, color='green')
    
    ax.plot(angles, success_norm, 'o-', linewidth=2, label='Success Score', color='red')
    ax.fill(angles, success_norm, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, ha='center')
    ax.set_ylim(0, 1)
    ax.set_title('Top Performing Categories: Judge vs Fan vs Success', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    
    # 保存雷达图
    radar_path = os.path.join(output_dir, 'figures', 'impact_analysis', f'top_categories_radar_{timestamp}.png')
    try:
        plt.savefig(radar_path, dpi=600, bbox_inches='tight')
        print(f"✓ Top categories radar chart saved: {radar_path}")
    except Exception as e:
        print(f"✗ Error saving radar chart: {e}")
    finally:
        plt.close()
    
    # 3. Industry Analysis: Judge vs Fan Performance
    industry_impact = impact_df[impact_df['Feature'] == 'industry_group'].copy()
    
    if not industry_impact.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort by judge score
        industry_impact = industry_impact.sort_values('Avg_Judge_Score', ascending=False)
        
        categories = industry_impact['Category'].tolist()
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, industry_impact['Avg_Judge_Score'].tolist(), 
                      width, label='Judge Score', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, industry_impact['Avg_Fan_Vote'].tolist(), 
                      width, label='Fan Vote (×10)', color='lightgreen', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Industry Group')
        ax.set_ylabel('Score')
        ax.set_title('Industry Performance: Judge Scores vs Fan Votes', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add twin axis for success score
        ax2 = ax.twinx()
        line = ax2.plot(x, industry_impact['Success_Score'].tolist(), 'r-', 
                       marker='o', linewidth=2, label='Success Score')
        ax2.set_ylabel('Success Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lines_labels = [ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        ax.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        
        # 保存行业分析图
        industry_path = os.path.join(output_dir, 'figures', 'impact_analysis', f'industry_performance_{timestamp}.png')
        try:
            plt.savefig(industry_path, dpi=600, bbox_inches='tight')
            print(f"✓ Industry performance chart saved: {industry_path}")
        except Exception as e:
            print(f"✗ Error saving industry chart: {e}")
        finally:
            plt.close()
    
    print("✓ Feature impact visualizations created")

def create_judge_fan_alignment_analysis(df, output_dir):
    """Create detailed analysis of judge-fan alignment"""
    print("\nCreating judge-fan alignment analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Alignment distribution
    ax1 = axes[0, 0]
    alignment_data = df['judge_fan_alignment'].dropna()
    ax1.hist(alignment_data, bins=30, edgecolor='black', alpha=0.7, color='purple', density=True)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Alignment')
    ax1.axvline(alignment_data.mean(), color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {alignment_data.mean():.3f}')
    
    # Add KDE
    kde = stats.gaussian_kde(alignment_data)
    x_range = np.linspace(alignment_data.min(), alignment_data.max(), 1000)
    ax1.plot(x_range, kde(x_range), 'k-', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Judge-Fan Alignment (Judge/10 - Fan Vote)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Judge-Fan Alignment', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Alignment by industry
    ax2 = axes[0, 1]
    industry_alignment = df.groupby('industry_group')['judge_fan_alignment'].agg(['mean', 'std', 'count']).round(3)
    industry_alignment = industry_alignment.sort_values('mean', ascending=False).head(10)
    
    bars = ax2.barh(industry_alignment.index, industry_alignment['mean'].values, 
                   xerr=industry_alignment['std'].values, capsize=5, alpha=0.7)
    
    # Color bars by direction
    for i, bar in enumerate(bars):
        if industry_alignment['mean'].iloc[i] > 0:
            bar.set_color('red')  # Judge favors more than fans
        else:
            bar.set_color('green')  # Fans favor more than judges
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Mean Alignment (Positive = Judge Favors More)')
    ax2.set_title('Judge-Fan Alignment by Industry (Top 10)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Alignment vs Performance
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['avg_judge_score'], df['Fan_Vote_Mean'] * 10,
                         c=df['judge_fan_alignment_abs'], cmap='coolwarm',
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add perfect alignment line
    perfect_line_x = np.linspace(df['avg_judge_score'].min(), df['avg_judge_score'].max(), 100)
    perfect_line_y = perfect_line_x / 10  # Convert to same scale
    ax3.plot(perfect_line_x, perfect_line_y, 'k--', linewidth=1, alpha=0.5, label='Perfect Alignment')
    
    ax3.set_xlabel('Judge Score')
    ax3.set_ylabel('Fan Vote (×10)')
    ax3.set_title('Judge Score vs Fan Vote (Color = Alignment Error)', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Alignment Error (Absolute Value)')
    
    # 4. Alignment by age group
    ax4 = axes[1, 1]
    age_alignment = df.groupby('age_group')['judge_fan_alignment'].agg(['mean', 'std', 'count']).round(3)
    
    bars2 = ax4.bar(range(len(age_alignment)), age_alignment['mean'].values,
                   yerr=age_alignment['std'].values, capsize=5, alpha=0.7,
                   color=plt.cm.coolwarm(np.arange(len(age_alignment)) / len(age_alignment)))
    
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel('Mean Alignment')
    ax4.set_xticks(range(len(age_alignment)))
    ax4.set_xticklabels(age_alignment.index, rotation=45, ha='right')
    ax4.set_title('Judge-Fan Alignment by Age Group', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for i, count in enumerate(age_alignment['count'].values):
        ax4.text(i, age_alignment['mean'].iloc[i] + (0.01 if age_alignment['mean'].iloc[i] >= 0 else -0.03),
                f'n={int(count)}', ha='center', va='bottom' if age_alignment['mean'].iloc[i] >= 0 else 'top',
                fontsize=8)
    
    plt.suptitle('Comprehensive Judge-Fan Alignment Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存对齐分析图
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    alignment_path = os.path.join(output_dir, 'figures', 'impact_analysis', f'judge_fan_alignment_{timestamp}.png')
    try:
        plt.savefig(alignment_path, dpi=600, bbox_inches='tight')
        print(f"✓ Judge-fan alignment analysis saved: {alignment_path}")
    except Exception as e:
        print(f"✗ Error saving alignment analysis: {e}")
    finally:
        plt.close()
    
    # Create alignment summary statistics
    alignment_stats = {
        'Overall Alignment Statistics': {
            'Mean Alignment': f"{alignment_data.mean():.4f}",
            'Alignment Std': f"{alignment_data.std():.4f}",
            'Median Alignment': f"{alignment_data.median():.4f}",
            'Min Alignment': f"{alignment_data.min():.4f}",
            'Max Alignment': f"{alignment_data.max():.4f}",
            'Perfect Alignment Count': f"{(abs(alignment_data) < 0.01).sum()}",
            'Judge Favors More Count': f"{(alignment_data > 0.05).sum()}",
            'Fan Favors More Count': f"{(alignment_data < -0.05).sum()}"
        }
    }
    
    # Save alignment statistics
    alignment_report_path = os.path.join(output_dir, 'reports', f'judge_fan_alignment_stats_{timestamp}.txt')
    try:
        with open(alignment_report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("JUDGE-FAN ALIGNMENT ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            for section, data in alignment_stats.items():
                f.write(f"{section}\n")
                f.write("-"*40 + "\n")
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        print(f"✓ Judge-fan alignment report saved: {alignment_report_path}")
    except Exception as e:
        print(f"✗ Error saving alignment report: {e}")
    
    print("✓ Judge-fan alignment analysis created")

def create_regression_analysis(df, output_dir):
    """Perform regression analysis to quantify feature impact"""
    print("\nPerforming regression analysis...")
    
    # Prepare data for regression
    regression_data = df.copy()
    
    # Create dummy variables for categorical features
    categorical_features = ['industry_group', 'age_group', 'region', 'partner_experience']
    
    for feature in categorical_features:
        if feature in regression_data.columns:
            dummies = pd.get_dummies(regression_data[feature], prefix=feature, drop_first=True)
            regression_data = pd.concat([regression_data, dummies], axis=1)
    
    # Define independent variables (features)
    feature_columns = [
        'age',
        'total_weeks',
        'partner_avg_score',
        'partner_experience_count'
    ] + [col for col in regression_data.columns if any(f in col for f in categorical_features)]
    
    # Remove any columns with too many missing values
    feature_columns = [col for col in feature_columns if col in regression_data.columns]
    
    # Convert all feature columns to numeric, forcing errors to NaN
    for col in feature_columns:
        regression_data[col] = pd.to_numeric(regression_data[col], errors='coerce')
    
    # Also ensure target columns are numeric
    regression_data['avg_judge_score'] = pd.to_numeric(regression_data['avg_judge_score'], errors='coerce')
    regression_data['Fan_Vote_Mean'] = pd.to_numeric(regression_data['Fan_Vote_Mean'], errors='coerce')
    
    # Drop rows with NaN in any of the feature columns or target columns
    all_cols = feature_columns + ['avg_judge_score', 'Fan_Vote_Mean']
    regression_data = regression_data.dropna(subset=all_cols)
    
    print(f"Regression analysis with {len(regression_data)} samples and {len(feature_columns)} features")
    
    # Perform correlation analysis
    correlation_results = []
    
    for feature in feature_columns:
        if feature in regression_data.columns:
            try:
                # Ensure data is numeric
                feature_data = pd.to_numeric(regression_data[feature], errors='coerce')
                judge_data = pd.to_numeric(regression_data['avg_judge_score'], errors='coerce')
                fan_data = pd.to_numeric(regression_data['Fan_Vote_Mean'], errors='coerce')
                
                # Drop NaN values for correlation calculation
                valid_idx = feature_data.notna() & judge_data.notna() & fan_data.notna()
                
                if valid_idx.sum() >= 3:  # Need at least 3 points for correlation
                    # Judge score correlation
                    judge_corr, judge_p = stats.pearsonr(
                        feature_data[valid_idx],
                        judge_data[valid_idx]
                    )
                    
                    # Fan vote correlation
                    fan_corr, fan_p = stats.pearsonr(
                        feature_data[valid_idx],
                        fan_data[valid_idx]
                    )
                    
                    correlation_results.append({
                        'Feature': feature,
                        'Judge_Correlation': judge_corr,
                        'Judge_P_Value': judge_p,
                        'Fan_Correlation': fan_corr,
                        'Fan_P_Value': fan_p,
                        'Correlation_Difference': abs(judge_corr - fan_corr),
                        'Significant_Judge': judge_p < 0.05,
                        'Significant_Fan': fan_p < 0.05,
                        'Sample_Size': int(valid_idx.sum())
                    })
                else:
                    print(f"Warning: Insufficient data for feature {feature}")
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {feature}: {e}")
    
    # Convert to DataFrame
    correlation_df = pd.DataFrame(correlation_results)
    
    if correlation_df.empty:
        print("Warning: No correlation results generated")
        return pd.DataFrame()
    
    # Sort by absolute correlation difference
    correlation_df['Abs_Judge_Corr'] = abs(correlation_df['Judge_Correlation'])
    correlation_df['Abs_Fan_Corr'] = abs(correlation_df['Fan_Correlation'])
    correlation_df = correlation_df.sort_values('Abs_Judge_Corr', ascending=False)
    
    # Save correlation results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    correlation_path = os.path.join(output_dir, 'data_exports', f'feature_correlation_analysis_{timestamp}.csv')
    
    try:
        correlation_df.to_csv(correlation_path, index=False)
        print(f"✓ Correlation analysis saved: {correlation_path}")
    except PermissionError as e:
        print(f"✗ Permission denied for {correlation_path}. Trying alternative location...")
        # 尝试在当前目录保存
        alt_path = f'feature_correlation_analysis_{timestamp}.csv'
        correlation_df.to_csv(alt_path, index=False)
        print(f"✓ Correlation analysis saved to current directory: {alt_path}")
    except Exception as e:
        print(f"✗ Error saving correlation analysis: {e}")
    
    # Create visualization of top correlations
    top_features = correlation_df.head(min(15, len(correlation_df))).copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Judge correlation plot
    ax1 = axes[0]
    if not top_features.empty:
        bars1 = ax1.barh(range(len(top_features)), top_features['Judge_Correlation'].values,
                        color=plt.cm.RdYlGn((top_features['Judge_Correlation'].values + 1) / 2))
        
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['Feature'].tolist())
        ax1.set_xlabel('Correlation Coefficient')
        ax1.set_title('Top Features Correlated with Judge Scores', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add significance markers
        for i, (bar, sig) in enumerate(zip(bars1, top_features['Significant_Judge'])):
            if sig:
                ax1.text(bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.03),
                        bar.get_y() + bar.get_height()/2, '*', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='red')
    
    # Fan correlation plot
    ax2 = axes[1]
    if not top_features.empty:
        bars2 = ax2.barh(range(len(top_features)), top_features['Fan_Correlation'].values,
                        color=plt.cm.RdYlBu((top_features['Fan_Correlation'].values + 1) / 2))
        
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['Feature'].tolist())
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_title('Top Features Correlated with Fan Votes', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add significance markers
        for i, (bar, sig) in enumerate(zip(bars2, top_features['Significant_Fan'])):
            if sig:
                ax2.text(bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.03),
                        bar.get_y() + bar.get_height()/2, '*', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='red')
    
    plt.suptitle('Feature Correlation Analysis: Judge Scores vs Fan Votes', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存相关性分析图
    correlation_plot_path = os.path.join(output_dir, 'figures', 'impact_analysis', f'feature_correlations_{timestamp}.png')
    try:
        plt.savefig(correlation_plot_path, dpi=600, bbox_inches='tight')
        print(f"✓ Feature correlations plot saved: {correlation_plot_path}")
    except Exception as e:
        print(f"✗ Error saving correlations plot: {e}")
    finally:
        plt.close()
    
    # Create summary of significant findings
    significant_judge = correlation_df[correlation_df['Significant_Judge']].copy() if not correlation_df.empty else pd.DataFrame()
    significant_fan = correlation_df[correlation_df['Significant_Fan']].copy() if not correlation_df.empty else pd.DataFrame()
    
    # Save significant findings
    significant_path = os.path.join(output_dir, 'reports', f'significant_correlations_{timestamp}.txt')
    try:
        with open(significant_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SIGNIFICANT FEATURE CORRELATIONS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("FEATURES SIGNIFICANTLY CORRELATED WITH JUDGE SCORES:\n")
            f.write("-"*60 + "\n")
            if not significant_judge.empty:
                for _, row in significant_judge.iterrows():
                    direction = "Positive" if row['Judge_Correlation'] > 0 else "Negative"
                    f.write(f"{row['Feature']}: {direction} correlation (r={row['Judge_Correlation']:.3f}, p={row['Judge_P_Value']:.4f})\n")
                f.write(f"\nTotal: {len(significant_judge)} significant features\n\n")
            else:
                f.write("No significant features found for Judge Scores\n\n")
            
            f.write("\nFEATURES SIGNIFICANTLY CORRELATED WITH FAN VOTES:\n")
            f.write("-"*60 + "\n")
            if not significant_fan.empty:
                for _, row in significant_fan.iterrows():
                    direction = "Positive" if row['Fan_Correlation'] > 0 else "Negative"
                    f.write(f"{row['Feature']}: {direction} correlation (r={row['Fan_Correlation']:.3f}, p={row['Fan_P_Value']:.4f})\n")
                f.write(f"\nTotal: {len(significant_fan)} significant features\n\n")
            else:
                f.write("No significant features found for Fan Votes\n\n")
            
            # Find features with opposite effects
            opposite_effects = []
            if not correlation_df.empty:
                for _, row in correlation_df.iterrows():
                    if row['Significant_Judge'] and row['Significant_Fan']:
                        if row['Judge_Correlation'] * row['Fan_Correlation'] < 0:
                            opposite_effects.append({
                                'Feature': row['Feature'],
                                'Judge_Correlation': row['Judge_Correlation'],
                                'Fan_Correlation': row['Fan_Correlation'],
                                'Difference': abs(row['Judge_Correlation'] - row['Fan_Correlation'])
                            })
            
            if opposite_effects:
                f.write("\nFEATURES WITH OPPOSITE EFFECTS ON JUDGES VS FANS:\n")
                f.write("-"*60 + "\n")
                for effect in sorted(opposite_effects, key=lambda x: x['Difference'], reverse=True):
                    f.write(f"{effect['Feature']}: Judge r={effect['Judge_Correlation']:.3f}, Fan r={effect['Fan_Correlation']:.3f} (Difference: {effect['Difference']:.3f})\n")
        print(f"✓ Significant correlations report saved: {significant_path}")
    except Exception as e:
        print(f"✗ Error saving significant correlations report: {e}")
    
    print("✓ Regression analysis completed")
    return correlation_df

# ============================================================================
# Enhanced Visualization Functions with Real Fan Vote Data
# ============================================================================

def create_radar_chart(df, output_dir):
    """Create radar charts for different industry groups using real fan vote data"""
    print("Creating radar charts with real fan vote data...")
    
    # Select top industries
    top_industries = df['industry_group'].value_counts().head(6).index.tolist()
    
    # Metrics to display (now including real fan votes)
    metrics = ['avg_judge_score', 'Fan_Vote_Mean', 'final_placement', 'age', 'total_weeks', 'improvement_trend']
    metric_names = ['Judge Score', 'Fan Vote', 'Placement', 'Age', 'Weeks', 'Improvement']
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for industry in top_industries:
        industry_data = df[df['industry_group'] == industry]
        
        if len(industry_data) < 3:
            continue
        
        # Calculate normalized metrics (0-1 scale)
        normalized_metrics = []
        for metric in metrics:
            if metric == 'final_placement':
                # Invert placement (lower is better)
                value = 1 - (industry_data[metric].mean() / df[metric].max())
            elif metric == 'age':
                # Normalize age (lower is better for performance)
                value = 1 - (industry_data[metric].mean() / df[metric].max())
            elif metric == 'Fan_Vote_Mean':
                # Fan vote is already 0-1 scale
                value = industry_data[metric].mean()
            else:
                # Normalize other metrics
                min_val = df[metric].min()
                max_val = df[metric].max()
                value = (industry_data[metric].mean() - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            normalized_metrics.append(value)
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        normalized_metrics += normalized_metrics[:1]  # Close the polygon
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot industry data
        ax.plot(angles, normalized_metrics, 'o-', linewidth=2, label=industry, 
                color=plt.cm.Set2(list(top_industries).index(industry)))
        ax.fill(angles, normalized_metrics, alpha=0.25, 
                color=plt.cm.Set2(list(top_industries).index(industry)))
        
        # Plot average for comparison
        avg_metrics = []
        for metric in metrics:
            if metric == 'final_placement':
                value = 1 - (df[metric].mean() / df[metric].max())
            elif metric == 'age':
                value = 1 - (df[metric].mean() / df[metric].max())
            elif metric == 'Fan_Vote_Mean':
                value = df[metric].mean()
            else:
                min_val = df[metric].min()
                max_val = df[metric].max()
                value = (df[metric].mean() - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            avg_metrics.append(value)
        
        avg_metrics += avg_metrics[:1]
        ax.plot(angles, avg_metrics, 'k--', linewidth=1, label='Average')
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Radar: {industry}\n(Real Fan Vote Data)', 
                    size=16, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        # 保存雷达图
        radar_path = os.path.join(output_dir, 'figures', 'radar_charts', f'radar_{industry.replace(" ", "_")}_{timestamp}.png')
        try:
            plt.savefig(radar_path, dpi=600, bbox_inches='tight')
        except Exception as e:
            print(f"✗ Error saving radar chart for {industry}: {e}")
        finally:
            plt.close()
    
    print("✓ Radar charts created with real fan vote data")

def create_heatmap_matrix(df, output_dir):
    """Create heatmap matrices for correlation and performance with real fan vote data"""
    print("Creating heatmap matrices with real fan vote data...")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. Enhanced Correlation Heatmap with Fan Vote Data
    corr_columns = ['avg_judge_score', 'Fan_Vote_Mean', 'final_placement', 'age', 
                   'total_weeks', 'improvement_trend', 'consistency', 'judge_fan_alignment']
    corr_matrix = df[corr_columns].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(len(corr_columns)):
        for j in range(len(corr_columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
    
    # Customize
    ax.set_xticks(np.arange(len(corr_columns)))
    ax.set_yticks(np.arange(len(corr_columns)))
    ax.set_xticklabels([c.replace('_', ' ').replace('Mean', '').title() for c in corr_columns], 
                      rotation=45, ha='right')
    ax.set_yticklabels([c.replace('_', ' ').replace('Mean', '').title() for c in corr_columns])
    ax.set_title('Correlation Matrix with Real Fan Vote Data', fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
    
    plt.tight_layout()
    
    # 保存相关性热力图
    correlation_heatmap_path = os.path.join(output_dir, 'figures', 'heatmaps', f'correlation_heatmap_fanvote_{timestamp}.png')
    try:
        plt.savefig(correlation_heatmap_path, dpi=600, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved: {correlation_heatmap_path}")
    except Exception as e:
        print(f"✗ Error saving correlation heatmap: {e}")
    finally:
        plt.close()
    
    # 2. Industry Performance Heatmap with Fan Votes
    industry_stats = df.groupby('industry_group').agg({
        'avg_judge_score': 'mean',
        'Fan_Vote_Mean': 'mean',
        'final_placement': 'mean',
        'total_weeks': 'mean',
        'judge_fan_alignment': 'mean'
    })
    
    # Create a comprehensive industry heatmap
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Heatmap 1: Raw performance metrics
    normalized_stats = (industry_stats - industry_stats.min()) / (industry_stats.max() - industry_stats.min())
    
    im1 = axes[0].imshow(normalized_stats.T, cmap='YlOrRd', aspect='auto')
    
    # Add text annotations with actual values
    for i in range(len(industry_stats)):
        for j in range(len(industry_stats.columns)):
            actual_value = industry_stats.iloc[i, j]
            axes[0].text(i, j, f'{actual_value:.2f}',
                        ha="center", va="center", 
                        color="black" if actual_value < industry_stats.iloc[:, j].median() else "white")
    
    axes[0].set_xticks(np.arange(len(industry_stats)))
    axes[0].set_yticks(np.arange(len(industry_stats.columns)))
    axes[0].set_xticklabels(industry_stats.index, rotation=45, ha='right')
    axes[0].set_yticklabels([col.replace('_', ' ').title() for col in industry_stats.columns])
    axes[0].set_title('Industry Performance Heatmap (Normalized)', fontsize=14, fontweight='bold')
    
    plt.colorbar(im1, ax=axes[0], label='Normalized Score')
    
    # Heatmap 2: Rank comparison (Judge vs Fan)
    industry_stats['judge_rank'] = industry_stats['avg_judge_score'].rank(ascending=False)
    industry_stats['fan_rank'] = industry_stats['Fan_Vote_Mean'].rank(ascending=False)
    industry_stats['rank_difference'] = abs(industry_stats['judge_rank'] - industry_stats['fan_rank'])
    
    rank_data = industry_stats[['judge_rank', 'fan_rank', 'rank_difference']]
    
    im2 = axes[1].imshow(rank_data.T, cmap='viridis', aspect='auto')
    
    for i in range(len(rank_data)):
        for j in range(len(rank_data.columns)):
            axes[1].text(i, j, f'{rank_data.iloc[i, j]:.1f}',
                        ha="center", va="center", 
                        color="white" if rank_data.iloc[i, j] > rank_data.iloc[:, j].median() else "black")
    
    axes[1].set_xticks(np.arange(len(rank_data)))
    axes[1].set_yticks(np.arange(len(rank_data.columns)))
    axes[1].set_xticklabels(rank_data.index, rotation=45, ha='right')
    axes[1].set_yticklabels(['Judge Rank', 'Fan Rank', 'Rank Difference'])
    axes[1].set_title('Industry Rank Comparison: Judge vs Fan', fontsize=14, fontweight='bold')
    
    plt.colorbar(im2, ax=axes[1], label='Rank Value')
    
    plt.suptitle('Industry Analysis with Real Fan Vote Data', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存行业热力图
    industry_heatmap_path = os.path.join(output_dir, 'figures', 'heatmaps', f'industry_heatmap_fanvote_{timestamp}.png')
    try:
        plt.savefig(industry_heatmap_path, dpi=600, bbox_inches='tight')
        print(f"✓ Industry heatmap saved: {industry_heatmap_path}")
    except Exception as e:
        print(f"✗ Error saving industry heatmap: {e}")
    finally:
        plt.close()
    
    print("✓ Heatmap matrices created with real fan vote data")

def create_3d_scatter_plots(df, output_dir):
    """Create 3D scatter plots for multidimensional analysis with real fan vote data"""
    print("Creating 3D scatter plots with real fan vote data...")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Judge Score vs Fan Vote vs Placement (with real fan votes)
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Color by performance category
    performance_categories = df['performance_category'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(performance_categories)))
    
    for i, category in enumerate(performance_categories):
        category_data = df[df['performance_category'] == category]
        if len(category_data) > 0:
            ax1.scatter(category_data['avg_judge_score'], 
                       category_data['Fan_Vote_Mean'] * 10, 
                       category_data['final_placement'],
                       c=[colors[i]], label=category, alpha=0.6, s=50)
    
    ax1.set_xlabel('Judge Score', labelpad=10)
    ax1.set_ylabel('Fan Vote (×10)', labelpad=10)
    ax1.set_zlabel('Final Placement', labelpad=10)
    ax1.set_title('3D: Judge Score vs Real Fan Vote vs Placement', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    # 2. Age vs Experience vs Performance (with fan vote as size)
    ax2 = fig.add_subplot(222, projection='3d')
    
    # Bubble size based on real fan vote
    sizes = df['Fan_Vote_Mean'] * 500
    
    scatter = ax2.scatter(df['age'], df['total_weeks'], df['avg_judge_score'],
                         c=df['Fan_Vote_Mean'], cmap='RdYlBu', s=sizes, alpha=0.7)
    
    ax2.set_xlabel('Age', labelpad=10)
    ax2.set_ylabel('Total Weeks', labelpad=10)
    ax2.set_zlabel('Judge Score', labelpad=10)
    ax2.set_title('3D: Age vs Experience vs Judge Score\n(Bubble Size = Fan Vote)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.1)
    cbar.set_label('Fan Vote', rotation=270, labelpad=15)
    
    # 3. Performance consistency 3D plot with alignment
    ax3 = fig.add_subplot(223, projection='3d')
    
    # Filter for sufficient data
    valid_data = df[df['total_weeks'] >= 3]
    
    # Create surface-like visualization
    x = valid_data['consistency']
    y = valid_data['improvement_trend']
    z = valid_data['avg_judge_score']
    
    # Color by judge-fan alignment
    colors_3d = plt.cm.RdBu((valid_data['judge_fan_alignment'] - valid_data['judge_fan_alignment'].min()) / 
                           (valid_data['judge_fan_alignment'].max() - valid_data['judge_fan_alignment'].min()))
    
    scatter3 = ax3.scatter(x, y, z, c=colors_3d, s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    ax3.set_xlabel('Consistency (Lower = Better)', labelpad=10)
    ax3.set_ylabel('Improvement Trend', labelpad=10)
    ax3.set_zlabel('Judge Score', labelpad=10)
    ax3.set_title('3D: Consistency vs Improvement vs Score\n(Color = Judge-Fan Alignment)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar3 = fig.colorbar(scatter3, ax=ax3, pad=0.1)
    cbar3.set_label('Judge-Fan Alignment\n(Positive = Judge Favors More)', rotation=270, labelpad=20)
    
    # 4. Season progression 3D with real fan votes
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Group by season group
    season_groups = df['season_group'].unique()
    colors_season = plt.cm.plasma(np.linspace(0, 1, len(season_groups)))
    
    for i, group in enumerate(season_groups):
        group_data = df[df['season_group'] == group]
        if len(group_data) > 0:
            ax4.scatter(group_data['season'], 
                       group_data['avg_judge_score'], 
                       group_data['Fan_Vote_Mean'] * 10,
                       c=[colors_season[i]], label=str(group), alpha=0.6, s=50)
    
    ax4.set_xlabel('Season', labelpad=10)
    ax4.set_ylabel('Judge Score', labelpad=10)
    ax4.set_zlabel('Real Fan Vote (×10)', labelpad=10)
    ax4.set_title('3D: Season vs Judge Score vs Real Fan Vote', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    plt.suptitle('3D Analysis with Real Fan Vote Data', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存3D散点图
    scatter_3d_path = os.path.join(output_dir, 'figures', '3d_plots', f'3d_scatter_real_fanvote_{timestamp}.png')
    try:
        plt.savefig(scatter_3d_path, dpi=600, bbox_inches='tight')
        print(f"✓ 3D scatter plots saved: {scatter_3d_path}")
    except Exception as e:
        print(f"✗ Error saving 3D scatter plots: {e}")
    finally:
        plt.close()
    
    print("✓ 3D scatter plots created with real fan vote data")

def create_time_series_analysis(df, output_dir):
    """Create time series analysis of performance trends with real fan vote data"""
    print("Creating time series analysis with real fan vote data...")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Judge Score and Fan Vote Trend Over Seasons
    season_stats = df.groupby('season').agg({
        'avg_judge_score': ['mean', 'std'],
        'Fan_Vote_Mean': ['mean', 'std'],
        'judge_fan_alignment': 'mean',
        'celebrity_name': 'count'
    }).round(3)
    
    # Flatten column names
    season_stats.columns = ['judge_mean', 'judge_std', 'fan_mean', 'fan_std', 'alignment_mean', 'count']
    
    # Plot with confidence intervals
    axes[0, 0].fill_between(season_stats.index, 
                           season_stats['judge_mean'] - season_stats['judge_std'],
                           season_stats['judge_mean'] + season_stats['judge_std'],
                           alpha=0.3, color='blue', label='Judge ±1 Std Dev')
    
    axes[0, 0].fill_between(season_stats.index, 
                           season_stats['fan_mean'] * 10 - season_stats['fan_std'] * 10,
                           season_stats['fan_mean'] * 10 + season_stats['fan_std'] * 10,
                           alpha=0.2, color='green', label='Fan ±1 Std Dev')
    
    axes[0, 0].plot(season_stats.index, season_stats['judge_mean'], 
                   'o-', linewidth=2, markersize=6, color='blue', label='Mean Judge Score')
    
    axes[0, 0].plot(season_stats.index, season_stats['fan_mean'] * 10, 
                   's-', linewidth=2, markersize=4, color='green', label='Mean Fan Vote (×10)')
    
    # Add trend lines
    z_judge = np.polyfit(season_stats.index, season_stats['judge_mean'], 1)
    p_judge = np.poly1d(z_judge)
    axes[0, 0].plot(season_stats.index, p_judge(season_stats.index), 'b--', 
                   linewidth=2, alpha=0.8, label='Judge Trend')
    
    z_fan = np.polyfit(season_stats.index, season_stats['fan_mean'], 1)
    p_fan = np.poly1d(z_fan)
    axes[0, 0].plot(season_stats.index, p_fan(season_stats.index) * 10, 'g--', 
                   linewidth=2, alpha=0.8, label='Fan Trend')
    
    axes[0, 0].set_xlabel('Season')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Judge Score and Real Fan Vote Evolution Over Seasons', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Judge-Fan Alignment Trend
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    # Plot alignment
    line1 = ax2.plot(season_stats.index, season_stats['alignment_mean'], 
                    'o-', color='purple', linewidth=2, markersize=6, label='Judge-Fan Alignment')
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Alignment', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(-0.5, 0.5)
    
    # Plot contestant count
    line2 = ax2_twin.plot(season_stats.index, season_stats['count'], 
                         's-', color='orange', linewidth=2, markersize=4, alpha=0.7, label='Contestant Count')
    
    ax2_twin.set_ylabel('Contestant Count', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    ax2.set_title('Judge-Fan Alignment and Contestant Count Over Seasons', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Categories Over Time
    axes[1, 0].stackplot(season_stats.index, 
                        [season_stats['judge_mean'], season_stats['fan_mean'] * 10],
                        labels=['Judge Score', 'Fan Vote (×10)'], alpha=0.7)
    
    axes[1, 0].set_xlabel('Season')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Stacked Performance: Judge vs Fan Over Time', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Correlation Over Time
    # Calculate rolling correlation between judge scores and fan votes
    seasons = sorted(df['season'].unique())
    correlations = []
    
    for season in seasons:
        season_data = df[df['season'] == season]
        if len(season_data) > 5:
            corr = season_data['avg_judge_score'].corr(season_data['Fan_Vote_Mean'])
            correlations.append(corr)
        else:
            correlations.append(np.nan)
    
    axes[1, 1].plot(seasons[:len(correlations)], correlations, 
                   'o-', linewidth=2, markersize=8, color='red')
    
    axes[1, 1].fill_between(seasons[:len(correlations)], 
                           np.array(correlations) - 0.1,
                           np.array(correlations) + 0.1,
                           alpha=0.3, color='red')
    
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Season')
    axes[1, 1].set_ylabel('Correlation Coefficient')
    axes[1, 1].set_title('Judge-Fan Correlation Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Time Series Analysis with Real Fan Vote Data', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存时间序列分析图
    time_series_path = os.path.join(output_dir, 'figures', 'time_series', f'performance_trends_fanvote_{timestamp}.png')
    try:
        plt.savefig(time_series_path, dpi=600, bbox_inches='tight')
        print(f"✓ Time series analysis saved: {time_series_path}")
    except Exception as e:
        print(f"✗ Error saving time series analysis: {e}")
    finally:
        plt.close()
    
    print("✓ Time series analysis created with real fan vote data")

# ============================================================================
# Main Analysis Function with Real Fan Vote Data
# ============================================================================

def perform_enhanced_analysis(df, output_dir):
    """Perform all enhanced visualizations with real fan vote data"""
    print("\n" + "="*80)
    print("PERFORMING ENHANCED VISUALIZATION ANALYSIS WITH REAL FAN VOTE DATA")
    print("="*80)
    
    # Create all enhanced visualizations
    create_radar_chart(df, output_dir)
    create_heatmap_matrix(df, output_dir)
    create_3d_scatter_plots(df, output_dir)
    create_time_series_analysis(df, output_dir)
    
    # Perform advanced analysis with real fan vote data
    impact_df = analyze_feature_impact(df, output_dir)
    create_judge_fan_alignment_analysis(df, output_dir)
    correlation_df = create_regression_analysis(df, output_dir)
    
    # Create summary statistics with real fan vote data
    create_summary_statistics(df, output_dir)
    
    return impact_df, correlation_df

def create_summary_statistics(df, output_dir):
    """Create comprehensive summary statistics with real fan vote data"""
    print("\nCreating summary statistics with real fan vote data...")
    
    summary_stats = {
        'Overall Statistics (With Real Fan Vote Data)': {
            'Total Contestants': len(df),
            'Total Seasons': df['season'].nunique(),
            'Average Judge Score': f"{df['avg_judge_score'].mean():.2f}",
            'Average Real Fan Vote': f"{df['Fan_Vote_Mean'].mean():.4f}",
            'Average Placement': f"{df['final_placement'].mean():.1f}",
            'Average Age': f"{df['age'].mean():.1f}",
            'Judge-Fan Correlation': f"{df['avg_judge_score'].corr(df['Fan_Vote_Mean']):.3f}",
            'Mean Judge-Fan Alignment': f"{df['judge_fan_alignment'].mean():.4f}",
            'Perfect Alignment Rate': f"{(abs(df['judge_fan_alignment']) < 0.01).sum() / len(df) * 100:.1f}%"
        },
        'Top 5 Industries by Judge Score': {},
        'Top 5 Industries by Fan Vote': {},
        'Best Judge-Fan Alignment by Industry': {},
        'Performance Category Distribution': {}
    }
    
    # Industry statistics
    industry_stats = df.groupby('industry_group').agg({
        'avg_judge_score': 'mean',
        'Fan_Vote_Mean': 'mean',
        'final_placement': 'mean',
        'judge_fan_alignment': 'mean',
        'celebrity_name': 'count'
    }).round(3)
    
    # Top industries by judge score
    top_judge = industry_stats.nlargest(5, 'avg_judge_score')
    for idx, row in top_judge.iterrows():
        summary_stats['Top 5 Industries by Judge Score'][idx] = {
            'Judge Score': row['avg_judge_score'],
            'Fan Vote': row['Fan_Vote_Mean'],
            'Avg Placement': row['final_placement'],
            'Count': int(row['celebrity_name'])
        }
    
    # Top industries by fan vote
    top_fan = industry_stats.nlargest(5, 'Fan_Vote_Mean')
    for idx, row in top_fan.iterrows():
        summary_stats['Top 5 Industries by Fan Vote'][idx] = {
            'Judge Score': row['avg_judge_score'],
            'Fan Vote': row['Fan_Vote_Mean'],
            'Avg Placement': row['final_placement'],
            'Count': int(row['celebrity_name'])
        }
    
    # Best alignment by industry (closest to 0)
    industry_stats['alignment_abs'] = abs(industry_stats['judge_fan_alignment'])
    best_alignment = industry_stats.nsmallest(5, 'alignment_abs')
    for idx, row in best_alignment.iterrows():
        summary_stats['Best Judge-Fan Alignment by Industry'][idx] = {
            'Alignment': row['judge_fan_alignment'],
            'Judge Score': row['avg_judge_score'],
            'Fan Vote': row['Fan_Vote_Mean'],
            'Count': int(row['celebrity_name'])
        }
    
    # Performance category distribution
    performance_dist = df['performance_category'].value_counts()
    for category, count in performance_dist.items():
        summary_stats['Performance Category Distribution'][category] = {
            'Count': int(count),
            'Percentage': f"{count / len(df) * 100:.1f}%"
        }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'reports', f'enhanced_summary_fanvote_{timestamp}.txt')
    try:
        with open(summary_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("ENHANCED ANALYSIS SUMMARY REPORT WITH REAL FAN VOTE DATA\n")
            f.write("="*100 + "\n\n")
            
            for section, data in summary_stats.items():
                f.write(f"{section}\n")
                f.write("-"*80 + "\n")
                
                if isinstance(data, dict):
                    if any(key in section for key in ['Statistics', 'Distribution']):
                        for key, value in data.items():
                            if isinstance(value, dict):
                                f.write(f"{key}:\n")
                                for subkey, subvalue in value.items():
                                    f.write(f"  {subkey}: {subvalue}\n")
                            else:
                                f.write(f"{key}: {value}\n")
                    else:
                        for category, stats in data.items():
                            f.write(f"{category}:\n")
                            for stat_key, stat_value in stats.items():
                                f.write(f"  {stat_key}: {stat_value}\n")
                    f.write("\n")
        print(f"✓ Summary statistics saved: {summary_path}")
    except Exception as e:
        print(f"✗ Error saving summary statistics: {e}")
    
    # Save detailed statistics to CSV
    detailed_stats = df.describe(include='all').round(3)
    detailed_stats_path = os.path.join(output_dir, 'data_exports', f'detailed_statistics_fanvote_{timestamp}.csv')
    try:
        detailed_stats.to_csv(detailed_stats_path)
        print(f"✓ Detailed statistics saved: {detailed_stats_path}")
    except Exception as e:
        print(f"✗ Error saving detailed statistics: {e}")
    
    # Save industry comparison
    industry_stats_path = os.path.join(output_dir, 'data_exports', f'industry_comparison_{timestamp}.csv')
    try:
        industry_stats.to_csv(industry_stats_path)
        print(f"✓ Industry comparison saved: {industry_stats_path}")
    except Exception as e:
        print(f"✗ Error saving industry comparison: {e}")
    
    print("✓ Summary statistics created with real fan vote data")

# ============================================================================
# Main Program
# ============================================================================

def main():
    """Main program execution with real fan vote data"""
    print("="*80)
    print("DANCING WITH THE STARS: ENHANCED VISUALIZATION SYSTEM")
    print("Comprehensive Analysis with REAL FAN VOTE DATA")
    print("="*80)
    
    # Setup
    setup_enhanced_visualization()
    output_dir = create_output_directory()
    
    # Load data with real fan vote data
    df_main, fan_votes_df = load_all_data()
    
    if df_main is None or fan_votes_df is None:
        print("Cannot load data, program exiting.")
        return
    
    # Preprocess data with real fan vote data
    print("\n" + "="*80)
    print("ENHANCED DATA PREPROCESSING WITH REAL FAN VOTE DATA")
    print("="*80)
    df_processed = preprocess_enhanced_data(df_main, fan_votes_df)
    
    # Perform enhanced analysis with real fan vote data
    impact_df, correlation_df = perform_enhanced_analysis(df_processed, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("ENHANCED ANALYSIS WITH REAL FAN VOTE DATA COMPLETE")
    print("="*80)
    print(f"All results saved in: {output_dir}/")
    print("\nGenerated Analysis:")
    print("  ✓ Feature Impact Analysis")
    print("  ✓ Judge-Fan Alignment Analysis")
    print("  ✓ Regression Correlation Analysis")
    print("  ✓ Radar Charts with Real Fan Votes")
    print("  ✓ Heatmap Matrices with Real Fan Votes")
    print("  ✓ 3D Scatter Plots with Real Fan Votes")
    print("  ✓ Time Series Analysis with Real Fan Votes")
    print("\nKey Insights Generated:")
    print("  - Industry performance comparison (Judge vs Fan)")
    print("  - Age group impact on Judge and Fan scores")
    print("  - Partner experience effect analysis")
    print("  - Performance category distribution")
    print("  - Significant feature correlations identified")
    print("="*80)

# Execute main program
if __name__ == "__main__":
    main()