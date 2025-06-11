import sqlite3
import random
import math
import numpy as np
from collections import defaultdict
from database import DB_PATH, update_bandit_arm, get_bandit_arms
from video_utils import get_available_video_count

class BanditRecommender:
    def __init__(self, epsilon=0.1, ucb_c=1.5):
        self.epsilon = epsilon  # Exploration rate for epsilon-greedy
        self.ucb_c = ucb_c      # Confidence parameter for UCB
        self.strategy = 'epsilon_greedy'  # Can be 'epsilon_greedy' or 'ucb'
        
        # Load persistent arms from database
        self.arms = get_bandit_arms()
        self.total_pulls = sum(arm.get('count', 0) for arm in self.arms.values())
        
        print(f"Bandit initialized with {len(self.arms)} arms, {self.total_pulls} total pulls")
    
    def update_arm(self, video_id, reward):
        """Update arm statistics both in memory and database"""
        # Update database (persistent storage)
        update_bandit_arm(video_id, reward)
        
        # Update in-memory cache
        if video_id not in self.arms:
            self.arms[video_id] = {'count': 1, 'avg_reward': reward}
        else:
            arm = self.arms[video_id]
            old_count = arm['count']
            new_count = old_count + 1
            
            # Update average reward using incremental formula
            old_avg = arm['avg_reward']
            new_avg = (old_avg * old_count + reward) / new_count
            
            arm['count'] = new_count
            arm['avg_reward'] = new_avg
        
        self.total_pulls += 1
        
        print(f"Updated arm {video_id}: reward={reward:.3f}, count={self.arms[video_id]['count']}, avg={self.arms[video_id]['avg_reward']:.3f}")
    
    def get_recommendations(self, user_id, n_recommendations=5):
        """Generate recommendations using multi-armed bandit strategies"""
        max_videos = get_available_video_count()
        
        # Get available videos (not yet interacted with by this user)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT videoId FROM videos 
            WHERE videoId <= ? 
            AND videoId NOT IN (
                SELECT videoId FROM interactions WHERE userId = ?
            )
        ''', (max_videos, user_id))
        
        available_videos = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not available_videos:
            # User has seen all videos, recommend best performing ones
            available_videos = list(range(1, min(max_videos + 1, n_recommendations + 1)))
        
        recommendations = []
        
        # Use different strategies
        if self.strategy == 'epsilon_greedy':
            recommendations = self._epsilon_greedy_selection(available_videos, n_recommendations)
        elif self.strategy == 'ucb':
            recommendations = self._ucb_selection(available_videos, n_recommendations)
        else:
            recommendations = self._epsilon_greedy_selection(available_videos, n_recommendations)
        
        return recommendations
    
    def _epsilon_greedy_selection(self, available_videos, n_recommendations):
        """Epsilon-greedy bandit algorithm"""
        recommendations = []
        
        for i in range(n_recommendations):
            if len(available_videos) == 0:
                break
                
            # Epsilon-greedy decision
            if random.random() < self.epsilon:
                # Exploration: choose random video
                video_id = random.choice(available_videos)
                reason = "Exploration (random)"
                score = 0.0
            else:
                # Exploitation: choose best performing video
                best_video = None
                best_reward = -float('inf')
                
                for video_id in available_videos:
                    arm_data = self.arms.get(video_id, {'count': 0, 'avg_reward': 0.0})
                    avg_reward = arm_data['avg_reward']
                    
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        best_video = video_id
                
                video_id = best_video if best_video else random.choice(available_videos)
                reason = "Exploitation (best performing)" if i == 0 else "Exploitation"
                score = self.arms.get(video_id, {'avg_reward': 0.0})['avg_reward']
            
            recommendations.append((video_id, score, reason))
            available_videos.remove(video_id)
        
        return recommendations
    
    def _ucb_selection(self, available_videos, n_recommendations):
        """Upper Confidence Bound (UCB1) bandit algorithm"""
        recommendations = []
        
        for i in range(n_recommendations):
            if len(available_videos) == 0:
                break
            
            best_video = None
            best_ucb_score = -float('inf')
            
            for video_id in available_videos:
                arm_data = self.arms.get(video_id, {'count': 0, 'avg_reward': 0.0})
                count = arm_data['count']
                avg_reward = arm_data['avg_reward']
                
                if count == 0:
                    # Unplayed arm gets infinite UCB score (exploration)
                    ucb_score = float('inf')
                else:
                    # UCB1 formula: avg_reward + c * sqrt(ln(total_pulls) / count)
                    confidence_interval = self.ucb_c * math.sqrt(
                        math.log(max(self.total_pulls, 1)) / count
                    )
                    ucb_score = avg_reward + confidence_interval
                
                if ucb_score > best_ucb_score:
                    best_ucb_score = ucb_score
                    best_video = video_id
            
            video_id = best_video if best_video else random.choice(available_videos)
            arm_data = self.arms.get(video_id, {'count': 0, 'avg_reward': 0.0})
            
            reason = "UCB exploration" if arm_data['count'] == 0 else "UCB exploitation"
            score = arm_data['avg_reward']
            
            recommendations.append((video_id, score, reason))
            available_videos.remove(video_id)
        
        return recommendations
    
    def set_strategy(self, strategy):
        """Change bandit strategy"""
        if strategy in ['epsilon_greedy', 'ucb']:
            self.strategy = strategy
            print(f"Bandit strategy changed to: {strategy}")
        else:
            print(f"Unknown strategy: {strategy}. Using epsilon_greedy.")
            self.strategy = 'epsilon_greedy'
    
    def get_arm_stats(self):
        """Get statistics for all arms"""
        stats = {}
        for video_id, data in self.arms.items():
            stats[video_id] = {
                'count': data['count'],
                'avg_reward': data['avg_reward'],
                'total_reward': data['avg_reward'] * data['count']
            }
        return stats
    
    def get_best_arms(self, n=5):
        """Get top N performing arms"""
        arm_performance = [(video_id, data['avg_reward'], data['count']) 
                          for video_id, data in self.arms.items() 
                          if data['count'] > 0]
        
        # Sort by average reward, then by count for tie-breaking
        arm_performance.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        return arm_performance[:n]
    
    def reset_arms(self):
        """Reset all arm statistics (for testing/debugging)"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM bandit_arms')
        conn.commit()
        conn.close()
        
        self.arms = {}
        self.total_pulls = 0
        print("All bandit arms reset")
    
    def calculate_regret(self, optimal_reward=1.0):
        """Calculate cumulative regret (for analysis)"""
        if self.total_pulls == 0:
            return 0.0
        
        # Calculate actual total reward
        actual_total_reward = sum(
            data['avg_reward'] * data['count'] 
            for data in self.arms.values()
        )
        
        # Calculate regret
        optimal_total_reward = optimal_reward * self.total_pulls
        regret = optimal_total_reward - actual_total_reward
        
        return max(0.0, regret)
    
    def get_model_stats(self):
        """Get comprehensive model statistics"""
        best_arms = self.get_best_arms(3)
        
        return {
            'strategy': self.strategy,
            'epsilon': self.epsilon,
            'ucb_c': self.ucb_c,
            'total_arms': len(self.arms),
            'total_pulls': self.total_pulls,
            'avg_reward': np.mean([data['avg_reward'] for data in self.arms.values()]) if self.arms else 0.0,
            'best_arms': best_arms,
            'cumulative_regret': self.calculate_regret()
        }
