import numpy as np
from scipy.optimize import linear_sum_assignment
def calculate_cle(ground_truth_pos, predicted_pos):
    """
    Calculates Center Location Error (CLE).
    :param ground_truth_pos: numpy array of shape (N, 2) or (2,)
    :param predicted_pos: numpy array of shape (N, 2) or (2,)
    :return: Mean Euclidean distance
    """
    if len(ground_truth_pos.shape) == 1:
        return np.linalg.norm(ground_truth_pos - predicted_pos)
    return np.mean(np.linalg.norm(ground_truth_pos - predicted_pos, axis=1))

def calculate_success_rate(cle_list, threshold=10.0):
    """
    Calculates Success Rate, the percentage of frames where CLE < threshold.
    """
    cle_arr = np.array(cle_list)
    return np.mean(cle_arr < threshold)

def calculate_mota(ground_truth_pos, predicted_pos, threshold=10.0):
    """
    Standard MOTA (Multi-Object Tracking Accuracy) using Hungarian matching on centroids.
    MOTA = 1 - (FN + FP) / GT
    (ID switches omitted for stateless metric).
    """
    if len(ground_truth_pos) == 0:
        return 1.0 if len(predicted_pos) == 0 else 0.0
    if len(predicted_pos) == 0:
        return 0.0

    # Pairwise distance matrix
    dist_matrix = np.linalg.norm(
        ground_truth_pos[:, np.newaxis, :] - predicted_pos[np.newaxis, :, :],
        axis=2
    )

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Valid matches (distance < threshold)
    valid_matches = dist_matrix[row_ind, col_ind] < threshold
    tp = np.sum(valid_matches)
    
    gt_count = len(ground_truth_pos)
    pred_count = len(predicted_pos)
    
    fn = gt_count - tp
    fp = pred_count - tp
    
    mota = 1.0 - (fn + fp) / gt_count
    return mota

class TrackingMetricsLogger:
    """Helper to accumulate metrics over an evaluation run."""
    def __init__(self):
        self.episode_cles = []
        self.episode_successes = []

    def add_step_info(self, info):
        if 'cle' in info:
            self.episode_cles.append(info['cle'])
            
    def get_episode_metrics(self):
        if not self.episode_cles:
            return {"mean_cle": 0.0, "success_rate": 0.0}
        
        return {
            "mean_cle": np.mean(self.episode_cles),
            "success_rate": calculate_success_rate(self.episode_cles, threshold=10.0)
        }
    
    def reset(self):
        self.episode_cles = []
        self.episode_successes = []
