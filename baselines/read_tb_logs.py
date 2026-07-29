import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_final_metrics(log_dir):
    # Find the events file
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                path = os.path.join(root, file)
                ea = EventAccumulator(path)
                ea.Reload()
                
                try:
                    rew = ea.Scalars("rollout/ep_rew_mean")[-1].value
                    length = ea.Scalars("rollout/ep_len_mean")[-1].value
                    return rew, length
                except:
                    pass
    return None, None

print("Sparse Reward Logs:")
rew_sparse, len_sparse = get_final_metrics("./results/logs_reward_diagnostics_adv/sparse")
print(f"  Ep Rew Mean: {rew_sparse}, Ep Len Mean: {len_sparse}")

print("Hackable Shaped Reward Logs:")
rew_shaped, len_shaped = get_final_metrics("./results/logs_reward_diagnostics_adv/hackable_shaped")
print(f"  Ep Rew Mean: {rew_shaped}, Ep Len Mean: {len_shaped}")
