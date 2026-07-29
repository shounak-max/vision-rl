from gymnasium.envs.registration import register

register(
    id='SingleObjectTracking-v0',
    entry_point='envs.tracking_envs:SingleObjectTrackingEnv',
)

register(
    id='MultiObjectTracking-v0',
    entry_point='envs.tracking_envs:MultiObjectTrackingEnv',
)

register(
    id='ActiveTracking-v0',
    entry_point='envs.tracking_envs:ActiveTrackingEnv',
)

register(
    id='MultiStageNavigation-v0',
    entry_point='envs.navigation_envs:MultiStageNavigationEnv',
)
