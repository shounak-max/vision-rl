import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

try:
    import gymnasium as gym
    from envs.wrappers import (
        BlurWrapper,
        ColorJitterWrapper,
        CompoundShiftWrapper,
        CutoutWrapper,
        DataAugmentationWrapper,
        DistractorWrapper,
        NoiseWrapper,
        OcclusionWrapper,
        ProcgenGymnasiumWrapper,
        RandomShiftWrapper,
        SparseRewardStepWrapper,
        ViewpointWrapper,
    )
except ImportError:
    gym = None


if gym is not None:
    class ImageEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                0, 255, shape=(3, 32, 32), dtype=np.uint8
            )
            self.action_space = gym.spaces.Discrete(2)
            self.max_steps = 1
            self.step_count = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            return np.arange(3 * 32 * 32, dtype=np.uint8).reshape(3, 32, 32), {}

        def step(self, action):
            self.step_count += 1
            observation = np.arange(
                3 * 32 * 32, dtype=np.uint8
            ).reshape(3, 32, 32)
            return observation, 0.0, False, self.step_count >= 1, {"cle": 1.0}


@unittest.skipIf(gym is None, "Gymnasium and wrapper dependencies are unavailable")
class ObservationWrapperTests(unittest.TestCase):
    def test_observation_wrappers_reproduce_outputs_from_reset_seed(self):
        wrapper_factories = (
            lambda env: NoiseWrapper(env),
            lambda env: DistractorWrapper(env),
            lambda env: ViewpointWrapper(env),
            lambda env: RandomShiftWrapper(env),
            lambda env: ColorJitterWrapper(env),
            lambda env: CutoutWrapper(env),
            lambda env: DataAugmentationWrapper(env),
            lambda env: OcclusionWrapper(env),
            lambda env: BlurWrapper(env),
            lambda env: CompoundShiftWrapper(env),
        )
        for make_wrapper in wrapper_factories:
            with self.subTest(wrapper=make_wrapper):
                first_wrapper = make_wrapper(ImageEnv())
                second_wrapper = make_wrapper(ImageEnv())
                first, _ = first_wrapper.reset(seed=91)
                second, _ = second_wrapper.reset(seed=91)
                self.assertEqual(first.shape, (3, 32, 32))
                self.assertEqual(first.dtype, np.uint8)
                np.testing.assert_array_equal(first, second)
                first_next = first_wrapper.step(0)[0]
                second_next = second_wrapper.step(0)[0]
                np.testing.assert_array_equal(first_next, second_next)

    def test_tracking_wrappers_preserve_shape_and_reproduce_one_step(self):
        import envs.tracking_envs

        wrapper_factories = (
            lambda env: NoiseWrapper(env),
            lambda env: DistractorWrapper(env),
            lambda env: ViewpointWrapper(env),
            lambda env: RandomShiftWrapper(env),
            lambda env: ColorJitterWrapper(env),
            lambda env: CutoutWrapper(env),
            lambda env: DataAugmentationWrapper(env),
            lambda env: OcclusionWrapper(env),
            lambda env: BlurWrapper(env),
            lambda env: CompoundShiftWrapper(env),
        )
        for make_wrapper in wrapper_factories:
            with self.subTest(wrapper=make_wrapper):
                trajectories = []
                for _ in range(2):
                    wrapped = make_wrapper(gym.make("SingleObjectTracking-v0"))
                    initial, _ = wrapped.reset(seed=1234)
                    next_observation, *_ = wrapped.step(
                        np.zeros(2, dtype=np.float32)
                    )
                    wrapped.close()
                    self.assertEqual(initial.shape, (3, 84, 84))
                    self.assertEqual(next_observation.shape, (3, 84, 84))
                    self.assertEqual(initial.dtype, np.uint8)
                    self.assertEqual(next_observation.dtype, np.uint8)
                    trajectories.append((initial, next_observation))
                np.testing.assert_array_equal(trajectories[0][0], trajectories[1][0])
                np.testing.assert_array_equal(trajectories[0][1], trajectories[1][1])

    def test_viewpoint_transform_is_fixed_within_episode(self):
        wrapper = ViewpointWrapper(ImageEnv(), max_angle=25)
        wrapper.reset(seed=19)
        observation = np.arange(3 * 32 * 32, dtype=np.uint8).reshape(3, 32, 32)
        first = wrapper.observation(observation)
        second = wrapper.observation(observation)
        np.testing.assert_array_equal(first, second)

    def test_sparse_reward_step_wrapper_returns_terminal_reward(self):
        wrapper = SparseRewardStepWrapper(ImageEnv(), success_threshold=10.0)
        wrapper.reset(seed=4)
        _, reward, terminated, truncated, info = wrapper.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["sparse_reward"], 1.0)

    def test_procgen_adapter_uses_nonoverlapping_level_ranges_and_seed(self):
        created = []

        class FakeLegacyEnv:
            observation_space = types.SimpleNamespace(
                shape=(64, 64, 3), dtype=np.uint8
            )
            action_space = types.SimpleNamespace(n=15)

            def __init__(self, kwargs):
                self.kwargs = kwargs
                self.closed = False

            def reset(self):
                return np.zeros((64, 64, 3), dtype=np.uint8)

            def step(self, action):
                return (
                    np.zeros((64, 64, 3), dtype=np.uint8),
                    1.0,
                    True,
                    {"prev_level_complete": 1},
                )

            def close(self):
                self.closed = True

        def fake_make(env_id, **kwargs):
            created.append(kwargs)
            return FakeLegacyEnv(kwargs)

        with patch.dict(sys.modules, {"gym": types.SimpleNamespace(make=fake_make)}):
            training = ProcgenGymnasiumWrapper(split="train")
            training.reset(seed=37)
            _, _, train_terminated, train_truncated, _ = training.step(0)
            evaluation = ProcgenGymnasiumWrapper(split="evaluation")
            evaluation.reset(seed=37)
            training.close()
            evaluation.close()
        self.assertTrue(train_terminated)
        self.assertFalse(train_truncated)

        train_kwargs, seeded_train_kwargs, eval_kwargs, seeded_eval_kwargs = created
        self.assertEqual(
            (train_kwargs["start_level"], train_kwargs["num_levels"]),
            (0, 500),
        )
        self.assertEqual(
            (eval_kwargs["start_level"], eval_kwargs["num_levels"]),
            (10000, 500),
        )
        self.assertEqual(
            seeded_train_kwargs["rand_seed"], 37
        )
        self.assertEqual(
            seeded_eval_kwargs["rand_seed"], 37
        )
        self.assertTrue(all(
            kwargs["use_sequential_levels"] for kwargs in created
        ))


if __name__ == "__main__":
    unittest.main()
