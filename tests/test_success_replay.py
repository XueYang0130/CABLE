import unittest

import numpy as np
import torch

from agents import FullBehaviorCloningSACAgent
from methods import bc_gradient_strategy_for_method, get_method
from training.continual_experiment import (
    add_relabelled_reference_memory,
    clone_module_state,
)
from training.bc_progress_gate import BCProgressGate
from training.success_replay import (
    BroaderStateReservoir,
    SuccessfulStateReservoir,
    select_success_dominant_memory,
)


class SuccessfulStateReservoirTests(unittest.TestCase):
    def test_broader_reservoir_is_disjoint_from_success_memory(self) -> None:
        memory = BroaderStateReservoir(
            capacity_per_bin=2, observation_dim=2, seed=5
        )
        failed = tuple(
            np.asarray([float(index), -float(index)], dtype=np.float32)
            for index in range(9)
        )
        successful = tuple(
            np.asarray([100.0 + index, 100.0], dtype=np.float32)
            for index in range(9)
        )
        memory.add_episode(failed, False)
        memory.add_episode(successful, True)

        self.assertEqual(memory.non_successful_episodes, 1)
        self.assertEqual(memory.counts(), {"early": 2, "middle": 2, "late": 2})
        selected = memory.observations(
            selected_bins=("middle",), capacity=2, seed=9
        )
        self.assertEqual(selected.shape, (2, 2))
        self.assertTrue(bool((selected[:, 0] < 100.0).all()))

    def test_broader_selection_is_reproducible(self) -> None:
        memory = BroaderStateReservoir(
            capacity_per_bin=10, observation_dim=1, seed=2
        )
        episode = tuple(
            np.asarray([float(index)], dtype=np.float32) for index in range(30)
        )
        memory.add_episode(episode, False)
        first = memory.observations(capacity=7, seed=17)
        second = memory.observations(capacity=7, seed=17)
        np.testing.assert_array_equal(first, second)

    def test_ignores_failed_episodes(self) -> None:
        memory = SuccessfulStateReservoir(capacity=3, observation_dim=2, seed=7)
        memory.add_episode((np.array([1.0, 2.0], dtype=np.float32),), False)
        self.assertEqual(memory.successful_episodes, 0)
        self.assertEqual(memory.seen_successful_states, 0)
        self.assertEqual(memory.observations().shape, (0, 2))

    def test_capacity_and_seed_are_deterministic(self) -> None:
        episodes = tuple(
            np.array([float(index), float(index + 1)], dtype=np.float32)
            for index in range(20)
        )
        first = SuccessfulStateReservoir(capacity=5, observation_dim=2, seed=11)
        second = SuccessfulStateReservoir(capacity=5, observation_dim=2, seed=11)
        first.add_episode(episodes, True)
        second.add_episode(episodes, True)
        self.assertEqual(first.successful_episodes, 1)
        self.assertEqual(first.seen_successful_states, 20)
        self.assertEqual(first.observations().shape, (5, 2))
        np.testing.assert_array_equal(first.observations(), second.observations())

    def test_rejects_invalid_observation_shape(self) -> None:
        memory = SuccessfulStateReservoir(capacity=2, observation_dim=3, seed=0)
        with self.assertRaisesRegex(ValueError, "expected"):
            memory.add_episode((np.zeros(2, dtype=np.float32),), True)

    def test_dynamic_mix_uses_broad_states_to_fill_success_shortfall(self) -> None:
        successful = SuccessfulStateReservoir(
            capacity=10, observation_dim=1, seed=3
        )
        broader = BroaderStateReservoir(
            capacity_per_bin=10, observation_dim=1, seed=4
        )
        successful.add_episode(
            tuple(np.asarray([100.0 + index], dtype=np.float32) for index in range(3)),
            True,
        )
        broader.add_episode(
            tuple(np.asarray([float(index)], dtype=np.float32) for index in range(12)),
            False,
        )

        success_states, broad_states = select_success_dominant_memory(
            successful=successful,
            broader=broader,
            total_capacity=10,
            nominal_success_capacity=8,
            seed=9,
        )

        self.assertEqual(success_states.shape, (3, 1))
        self.assertEqual(broad_states.shape, (7, 1))
        self.assertTrue(bool((success_states[:, 0] >= 100.0).all()))
        self.assertTrue(bool((broad_states[:, 0] < 100.0).all()))

    def test_dynamic_mix_backfills_missing_broad_quota_with_success(self) -> None:
        successful = SuccessfulStateReservoir(
            capacity=10, observation_dim=1, seed=5
        )
        broader = BroaderStateReservoir(
            capacity_per_bin=2, observation_dim=1, seed=6
        )
        successful.add_episode(
            tuple(np.asarray([100.0 + index], dtype=np.float32) for index in range(10)),
            True,
        )

        success_states, broad_states = select_success_dominant_memory(
            successful=successful,
            broader=broader,
            total_capacity=10,
            nominal_success_capacity=8,
            seed=10,
        )

        self.assertEqual(success_states.shape, (10, 1))
        self.assertEqual(broad_states.shape, (0, 1))


class BCProgressGateTests(unittest.TestCase):
    def test_uses_recent_success_window_and_piecewise_multipliers(self) -> None:
        gate = BCProgressGate(window=3)
        self.assertEqual(gate.multiplier, 0.0)
        self.assertEqual(gate.update(0.0), 0.0)
        self.assertEqual(gate.update(0.6), 0.1)
        self.assertEqual(gate.update(0.9), 0.4)
        self.assertAlmostEqual(gate.score, 0.5)
        self.assertEqual(gate.update(1.0), 1.0)

    def test_reset_restores_early_task_gate(self) -> None:
        gate = BCProgressGate(window=1)
        self.assertEqual(gate.update(1.0), 1.0)
        gate.reset()
        self.assertEqual(gate.score, 0.0)
        self.assertEqual(gate.multiplier, 0.0)


class SuccessfulReplayMethodTests(unittest.TestCase):
    def test_final_pcgrad_differs_from_best_pcgrad_only_by_teacher(self) -> None:
        best = get_method("success_replay_best_pcgrad")
        final = get_method("success_replay_final_pcgrad")

        self.assertEqual(best.success_replay_teacher, "best")
        self.assertEqual(final.success_replay_teacher, "final")
        self.assertEqual(best.defaults, final.defaults)
        for attribute in (
            "modes",
            "append_task_id",
            "multi_head",
            "hide_task_id",
            "transfer_alpha",
            "agent_factory",
            "reference_exploration",
            "complete_reference_memory",
            "broader_replay_selector",
            "broader_replay_ratio",
            "dynamic_broader_replay_fill",
            "guide_mode",
            "replay_factory",
        ):
            self.assertEqual(getattr(best, attribute), getattr(final, attribute))








    def test_teacher_relabel_restores_live_actor(self) -> None:
        torch.manual_seed(5)
        agent = FullBehaviorCloningSACAgent(
            observation_dim=5,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=2,
            task_id_dim=2,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=2,
        )
        observations = np.zeros((3, 5), dtype=np.float32)
        observations[:, -2] = 1.0
        teacher_state = clone_module_state(agent.actor)
        teacher_means, teacher_log_stds = agent.compute_reference_targets(observations)
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.add_(0.05)
        live_state = clone_module_state(agent.actor)

        add_relabelled_reference_memory(
            agent=agent,
            observations=observations,
            teacher_actor_state=teacher_state,
        )

        for name, value in agent.actor.state_dict().items():
            torch.testing.assert_close(value.cpu(), live_state[name])
        torch.testing.assert_close(
            agent._episodic_target_means,
            teacher_means.cpu(),
        )
        torch.testing.assert_close(
            agent._episodic_target_log_stds,
            teacher_log_stds.cpu(),
        )

    def test_final_teacher_labels_memory_with_live_actor(self) -> None:
        torch.manual_seed(8)
        agent = FullBehaviorCloningSACAgent(
            observation_dim=5,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=2,
            task_id_dim=2,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=2,
        )
        observations = np.zeros((3, 5), dtype=np.float32)
        observations[:, -2] = 1.0
        expected_means, expected_log_stds = agent.compute_reference_targets(
            observations
        )
        live_state = clone_module_state(agent.actor)

        add_relabelled_reference_memory(
            agent=agent,
            observations=observations,
            teacher_actor_state=None,
        )

        for name, value in agent.actor.state_dict().items():
            torch.testing.assert_close(value.cpu(), live_state[name])
        torch.testing.assert_close(
            agent._episodic_target_means,
            expected_means.cpu(),
        )
        torch.testing.assert_close(
            agent._episodic_target_log_stds,
            expected_log_stds.cpu(),
        )


if __name__ == "__main__":
    unittest.main()
