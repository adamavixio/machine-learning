"""
Train DQN on 2x2 Rubik's Cube with Curriculum Learning

Matches the Zig implementation in src/main_2x2.zig
- Supervised pretraining on depth-1 states
- Curriculum learning: depth 1→2→3→4
- Episode cap: 800 episodes per depth
- Epsilon decay: 0.999 → 0.05
"""

import numpy as np
import torch
from cube_env import Cube2x2
from dqn import DoubleDQN
import time


def supervised_pretraining(agent, num_iterations=1000):
    """Pretrain on depth-1 states with known optimal actions"""
    print("\n=== Supervised Pretraining ===")
    print("Training on depth-1 states for {} iterations...".format(num_iterations))

    # Generate all 6 depth-1 states (one move from solved)
    # For each state, the optimal action is the inverse of the scramble move
    training_data = []
    for scramble_action in range(6):
        cube = Cube2x2()
        state = cube.get_state()
        cube.step(scramble_action)
        scrambled_state = cube.get_state()

        # Optimal action is the inverse
        if scramble_action % 2 == 0:
            optimal_action = scramble_action + 1
        else:
            optimal_action = scramble_action - 1

        training_data.append((scrambled_state, optimal_action))

    # Train
    for iteration in range(num_iterations):
        # Sample a training example
        state, action = training_data[np.random.randint(0, len(training_data))]

        # Store transition (reward=1.0 for solving, done=True)
        solved_state = Cube2x2().get_state()
        agent.store_transition(state, action, 1.0, solved_state, True)

        # Train
        loss = agent.train_step()

        if iteration % 100 == 0 and loss is not None:
            # Evaluate accuracy
            correct = 0
            for s, a in training_data:
                pred_action = agent.select_action(s, epsilon=0.0)
                if pred_action == a:
                    correct += 1
            accuracy = correct / len(training_data) * 100

            print(f"  Iter {iteration:4d}/{num_iterations} | Loss: {loss:.4f} | Accuracy: {accuracy:.1f}%")

            if accuracy >= 95:
                print(f"✓ Reached >95% accuracy! Stopping pretraining.")
                break

    print(f"✓ Supervised pretraining successful\n")


def train_curriculum(
    agent,
    curriculum=[(1, 800), (2, 800), (3, 800), (4, 800)],
    epsilon_start=0.999,
    epsilon_end=0.05,
    epsilon_decay=0.9995,
):
    """Train with curriculum learning"""
    print("=== RL Training with Curriculum Learning ===\n")

    epsilon = epsilon_start
    global_episode = 0

    for depth, max_episodes in curriculum:
        print(f"Starting curriculum at depth {depth}")
        depth_episode = 0
        depth_successes = 0

        while depth_episode < max_episodes:
            # Reset environment and scramble
            cube = Cube2x2()
            cube.scramble(depth=depth)
            state = cube.get_state()

            episode_reward = 0
            episode_steps = 0
            max_steps = 50

            # Episode loop
            while episode_steps < max_steps:
                # Select action
                action = agent.select_action(state, epsilon=epsilon)

                # Take step
                next_state, reward, done = cube.step(action)
                episode_reward += reward
                episode_steps += 1

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Train
                agent.train_step()

                state = next_state

                if done:
                    depth_successes += 1
                    break

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Logging
            global_episode += 1
            depth_episode += 1

            if global_episode % 100 == 0:
                success_rate = depth_successes / depth_episode * 100
                print(
                    f"Ep {global_episode:4d} | Depth {depth} ({depth_episode:4d} eps) | "
                    f"Steps: {episode_steps:2d} | Reward: {episode_reward:5.2f} | "
                    f"ε: {epsilon:.3f} | Success: {success_rate:.1f}%"
                )

        # Final stats for this depth
        final_success = depth_successes / depth_episode * 100
        print(f"✓ Depth {depth} complete: {final_success:.1f}% success\n")


def evaluate_greedy(agent, depth, num_episodes=200):
    """Evaluate policy greedily (ε=0)"""
    successes = 0
    total_moves = 0

    for _ in range(num_episodes):
        cube = Cube2x2()
        cube.scramble(depth=depth)
        state = cube.get_state()

        for moves in range(50):
            action = agent.select_action(state, epsilon=0.0)
            next_state, _, done = cube.step(action)
            state = next_state

            if done:
                successes += 1
                total_moves += moves + 1
                break

    success_rate = successes / num_episodes * 100
    avg_moves = total_moves / successes if successes > 0 else 0
    return success_rate, avg_moves


if __name__ == "__main__":
    print("=== DQN 2x2 Rubik's Cube Solver (PyTorch) ===\n")
    print("CAPACITY TEST: Wider backbone + LayerNorm")
    print("Configuration:")
    print("  Network: 144 → 256 → 128 → 64 → 6 (WITH LayerNorm)")
    print("  Batch size: 64")
    print("  Learning rate: 0.005")
    print("  Double DQN: True")
    print("  Device: CPU (for fair comparison)\n")

    # Create agent
    agent = DoubleDQN(
        state_dim=144,
        action_dim=6,
        lr=0.005,
        gamma=0.99,
        tau=0.01,
        batch_size=64,
        buffer_capacity=20000,
        device=torch.device("cpu"),
    )

    # Supervised pretraining
    start_time = time.time()
    supervised_pretraining(agent, num_iterations=1000)

    # Curriculum learning
    curriculum = [
        (1, 800),  # Depth 1: 800 episodes
        (2, 800),  # Depth 2: 800 episodes
        (3, 800),  # Depth 3: 800 episodes
    ]

    train_curriculum(agent, curriculum=curriculum)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    for depth in [1, 2, 3]:
        success_rate, avg_moves = evaluate_greedy(agent, depth, num_episodes=200)
        print(f"Depth {depth}: {success_rate:.1f}% success, {avg_moves:.1f} avg moves")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f} seconds")
    print(f"Total episodes: 2400 (800 per depth)")
    print(f"Episodes/min: {2400 / (total_time / 60):.1f}")

    # Save model
    model_path = "dqn_model.pt"
    torch.save(agent.online_net.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")
