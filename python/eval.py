"""
Deterministic evaluation harness for comparing DQN policies.

This script runs greedy evaluation (epsilon=0) on fixed scrambles
to provide apples-to-apples comparison between implementations.
"""

import numpy as np
import torch
from cube_env import Cube2x2
from dqn import QNetwork


def generate_deterministic_scrambles(depth: int, num_scrambles: int, seed: int = 42):
    """Generate deterministic scrambles for reproducible evaluation."""
    np.random.seed(seed)
    scrambles = []

    for _ in range(num_scrambles):
        # Generate random sequence of moves
        moves = np.random.randint(0, 6, size=depth)
        scrambles.append(moves.tolist())

    return scrambles


def evaluate_greedy(model_path: str, scrambles: list, max_steps: int = 100, verbose: bool = True):
    """
    Evaluate trained model with epsilon=0 (pure greedy) on fixed scrambles.

    Args:
        model_path: Path to saved model weights
        scrambles: List of scramble sequences (each is list of action indices)
        max_steps: Maximum steps per episode
        verbose: Print detailed results

    Returns:
        Dictionary with success rate and step statistics
    """
    # Load model
    device = torch.device("cpu")
    model = QNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = Cube2x2()
    successes = 0
    total_steps = []

    if verbose:
        print(f"\nEvaluating on {len(scrambles)} scrambles (epsilon=0, greedy only)")
        print("=" * 60)

    for scramble_idx, scramble_moves in enumerate(scrambles):
        # Reset environment
        env.reset()

        # Apply scramble
        for move in scramble_moves:
            env.step(move)

        # Get initial state
        state = env.get_state()

        # Run greedy episode (epsilon=0)
        solved = False
        steps = 0

        for step in range(max_steps):
            # Pure greedy action selection (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            # Take action
            state, reward, done = env.step(action)
            steps += 1

            if done:
                solved = True
                break

        if solved:
            successes += 1
            total_steps.append(steps)
            if verbose:
                print(f"Scramble {scramble_idx+1}/{len(scrambles)}: SOLVED in {steps} steps")
        else:
            if verbose:
                print(f"Scramble {scramble_idx+1}/{len(scrambles)}: FAILED (max steps reached)")

    success_rate = (successes / len(scrambles)) * 100
    avg_steps = np.mean(total_steps) if total_steps else 0

    if verbose:
        print("=" * 60)
        print(f"Success Rate: {success_rate:.1f}% ({successes}/{len(scrambles)})")
        if total_steps:
            print(f"Average Steps (successful): {avg_steps:.1f}")
            print(f"Min/Max Steps: {min(total_steps)}/{max(total_steps)}")

    return {
        'success_rate': success_rate,
        'successes': successes,
        'total': len(scrambles),
        'avg_steps': avg_steps,
        'all_steps': total_steps,
    }


def main():
    """Run deterministic evaluation on trained models."""

    print("Deterministic DQN Evaluation Harness")
    print("=" * 60)

    # Generate fixed scrambles for each depth
    depths = [1, 2, 3]
    num_scrambles_per_depth = 100
    seed = 42

    print(f"\nGenerating {num_scrambles_per_depth} scrambles per depth (seed={seed})")
    all_scrambles = {}
    for depth in depths:
        scrambles = generate_deterministic_scrambles(depth, num_scrambles_per_depth, seed)
        all_scrambles[depth] = scrambles
        print(f"Depth-{depth}: {len(scrambles)} scrambles")

    # Save scrambles to file for Zig implementation to use
    with open('eval_scrambles.txt', 'w') as f:
        f.write("# Deterministic scrambles for evaluation (seed=42)\n")
        f.write("# Format: depth,scramble_index,move1,move2,...\n")
        for depth, scrambles in all_scrambles.items():
            for idx, scramble in enumerate(scrambles):
                moves_str = ','.join(map(str, scramble))
                f.write(f"{depth},{idx},{moves_str}\n")

    print(f"\nSaved scrambles to eval_scrambles.txt")

    # Evaluate model (assuming we saved it during training)
    model_path = "dqn_model.pt"

    print(f"\n{'='*60}")
    print("GREEDY EVALUATION (epsilon=0)")
    print(f"{'='*60}")

    try:
        for depth in depths:
            print(f"\n--- Depth {depth} ---")
            results = evaluate_greedy(
                model_path=model_path,
                scrambles=all_scrambles[depth],
                max_steps=100,
                verbose=True
            )
    except FileNotFoundError:
        print(f"\nError: Model file '{model_path}' not found.")
        print("Please run train.py first to create the model.")
        print("\nGenerating scrambles file only...")


if __name__ == "__main__":
    main()
