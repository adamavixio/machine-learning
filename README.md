# Zig Machine Learning Library

A from-scratch SIMD-accelerated machine learning library in Zig for deep reinforcement learning, with a complete DQN implementation for solving the Rubik's Cube.

## Features

### Tensor Operations (SIMD-Accelerated)
- Auto-detected SIMD lanes with scalar fallbacks
- Matrix multiplication with blocking for cache efficiency
- Element-wise operations: add, multiply, ReLU
- Reductions: sum, max, mean
- Broadcast operations
- Batched gather and axis-wise max

### Autodiff Engine
- Tape-based reverse-mode automatic differentiation
- Gradient storage and accumulation
- Tracked operations: add, mul, matmul, relu, broadcast_add
- Efficient backward pass with operation-specific kernels

### Neural Networks
- Dense layers with He initialization
- Sequential model container
- Parameter management (handles + tensors)
- Loss functions: MSE, Huber (planned)
- Optimizers: SGD (Adam planned)

### Rubik's Cube Environment
- 54-facelet state representation (6 faces × 9 stickers)
- Precomputed move tables for O(1) transitions
- 12 moves: U, U', D, D', F, F', B, B', L, L', R, R'
- One-hot encoding (324-dimensional) for neural network input
- Fast scrambling and solved-state checking

### Deep Q-Learning (DQN)
- Experience replay with ring buffer
- Online and target Q-networks
- Epsilon-greedy exploration
- Hard and soft target updates
- Complete training loop with reward shaping

## Project Structure

```
src/
├── tensor/
│   ├── config.zig       # SIMD configuration
│   ├── tensor.zig       # Tensor types and shapes
│   ├── context.zig      # Arena-based memory management
│   ├── ops.zig          # SIMD operations
│   ├── matmul.zig       # Matrix multiplication
│   ├── grad.zig         # Gradient storage
│   └── autodiff.zig     # Automatic differentiation
├── nn/
│   ├── dense.zig        # Dense layers
│   ├── model.zig        # Sequential model
│   ├── activations.zig  # Activation functions
│   ├── loss.zig         # Loss functions
│   └── optimizer.zig    # SGD optimizer
├── env/
│   └── cube.zig         # Rubik's Cube environment
├── rl/
│   ├── replay.zig       # Experience replay buffer
│   ├── qnetwork.zig     # Q-network wrapper
│   ├── dqn.zig          # DQN agent
│   └── episode.zig      # Episode management
└── main.zig             # Training script
```

## Build and Test

### Run All Tests
```bash
zig build test
```

All 32 tests should pass.

### Run Training

Run the DQN smoke test (100 episodes):
```bash
zig build run
```

This will train a DQN agent to solve scrambled Rubik's Cubes and display progress every 10 episodes.

## Training Configuration

Edit `src/main.zig` to customize training parameters:

```zig
const config = DQNConfig{
    .gamma = 0.99,                    // Discount factor
    .epsilon_start = 1.0,             // Initial exploration rate
    .epsilon_end = 0.1,               // Final exploration rate
    .epsilon_decay = 0.995,           // Epsilon decay per episode
    .learning_rate = 0.001,           // SGD learning rate
    .batch_size = 16,                 // Training batch size
    .replay_buffer_size = 500,        // Experience replay capacity
    .target_update_freq = 10,         // Target network update frequency
    .max_episode_steps = 20,          // Max steps per episode
    .scramble_depth = 3,              // Cube scramble complexity
};
```

### Network Architecture

Modify the `layer_sizes` array to change the network structure:

```zig
const layer_sizes = [_]usize{ 324, 128, 64, 12 };
//                            ^^^  ^^^  ^^  ^^
//                            |    |    |   └─ 12 possible moves
//                            |    └────┴───── Hidden layers
//                            └────────────── One-hot cube state (54×6)
```

## Example Output

```
=== DQN Rubik's Cube Solver - Smoke Test ===

Configuration:
  Network: 324 → 128 → 64 → 12
  Batch size: 16
  Replay buffer: 500
  Learning rate: 0.001
  Max steps: 20
  Scramble depth: 3

Initializing DQN agent...
Agent initialized!

Starting training for 100 episodes...

Episode   0 | Steps: 20 | Reward:  -0.20 | Avg Loss: 3.4415 | Epsilon: 0.995 | Solved: 0/1
Episode  10 | Steps: 20 | Reward:  -0.20 | Avg Loss: 6.6671 | Epsilon: 0.946 | Solved: 0/11
...
Episode  99 | Steps: 20 | Reward:  -0.20 | Avg Loss: 4.4295 | Epsilon: 0.606 | Solved: 1/100

=== Training Complete ===
Total episodes: 100
Total steps: 1981
Cubes solved: 1/100 (1.0%)
Final epsilon: 0.606
Replay buffer size: 500/500
```

## Performance

- **SIMD lanes**: 4 (auto-detected on Apple Silicon)
- **Preferred alignment**: 16 bytes
- **Memory management**: Arena allocators for minimal overhead
- **Cache optimization**: Blocked matrix multiplication

## Implementation Notes

### Memory Management

- **TensorContext**: Arena allocator for tensors. Contains model parameters (not reset during training)
- **AutodiffContext**: Tape-based gradient tracking. Reset after each training step
- **GradContext**: Gradient storage. Zeroed after each SGD update

### Design Principles

1. **Narrow API**: Focused on DQN requirements, not general-purpose ML
2. **Explicit composition**: Layers own parameters, no magic initialization
3. **Type-safe gradients**: Tracked tensors link values to gradients
4. **Precomputed tables**: Rubik's Cube moves via lookup, not computation
5. **Struct-based layers**: Clean ownership and lifetime management

## Statistics

- **Lines of Code**: ~4800
- **Files**: 20 modules
- **Tests**: 32 passing
- **Dependencies**: Zero (pure Zig stdlib)
- **Language**: Zig 0.15.1

## Future Enhancements

- [ ] Adam optimizer
- [ ] Huber loss implementation
- [ ] Double DQN (use online network for action selection)
- [ ] Prioritized experience replay
- [ ] Dueling DQN architecture
- [ ] Multi-step returns
- [ ] Curriculum learning (progressive scramble depth)
- [ ] Benchmarking and performance profiling

## License

MIT
