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
- Tracked operations: add, mul, matmul, relu, broadcast_add, layer_norm, dueling_q
- Efficient backward pass with operation-specific kernels
- Dueling Q operation with proper gradient flow for advantage and value streams

### Neural Networks
- Dense layers with He initialization
- Layer normalization for training stability
- Sequential model container with optional dueling architecture
- Dueling DQN: separate value and advantage streams (Q = V + (A - mean(A)))
- Parameter management (handles + tensors)
- Loss functions: MSE, Huber
- Optimizers: SGD (Adam planned)

### Rubik's Cube Environment
- 54-facelet state representation (6 faces × 9 stickers)
- Precomputed move tables for O(1) transitions
- 12 moves: U, U', D, D', F, F', B, B', L, L', R, R'
- One-hot encoding (324-dimensional) for neural network input
- Fast scrambling and solved-state checking

### Deep Q-Learning (DQN)
- Experience replay with ring buffer (tracks n-step metadata)
- Prioritized experience replay (PER) with importance sampling
- Online and target Q-networks with dueling architecture
- N-step returns (configurable 1-5 steps) for better credit assignment
- Epsilon-greedy exploration with annealing
- Hard and soft (Polyak averaging) target updates
- Curriculum learning with progressive scramble depth
- Complete training loop with reward shaping
- Supervised pretraining for faster convergence

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

## Performance Benchmarks

### 2x2 Rubik's Cube (October 2025)

**Architecture**: Dueling DQN with 3-step returns (default configuration)
- Backbone: 144 → 256 → 128 → 64
- Value head: 64 → 1
- Advantage head: 64 → 6
- Learning rate: 0.005, batch size: 64

**Results**:
- **200 episodes**: 10.67% solve rate (depth-3 scrambles)
- **500 episodes**: 12.6% solve rate
- **1000 episodes**: 21.8% overall (17.0% depth-3, 41.0% depth-2 via curriculum)
- **2000 episodes**: 35.85% overall, **45.8% depth-4** (curriculum escalated to depth-4!)

The curriculum learning system automatically increased scramble difficulty from depth-3 to depth-4 after detecting strong performance. The agent achieved 45.8% success on depth-4 scrambles, demonstrating robust generalization.

Convergence metrics at 2000 episodes: TD-error std reduced 90% (1.66 → 0.17), gradient norm reduced 97.4% (5.87 → 0.15).

## Example Output

```
=== DQN 2x2 Rubik's Cube Solver ===

Configuration:
  Architecture: Dueling (144 → 256 → 128 → 64, value/advantage heads)
  Batch size: 64
  Replay buffer: 10000
  Learning rate: 0.005
  N-step returns: 3
  Curriculum: depth 3 → adaptive

Starting training...

Ep   100/2000 | Depth 3 ( 101 eps) | Steps: 18 | Reward:  0.64 | ε: 0.828
Ep   200/2000 | Depth 3 ( 201 eps) | Steps: 23 | Reward:  0.54 | ε: 0.686
...
Ep  1100/2000 | Depth 4 ( 301 eps) | Steps: 15 | Reward:  0.70 | ε: 0.128
Ep  2000/2000 | Depth 4 (1000 eps) | Steps: 11 | Reward:  0.78 | ε: 0.100

=== Training Complete ===
Total episodes: 2000
Success rate: 35.85% (717/2000)
Depth-4 success: 45.8% (343/749)
Curriculum depth: 4 (escalated from 3)
```

## Performance

- **BLAS acceleration**: Enabled by default on macOS via Accelerate framework (100x speedup)
- **SIMD fallback**: Auto-detected SIMD lanes with scalar fallbacks
- **Preferred alignment**: 16 bytes
- **Memory management**: Arena allocators for minimal overhead
- **Cache optimization**: Blocked matrix multiplication

### Requirements

On macOS, the Accelerate framework is required for optimal performance. It's included with macOS by default.

To disable BLAS and use SIMD-only implementation:
```bash
zig build train2x2 -Doptimize=ReleaseFast -Duse_blas=false
```

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
- [x] Huber loss implementation
- [ ] Double DQN (use online network for action selection)
- [x] Prioritized experience replay
- [x] Dueling DQN architecture
- [x] Multi-step returns (n-step TD)
- [x] Curriculum learning (progressive scramble depth)
- [x] Layer normalization
- [x] Supervised pretraining
- [ ] Tensor buffer pooling for reduced allocations
- [ ] Further performance profiling and optimization

## License

MIT
