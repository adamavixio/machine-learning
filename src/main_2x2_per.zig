// EXPERIMENTAL: Prioritized Experience Replay variant
// A/B benchmark showed uniform replay (main_2x2.zig) outperformed PER by 49% (70 vs 47 solves @ 1000 eps)
// Kept for future experiments at higher scramble depths (7-9 moves)
// Use main_2x2.zig for production training

const std = @import("std");
const ml = @import("machine_learning");

const TensorContext = ml.tensor.TensorContext;
const GradContext = ml.tensor.GradContext;
const AutodiffContext = ml.tensor.AutodiffContext;
const Cube2x2State = ml.env.Cube2x2State;
const Move2x2 = ml.env.Move2x2;
const DQNConfig = ml.rl.DQNConfig;

const DQNAgent = ml.rl.DQNAgent(Cube2x2State, true); // Use PER
const Experience = ml.rl.PrioritizedExperience(Cube2x2State);

pub fn main() !void {
    std.debug.print("=== DQN 2x2 Rubik's Cube Solver (Prioritized Experience Replay) ===\n\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize contexts
    var tensor_ctx = TensorContext.init(allocator);
    defer tensor_ctx.deinit();

    // Initialize RNG
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = prng.random();

    // DQN configuration with PER enabled
    const config = DQNConfig{
        .gamma = 0.99,
        .epsilon_start = 1.0,
        .epsilon_end = 0.01,
        .epsilon_decay = 0.999,
        .learning_rate = 0.005,
        .batch_size = 64,
        .replay_buffer_size = 20000,
        .target_update_freq = 200,
        .max_episode_steps = 50,
        .scramble_depth = 5,
        .use_double_dqn = true,
        .polyak_tau = 0.01,
        // PER-specific parameters
        .use_per = true,
        .per_alpha = 0.6,
        .per_beta_start = 0.4,
        .per_beta_end = 1.0,
        .per_beta_frames = 100000,
        .per_epsilon = 1e-6,
    };

    // Network architecture: 144 (one-hot 2x2) → 256 → 128 → 64 → 6 (actions)
    const layer_sizes = [_]usize{ 144, 256, 128, 64, 6 };

    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Network: {d} → {d} → {d} → {d} → {d}\n", .{ layer_sizes[0], layer_sizes[1], layer_sizes[2], layer_sizes[3], layer_sizes[4] });
    std.debug.print("  Batch size: {d}\n", .{config.batch_size});
    std.debug.print("  Learning rate: {d}\n", .{config.learning_rate});
    std.debug.print("  Double DQN: {}\n", .{config.use_double_dqn});
    std.debug.print("  PER: ENABLED (α={d}, β={d}→{d})\n\n", .{config.per_alpha, config.per_beta_start, config.per_beta_end});

    // Initialize agent
    var agent = try DQNAgent.init(
        allocator,
        &layer_sizes,
        &tensor_ctx,
        config,
        rng,
    );
    defer agent.deinit();

    // === SUPERVISED PRETRAINING ON DEPTH-1 STATES (matching uniform baseline) ===
    std.debug.print("=== Supervised Pretraining ===\n", .{});
    std.debug.print("Generating depth-1 dataset (6 states with optimal actions)...\n", .{});

    // Generate depth-1 dataset: for each move, the optimal action is its inverse
    const depth1_moves = [_]Move2x2{ .R, .R_prime, .U, .U_prime, .F, .F_prime };
    const inverse_actions = [_]u8{ 1, 0, 3, 2, 5, 4 };

    var depth1_states = try allocator.alloc([144]f32, 6);
    defer allocator.free(depth1_states);

    for (depth1_moves, 0..) |move, i| {
        var state = Cube2x2State.initSolved();
        state.step(move);
        state.toOneHot(&depth1_states[i]);
    }

    std.debug.print("Training on depth-1 states for 1000 iterations...\n", .{});
    const pretrain_iters = 1000;
    const pretrain_batch_size = 6;

    var pretrain_states_flat = try allocator.alloc(f32, 6 * 144);
    defer allocator.free(pretrain_states_flat);

    for (0..6) |i| {
        @memcpy(pretrain_states_flat[i * 144 .. (i + 1) * 144], &depth1_states[i]);
    }

    var best_accuracy: f32 = 0.0;
    for (0..pretrain_iters) |iter| {
        const loss = try agent.pretrainStep(
            pretrain_states_flat,
            &inverse_actions,
            pretrain_batch_size,
            &tensor_ctx,
        );

        // Evaluate accuracy every 100 iterations
        if (iter % 100 == 0 or iter == pretrain_iters - 1) {
            var correct: usize = 0;
            const old_epsilon = agent.epsilon;
            agent.epsilon = 0.0;

            for (0..6) |i| {
                const action = try agent.selectAction(&depth1_states[i], &tensor_ctx, rng);
                if (action == inverse_actions[i]) {
                    correct += 1;
                }
            }

            agent.epsilon = old_epsilon;

            const accuracy = @as(f32, @floatFromInt(correct)) / 6.0 * 100.0;
            best_accuracy = @max(best_accuracy, accuracy);
            std.debug.print("  Iter {d:4}/{d} | Loss: {d:.4} | Accuracy: {d:.1}%\n", .{ iter, pretrain_iters, loss, accuracy });

            if (accuracy >= 95.0) {
                std.debug.print("✓ Reached >95% accuracy! Stopping pretraining.\n\n", .{});
                break;
            }
        }
    }

    if (best_accuracy < 95.0) {
        std.debug.print("✗ Warning: Pretraining did not reach 95% (best: {d:.1}%)\n\n", .{best_accuracy});
    } else {
        std.debug.print("✓ Supervised pretraining successful: {d:.1}% accuracy\n\n", .{best_accuracy});
    }

    // === RL TRAINING ===
    std.debug.print("=== RL Training ===\n\n", .{});

    const min_buffer_warmup: usize = 1500;
    const total_episodes: usize = 1000; // Increased per ChatGPT 20251026092411
    const log_interval: usize = 100;

    var total_steps: usize = 0;
    var total_successes: usize = 0;

    // Statistics tracking
    var last_td_error_mean: f32 = 0.0;
    var last_td_error_std: f32 = 0.0;
    var last_td_error_max: f32 = 0.0;
    var last_grad_norm: f32 = 0.0;
    var last_time_total_us: f64 = 0.0;
    var last_priority_min: f32 = 0.0;
    var last_priority_max: f32 = 0.0;
    var last_priority_mean: f32 = 0.0;
    var last_beta: f32 = 0.0;

    std.debug.print("Starting training at depth {d}\n", .{config.scramble_depth});
    std.debug.print("Warmup: waiting for {d} transitions before training\n\n", .{min_buffer_warmup});

    for (0..total_episodes) |episode| {
        var cube = Cube2x2State.initSolved();
        cube.scramble(config.scramble_depth, rng);

        // Debug: Confirm we're training at depth-5 (per ChatGPT 20251026092411)
        if (episode == 0) {
            std.debug.print("[DEPTH CHECK] Episode 0: Scrambled at depth {d}\n", .{config.scramble_depth});
        }

        var state_onehot: [144]f32 = undefined;
        var episode_reward: f32 = 0.0;
        var episode_steps: usize = 0;
        var solved = false;

        for (0..config.max_episode_steps) |step_idx| {
            cube.toOneHot(&state_onehot);

            // Select action
            const action = try agent.selectAction(&state_onehot, &tensor_ctx, rng);

            // Execute action
            var next_cube = cube;
            next_cube.step(@enumFromInt(action));

            const is_solved = next_cube.isSolved();
            const reward: f32 = if (is_solved) 1.0 else -0.01;
            const done = is_solved or step_idx >= config.max_episode_steps - 1;

            episode_reward += reward;
            episode_steps += 1;

            if (is_solved) {
                solved = true;
            }

            // Store experience (PER buffer handles priority automatically)
            agent.replay_buffer.push(Experience{
                .state = cube,
                .action = action,
                .reward = reward,
                .next_state = next_cube,
                .done = done,
                .episode_id = episode,
            });

            // Train if buffer has enough warmup
            if (agent.replay_buffer.size >= min_buffer_warmup and
                agent.replay_buffer.canSample(config.batch_size))
            {
                const diag = try agent.trainStep(&tensor_ctx, rng);
                total_steps += 1;

                // Store diagnostics for logging
                last_td_error_mean = diag.td_error_mean;
                last_td_error_std = diag.td_error_std;
                last_td_error_max = diag.td_error_max;
                last_grad_norm = diag.grad_norm;
                last_time_total_us = diag.time_total_us;
                last_priority_min = diag.priority_min;
                last_priority_max = diag.priority_max;
                last_priority_mean = diag.priority_mean;
                last_beta = diag.beta;

                // Update target network
                if (total_steps % config.target_update_freq == 0) {
                    const drift = try agent.computeTargetDrift();
                    agent.updateTarget();
                    std.debug.print("  [TARGET UPDATE] Step {d}: L2 drift = {d:.4}\n", .{ total_steps, drift });
                }
            }

            if (done) {
                break;
            }

            cube = next_cube;
        }

        // Update statistics
        if (solved) {
            total_successes += 1;
        }

        // Decay epsilon
        agent.decayEpsilon();

        // Logging
        if (episode % log_interval == 0) {
            const success_rate = if (episode > 0)
                @as(f32, @floatFromInt(total_successes)) / @as(f32, @floatFromInt(episode + 1)) * 100.0
            else
                0.0;

            std.debug.print("Ep {d:5}/{d} | Steps: {d:2} | Reward: {d:5.2} | ε: {d:.3} | Success: {d:.1}%\n", .{
                episode,
                total_episodes,
                episode_steps,
                episode_reward,
                agent.epsilon,
                success_rate,
            });

            // Log training diagnostics including PER stats
            if (total_steps > 0) {
                std.debug.print("  TD-error: mean={d:.4}, std={d:.4}, max={d:.4} | Grad: {d:.4}\n", .{
                    last_td_error_mean,
                    last_td_error_std,
                    last_td_error_max,
                    last_grad_norm,
                });
                std.debug.print("  PER: β={d:.2}, priorities=[min:{d:.2}, mean:{d:.2}, max:{d:.2}]\n", .{
                    last_beta,
                    last_priority_min,
                    last_priority_mean,
                    last_priority_max,
                });
                std.debug.print("  Timing: {d:.0}µs | Buffer: {d}/{d}\n", .{
                    last_time_total_us,
                    agent.replay_buffer.size,
                    agent.replay_buffer.capacity,
                });
            }
        }
    }

    std.debug.print("\n=== Training Complete ===\n", .{});
    std.debug.print("Total episodes: {d}\n", .{total_episodes});
    std.debug.print("Total successes: {d}\n", .{total_successes});
    std.debug.print("Total training steps: {d}\n", .{total_steps});
    std.debug.print("Final success rate: {d:.1}%\n", .{
        @as(f32, @floatFromInt(total_successes)) / @as(f32, @floatFromInt(total_episodes)) * 100.0,
    });
}
