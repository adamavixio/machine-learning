const std = @import("std");
const ml = @import("zinc");

const TensorContext = ml.tensor.TensorContext;
const GradContext = ml.tensor.GradContext;
const AutodiffContext = ml.tensor.AutodiffContext;
const Cube2x2State = ml.env.Cube2x2State;
const Move2x2 = ml.env.Move2x2;
const DQNConfig = ml.rl.DQNConfig;

const DQNAgent = ml.rl.DQNAgent(Cube2x2State, false); // Use uniform replay
const Experience = ml.rl.Experience(Cube2x2State);

pub fn main() !void {
    std.debug.print("=== DQN 2x2 Rubik's Cube Solver ===\n\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize contexts
    var tensor_ctx = TensorContext.init(allocator);
    defer tensor_ctx.deinit();

    // Initialize RNG
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = prng.random();

    // DQN configuration (conservative per ChatGPT 20251025081557)
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
        .scramble_depth = 3,
        .use_double_dqn = true,
        .polyak_tau = 0.01,
    };

    // Network architecture: 144 (one-hot 2x2) → 256 → 128 → 64 → 6 (actions)
    // CAPACITY BOOST: Matching successful PyTorch architecture (without LayerNorm for now)
    const layer_sizes = [_]usize{ 144, 256, 128, 64, 6 };

    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Network: {d} → {d} → {d} → {d} → {d} (WIDER BACKBONE)\n", .{ layer_sizes[0], layer_sizes[1], layer_sizes[2], layer_sizes[3], layer_sizes[4] });
    std.debug.print("  Batch size: {d}\n", .{config.batch_size});
    std.debug.print("  Learning rate: {d}\n", .{config.learning_rate});
    std.debug.print("  Double DQN: {}\n\n", .{config.use_double_dqn});

    // Initialize agent (creates its own grad contexts internally)
    var agent = try DQNAgent.init(
        allocator,
        &layer_sizes,
        &tensor_ctx,
        config,
        rng,
    );
    defer agent.deinit();

    // === SUPERVISED PRETRAINING ON DEPTH-1 STATES ===
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
            std.debug.print("  Iter {d:4}/{d} | Loss: {d:.4} | Accuracy: {d:.1}%\n",
                .{ iter, pretrain_iters, loss, accuracy });

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

    // === REINFORCEMENT LEARNING TRAINING ===
    std.debug.print("=== RL Training with Curriculum Learning ===\n\n", .{});

    // Training constants
    const min_buffer_warmup: usize = 1500;
    const total_episodes: usize = 5000;
    const log_interval: usize = 100;

    // Curriculum thresholds
    const CurriculumStage = struct {
        depth: usize,
        max_episodes: usize,
        success_threshold: f32,
        window_size: usize = 200,
    };

    const curriculum = [_]CurriculumStage{
        .{ .depth = 1, .max_episodes = 800, .success_threshold = 0.85 },
        .{ .depth = 2, .max_episodes = 1200, .success_threshold = 0.70 },
        .{ .depth = 3, .max_episodes = 2000, .success_threshold = 0.55 },
        .{ .depth = 4, .max_episodes = 99999, .success_threshold = 0.0 }, // Cap at depth 4
    };

    var current_depth: usize = 4; // FIXED at depth 4
    var current_stage_idx: usize = curriculum.len - 1; // Disable curriculum advancement
    var episodes_at_current_depth: usize = 0;

    // Rolling window for success rate
    const window_size: usize = 200;
    var success_window = try allocator.alloc(bool, window_size);
    defer allocator.free(success_window);
    @memset(success_window, false);
    var window_pos: usize = 0;
    var window_count: usize = 0;

    // Statistics tracking
    var total_steps: usize = 0;
    var total_successes: usize = 0;
    var depth_successes = try allocator.alloc(usize, 5);
    defer allocator.free(depth_successes);
    var depth_episodes = try allocator.alloc(usize, 5);
    defer allocator.free(depth_episodes);
    @memset(depth_successes, 0);
    @memset(depth_episodes, 0);

    // Training diagnostics tracking
    var last_td_error_mean: f32 = 0.0;
    var last_td_error_std: f32 = 0.0;
    var last_td_error_max: f32 = 0.0;
    var last_grad_norm: f32 = 0.0;
    var last_time_total_us: f64 = 0.0;
    var last_time_sample_us: f64 = 0.0;
    var last_time_forward_us: f64 = 0.0;
    var last_time_backward_us: f64 = 0.0;
    var last_time_sgd_us: f64 = 0.0;

    // Evaluation helper: run policy greedily without training
    const EvalResult = struct { success_rate: f32, avg_moves: f32 };
    const evaluatePolicy = struct {
        fn eval(
            agent_ptr: *DQNAgent,
            depth: usize,
            num_eval: usize,
            tensor_ctx_ptr: *TensorContext,
            rng_val: std.Random,
        ) !EvalResult {
            const old_epsilon = agent_ptr.epsilon;
            agent_ptr.epsilon = 0.0; // Greedy policy
            defer agent_ptr.epsilon = old_epsilon;

            var successes: usize = 0;
            var total_moves: usize = 0;

            for (0..num_eval) |_| {
                var cube_eval = Cube2x2State.initSolved();
                cube_eval.scramble(depth, rng_val);

                var state_onehot_eval: [144]f32 = undefined;
                var moves: usize = 0;

                for (0..50) |_| { // Max 50 moves
                    cube_eval.toOneHot(&state_onehot_eval);
                    const action_eval = try agent_ptr.selectAction(&state_onehot_eval, tensor_ctx_ptr, rng_val);
                    cube_eval.step(@enumFromInt(action_eval));
                    moves += 1;

                    if (cube_eval.isSolved()) {
                        successes += 1;
                        total_moves += moves;
                        break;
                    }
                }
            }

            const success_rate = @as(f32, @floatFromInt(successes)) / @as(f32, @floatFromInt(num_eval)) * 100.0;
            const avg_moves = if (successes > 0)
                @as(f32, @floatFromInt(total_moves)) / @as(f32, @floatFromInt(successes))
            else
                0.0;

            return EvalResult{ .success_rate = success_rate, .avg_moves = avg_moves };
        }
    }.eval;

    std.debug.print("Starting curriculum at depth {d}\n", .{current_depth});
    std.debug.print("Warmup: waiting for {d} transitions before training\n\n", .{min_buffer_warmup});

    // N-step trajectory accumulator
    const Transition = struct {
        state: Cube2x2State,
        action: u8,
        reward: f32,
        next_state: Cube2x2State,
        done: bool,
    };
    var trajectory = std.ArrayList(Transition){};
    defer trajectory.deinit(allocator);

    for (0..total_episodes) |episode| {
        var cube = Cube2x2State.initSolved();
        cube.scramble(current_depth, rng);

        var state_onehot: [144]f32 = undefined;
        var episode_reward: f32 = 0.0;
        var episode_steps: usize = 0;
        var solved = false;

        // Clear trajectory at start of episode
        trajectory.clearRetainingCapacity();

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

            // Accumulate transition into trajectory
            try trajectory.append(allocator, .{
                .state = cube,
                .action = action,
                .reward = reward,
                .next_state = next_cube,
                .done = done,
            });

            // Push n-step experience when trajectory is full or episode ends
            if (trajectory.items.len >= config.n_step or done) {
                // Compute n-step return: G = r_0 + γ*r_1 + ... + γ^{n-1}*r_{n-1}
                var n_step_return: f32 = 0.0;
                var discount: f32 = 1.0;
                for (trajectory.items) |trans| {
                    n_step_return += discount * trans.reward;
                    discount *= config.gamma;
                }

                // Push experience with start state/action, n-step return, final next_state
                const first_trans = trajectory.items[0];
                const last_trans = trajectory.items[trajectory.items.len - 1];
                const n_steps_taken: u8 = @intCast(trajectory.items.len);

                agent.replay_buffer.push(Experience{
                    .state = first_trans.state,
                    .action = first_trans.action,
                    .reward = n_step_return,
                    .next_state = last_trans.next_state,
                    .done = last_trans.done,
                    .episode_id = episode,
                    .n_steps_taken = n_steps_taken,
                });

                // Clear trajectory for next n steps
                trajectory.clearRetainingCapacity();
            }

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
                last_time_sample_us = diag.time_sample_us;
                last_time_forward_us = diag.time_forward_us;
                last_time_backward_us = diag.time_backward_us;
                last_time_sgd_us = diag.time_sgd_us;

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
        depth_episodes[current_depth] += 1;
        if (solved) {
            total_successes += 1;
            depth_successes[current_depth] += 1;
            success_window[window_pos] = true;
        } else {
            success_window[window_pos] = false;
        }
        window_pos = (window_pos + 1) % window_size;
        if (window_count < window_size) window_count += 1;

        episodes_at_current_depth += 1;

        // Decay epsilon
        agent.decayEpsilon();

        // Logging every 100 episodes
        if (episode % log_interval == 0) {
            const overall_success_rate = if (episode > 0)
                @as(f32, @floatFromInt(total_successes)) / @as(f32, @floatFromInt(episode + 1)) * 100.0
            else
                0.0;

            const depth_success_rate = if (depth_episodes[current_depth] > 0)
                @as(f32, @floatFromInt(depth_successes[current_depth])) /
                @as(f32, @floatFromInt(depth_episodes[current_depth])) * 100.0
            else
                0.0;

            var window_success_count: usize = 0;
            for (success_window[0..window_count]) |s| {
                if (s) window_success_count += 1;
            }
            const rolling_success_rate = if (window_count > 0)
                @as(f32, @floatFromInt(window_success_count)) / @as(f32, @floatFromInt(window_count)) * 100.0
            else
                0.0;

            std.debug.print("Ep {d:5}/{d} | Depth {d} ({d:4} eps) | Steps: {d:2} | Reward: {d:5.2} | ε: {d:.3}\n", .{
                episode,
                total_episodes,
                current_depth,
                episodes_at_current_depth,
                episode_steps,
                episode_reward,
                agent.epsilon,
            });
            std.debug.print("  Success: Overall={d:.1}% | Depth-{d}={d:.1}% | Rolling-200={d:.1}% | Buffer: {d}/{d}\n", .{
                overall_success_rate,
                current_depth,
                depth_success_rate,
                rolling_success_rate,
                agent.replay_buffer.size,
                agent.replay_buffer.capacity,
            });

            // Log training diagnostics
            if (total_steps > 0) {
                const age_stats = agent.replay_buffer.getAgeStats();
                std.debug.print("  TD-error: mean={d:.4}, std={d:.4}, max={d:.4} | Grad norm: {d:.4}\n", .{
                    last_td_error_mean,
                    last_td_error_std,
                    last_td_error_max,
                    last_grad_norm,
                });
                std.debug.print("  Timing: total={d:.0}µs (sample={d:.0} fwd={d:.0} bwd={d:.0} sgd={d:.0})\n", .{
                    last_time_total_us,
                    last_time_sample_us,
                    last_time_forward_us,
                    last_time_backward_us,
                    last_time_sgd_us,
                });
                std.debug.print("  Replay age: episodes {d}-{d} (span={d})\n", .{
                    age_stats.min_episode,
                    age_stats.max_episode,
                    age_stats.max_episode -| age_stats.min_episode,
                });
            }
        }

        // Check curriculum progression
        if (current_stage_idx < curriculum.len - 1) {
            const stage = curriculum[current_stage_idx];

            var window_success_count: usize = 0;
            for (success_window[0..window_count]) |s| {
                if (s) window_success_count += 1;
            }
            const rolling_success_rate = if (window_count > 0)
                @as(f32, @floatFromInt(window_success_count)) / @as(f32, @floatFromInt(window_count))
            else
                0.0;

            const should_advance = (episodes_at_current_depth >= stage.max_episodes) or
                (window_count >= stage.window_size and rolling_success_rate >= stage.success_threshold);

            if (should_advance) {
                const prev_depth = current_depth;
                current_stage_idx += 1;
                current_depth = curriculum[current_stage_idx].depth;
                episodes_at_current_depth = 0;

                std.debug.print("\n🎯 CURRICULUM ADVANCE: Moving to depth {d}\n", .{current_depth});
                std.debug.print("   Previous depth-{d} success: {d:.1}%\n", .{
                    prev_depth,
                    rolling_success_rate * 100.0,
                });

                // Evaluate policy greedily on previous depth
                std.debug.print("   Evaluating policy (200 episodes, ε=0)...\n", .{});
                const eval_result = try evaluatePolicy(
                    &agent,
                    prev_depth,
                    200,
                    &tensor_ctx,
                    rng,
                );
                std.debug.print("   ✓ Depth-{d} greedy: {d:.1}% success, {d:.1} avg moves\n\n", .{
                    prev_depth,
                    eval_result.success_rate,
                    eval_result.avg_moves,
                });
            }
        }
    }

    std.debug.print("\n=== Training Complete ===\n", .{});
    std.debug.print("Total episodes: {d}\n", .{total_episodes});
    std.debug.print("Total successes: {d}\n", .{total_successes});
    std.debug.print("Total training steps: {d}\n", .{total_steps});
    std.debug.print("Final scramble depth: {d}\n", .{current_depth});
    std.debug.print("\nSuccess by depth:\n", .{});
    for (1..5) |d| {
        if (depth_episodes[d] > 0) {
            const rate = @as(f32, @floatFromInt(depth_successes[d])) /
                @as(f32, @floatFromInt(depth_episodes[d])) * 100.0;
            std.debug.print("  Depth {d}: {d}/{d} ({d:.1}%)\n", .{ d, depth_successes[d], depth_episodes[d], rate });
        }
    }
}
