const std = @import("std");
const ml = @import("machine_learning");

const TensorContext = ml.tensor.TensorContext;
const GradContext = ml.tensor.GradContext;
const AutodiffContext = ml.tensor.AutodiffContext;
const GridWorld = ml.env.GridWorld;
const DQNConfig = ml.rl.DQNConfig;

const DQNAgent = ml.rl.DQNAgent(GridWorld);
const Experience = ml.rl.Experience(GridWorld);

pub fn main() !void {
    std.debug.print("=== DQN Gridworld - Simplified Architecture ===\n\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize contexts
    var tensor_ctx = TensorContext.init(allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(allocator);
    defer grad_ctx.deinit();

    var ad_ctx = AutodiffContext.init(allocator, &grad_ctx);
    defer ad_ctx.deinit();

    // Initialize RNG
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = prng.random();

    // DQN configuration (optimized per ChatGPT recommendations)
    const config = DQNConfig{
        .gamma = 0.99,
        .epsilon_start = 1.0,
        .epsilon_end = 0.01,
        .epsilon_decay = 0.995,  // Faster decay to exploit sooner
        .learning_rate = 0.01,  // Higher LR now that gradients flow
        .batch_size = 32,
        .replay_buffer_size = 1024,  // Smaller buffer for fresher samples
        .target_update_freq = 50,
        .max_episode_steps = 50,
        .use_double_dqn = true,
        .polyak_tau = 0.2,  // Faster target tracking
    };

    // Network: 16 (one-hot 4x4) → 32 → 4 (actions) - Simplified single hidden layer
    const layer_sizes = [_]usize{ 16, 32, 4 };

    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Network: {d} → {d} → {d}\n", .{ layer_sizes[0], layer_sizes[1], layer_sizes[2] });
    std.debug.print("  Batch size: {d}\n", .{config.batch_size});
    std.debug.print("  Learning rate: {d}\n\n", .{config.learning_rate});

    // Initialize agent
    var agent = try DQNAgent.init(
        allocator,
        &layer_sizes,
        &tensor_ctx,
        &grad_ctx,
        config,
        rng,
    );
    defer agent.deinit();

    std.debug.print("Training for 1000 episodes...\n\n", .{});

    var total_steps: usize = 0;
    var successes: usize = 0;
    var total_episodes: usize = 0;

    for (0..1000) |episode| {
        var env = GridWorld.init();
        var state_onehot: [16]f32 = undefined;

        var episode_reward: f32 = 0.0;
        var episode_steps: usize = 0;

        for (0..config.max_episode_steps) |_| {
            // Get state
            env.toOneHot(&state_onehot);

            // Select action
            const action = try agent.selectAction(&state_onehot, &tensor_ctx, &ad_ctx, rng);

            // Execute action
            const result = env.step(@intCast(action));
            var next_state_onehot: [16]f32 = undefined;
            env.toOneHot(&next_state_onehot);

            episode_reward += result.reward;
            episode_steps += 1;

            // Store experience (construct old state for replay)
            var old_env = GridWorld.init();
            old_env.toOneHot(&state_onehot);
            const next_env = env;

            agent.replay_buffer.push(Experience{
                .state = old_env,
                .action = action,
                .reward = result.reward,
                .next_state = next_env,
                .done = result.done,
            });

            // Train (require 256 samples warmup, 4 updates per step)
            const min_buffer_size: usize = 256;
            if (agent.replay_buffer.canSample(config.batch_size) and
                agent.replay_buffer.size >= min_buffer_size)
            {
                // Multiple gradient updates per env step
                for (0..4) |_| {
                    const loss = try agent.trainStep(&tensor_ctx, &grad_ctx, &ad_ctx, rng);
                    total_steps += 1;
                    _ = loss;
                }

                // Update target network and log drift
                if (total_steps % config.target_update_freq == 0) {
                    // Compute L2 distance before update
                    const online_params = try agent.qnet.online_model.getParameterTensors(allocator);
                    defer allocator.free(online_params);
                    const target_params = try agent.qnet.target_model.getParameterTensors(allocator);
                    defer allocator.free(target_params);

                    var l2_dist: f32 = 0.0;
                    for (online_params, target_params) |op, tp| {
                        for (op.data, tp.data) |ov, tv| {
                            const diff = ov - tv;
                            l2_dist += diff * diff;
                        }
                    }
                    l2_dist = @sqrt(l2_dist);

                    agent.updateTarget();

                    std.debug.print("[TARGET UPDATE] Step {d}: L2 drift = {d:.4}\n", .{total_steps, l2_dist});
                }
            }

            ad_ctx.reset();

            if (result.done) {
                if (result.reward > 0) {
                    successes += 1;
                }
                total_episodes += 1;
                break;
            }
        }

        // Decay epsilon
        agent.decayEpsilon();

        // Log progress every 100 episodes
        if (episode % 100 == 0) {
            const success_rate = if (total_episodes > 0)
                @as(f32, @floatFromInt(successes)) / @as(f32, @floatFromInt(total_episodes)) * 100.0
            else
                0.0;

            std.debug.print("Episode {d:4}/1000 | Steps: {d:2} | Reward: {d:5.2} | ε: {d:.3} | Success: {d:.1}%\n", .{ episode, episode_steps, episode_reward, agent.epsilon, success_rate });

            // Log Q-value stats for reference state (start position 0,0)
            var start_state: [16]f32 = undefined;
            var start_env = GridWorld.init();
            start_env.toOneHot(&start_state);

            const old_eps = agent.epsilon;
            agent.epsilon = 0.0;  // Disable exploration for Q-value check

            // Get parameter norm to verify network is updating
            const params_check = try agent.qnet.online_model.getParameterTensors(allocator);
            defer allocator.free(params_check);
            var param_norm: f32 = 0.0;
            for (params_check[0].data) |w| {
                param_norm += w * w;
            }
            param_norm = @sqrt(param_norm);

            const states_tensor = try tensor_ctx.allocTensor(&[_]usize{ 1, 16 });
            @memcpy(states_tensor.data, &start_state);
            const states_tracked = try ad_ctx.track(states_tensor);
            const q_vals = try agent.qnet.online_model.forward(states_tracked, &ad_ctx, &tensor_ctx);

            var q_min = q_vals.data()[0];
            var q_max = q_vals.data()[0];
            var q_sum: f32 = 0.0;
            for (q_vals.data()[0..4]) |q| {
                q_min = @min(q_min, q);
                q_max = @max(q_max, q);
                q_sum += q;
            }
            const q_mean = q_sum / 4.0;

            std.debug.print("  [Q-VALUES] Start state: mean={d:.3}, min={d:.3}, max={d:.3}, param_norm={d:.6}\n", .{q_mean, q_min, q_max, param_norm});

            agent.epsilon = old_eps;
            ad_ctx.reset();
        }
    }

    const final_success_rate = @as(f32, @floatFromInt(successes)) / @as(f32, @floatFromInt(total_episodes)) * 100.0;
    std.debug.print("\n=== Training Complete ===\n", .{});
    std.debug.print("Total episodes: {d}\n", .{total_episodes});
    std.debug.print("Successes: {d}\n", .{successes});
    std.debug.print("Success rate: {d:.1}%\n", .{final_success_rate});
    std.debug.print("Total training steps: {d}\n", .{total_steps});
}
