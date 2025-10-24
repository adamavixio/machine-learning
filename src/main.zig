const std = @import("std");
const ml = @import("machine_learning");

const TensorContext = ml.tensor.TensorContext;
const GradContext = ml.tensor.GradContext;
const AutodiffContext = ml.tensor.AutodiffContext;
const CubeState = ml.env.CubeState;
const Move = ml.env.Move;
const DQNAgent = ml.rl.DQNAgent;
const DQNConfig = ml.rl.DQNConfig;
const Experience = ml.rl.Experience;

pub fn main() !void {
    std.debug.print("=== DQN Rubik's Cube Solver - Smoke Test ===\n\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize contexts
    var tensor_ctx = TensorContext.init(allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(allocator);
    defer grad_ctx.deinit();

    var ad_ctx = AutodiffContext.init(allocator);
    defer ad_ctx.deinit();

    // Initialize RNG
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = prng.random();

    // DQN configuration (small for smoke test)
    const config = DQNConfig{
        .gamma = 0.99,
        .epsilon_start = 1.0,
        .epsilon_end = 0.1,
        .epsilon_decay = 0.995,
        .learning_rate = 0.001,
        .batch_size = 16,
        .replay_buffer_size = 500,
        .target_update_freq = 10,
        .max_episode_steps = 20,
        .scramble_depth = 3,
    };

    // Network architecture: 324 (one-hot cube) → 128 → 64 → 12 (actions)
    const layer_sizes = [_]usize{ 324, 128, 64, 12 };

    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Network: {d} → {d} → {d} → {d}\n", .{ layer_sizes[0], layer_sizes[1], layer_sizes[2], layer_sizes[3] });
    std.debug.print("  Batch size: {d}\n", .{config.batch_size});
    std.debug.print("  Replay buffer: {d}\n", .{config.replay_buffer_size});
    std.debug.print("  Learning rate: {d}\n", .{config.learning_rate});
    std.debug.print("  Max steps: {d}\n", .{config.max_episode_steps});
    std.debug.print("  Scramble depth: {d}\n\n", .{config.scramble_depth});

    // Initialize agent
    std.debug.print("Initializing DQN agent...\n", .{});
    var agent = try DQNAgent.init(
        allocator,
        &layer_sizes,
        &tensor_ctx,
        &grad_ctx,
        config,
        rng,
    );
    defer agent.deinit();
    std.debug.print("Agent initialized!\n\n", .{});

    // Training loop
    const num_episodes = 100;
    var total_steps: usize = 0;
    var solved_count: usize = 0;

    std.debug.print("Starting training for {d} episodes...\n\n", .{num_episodes});

    for (0..num_episodes) |episode| {
        // Initialize cube with scramble
        var state = CubeState.initSolved();
        state.scramble(config.scramble_depth, rng);

        var episode_reward: f32 = 0.0;
        var episode_loss: f32 = 0.0;
        var episode_steps: usize = 0;
        var loss_count: usize = 0;

        // Episode loop
        for (0..config.max_episode_steps) |step| {
            // Select action
            const action = try agent.selectAction(state, &tensor_ctx, &ad_ctx, rng);

            // Execute action
            var next_state = state;
            next_state.step(@enumFromInt(action));

            // Compute reward
            const reward: f32 = if (next_state.isSolved()) 1.0 else -0.01;
            const done = next_state.isSolved() or step >= config.max_episode_steps - 1;

            episode_reward += reward;

            // Store experience
            agent.replay_buffer.push(Experience{
                .state = state,
                .action = action,
                .reward = reward,
                .next_state = next_state,
                .done = done,
            });

            // Train if enough samples
            if (agent.replay_buffer.canSample(config.batch_size)) {
                const loss = try agent.trainStep(&tensor_ctx, &grad_ctx, &ad_ctx, rng);
                episode_loss += loss;
                loss_count += 1;
            }

            episode_steps = step + 1;
            total_steps += 1;

            // Clear autodiff tape
            ad_ctx.reset();

            if (done) {
                if (next_state.isSolved()) {
                    solved_count += 1;
                }
                break;
            }

            state = next_state;
        }

        // Decay epsilon
        agent.decayEpsilon();

        // Update target network
        if (episode > 0 and episode % config.target_update_freq == 0) {
            agent.updateTarget();
        }

        // Note: We don't reset tensor_ctx because it contains model parameters
        // The arena allocator will grow as needed and be freed at the end

        // Log progress every 10 episodes
        if (episode % 10 == 0 or episode == num_episodes - 1) {
            const avg_loss = if (loss_count > 0) episode_loss / @as(f32, @floatFromInt(loss_count)) else 0.0;
            std.debug.print(
                "Episode {d:3} | Steps: {d:2} | Reward: {d:6.2} | Avg Loss: {d:.4} | Epsilon: {d:.3} | Solved: {d}/{d}\n",
                .{ episode, episode_steps, episode_reward, avg_loss, agent.epsilon, solved_count, episode + 1 },
            );
        }
    }

    std.debug.print("\n=== Training Complete ===\n", .{});
    std.debug.print("Total episodes: {d}\n", .{num_episodes});
    std.debug.print("Total steps: {d}\n", .{total_steps});
    std.debug.print("Cubes solved: {d}/{d} ({d:.1}%)\n", .{
        solved_count,
        num_episodes,
        @as(f32, @floatFromInt(solved_count)) / @as(f32, @floatFromInt(num_episodes)) * 100.0,
    });
    std.debug.print("Final epsilon: {d:.3}\n", .{agent.epsilon});
    std.debug.print("Replay buffer size: {d}/{d}\n", .{ agent.replay_buffer.size, agent.replay_buffer.capacity });
}

test {
    std.testing.refAllDecls(@This());
}
