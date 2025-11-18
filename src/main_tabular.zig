const std = @import("std");
const ml = @import("zinc");

const GridWorld = ml.env.GridWorld;

/// Simple tabular Q-learning for 4x4 gridworld
pub fn main() !void {
    std.debug.print("=== Tabular Q-Learning Gridworld Baseline ===\n\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize RNG
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = prng.random();

    // Q-table: 16 states × 4 actions
    const Q = try allocator.alloc(f32, 16 * 4);
    defer allocator.free(Q);
    @memset(Q, 0.0);

    // Hyperparameters
    const alpha: f32 = 0.1; // Learning rate
    const gamma: f32 = 0.99; // Discount factor
    var epsilon: f32 = 1.0; // Exploration rate
    const epsilon_decay: f32 = 0.995;
    const epsilon_min: f32 = 0.01;
    const max_steps: usize = 50;

    std.debug.print("Hyperparameters:\n", .{});
    std.debug.print("  Learning rate (α): {d}\n", .{alpha});
    std.debug.print("  Discount (γ): {d}\n", .{gamma});
    std.debug.print("  Epsilon decay: {d}\n\n", .{epsilon_decay});

    std.debug.print("Training for 1000 episodes...\n\n", .{});

    var successes: usize = 0;
    var total_episodes: usize = 0;

    for (0..1000) |episode| {
        var env = GridWorld.init();
        var episode_reward: f32 = 0.0;
        var episode_steps: usize = 0;

        for (0..max_steps) |_| {
            // Get current state index
            const state_idx = env.y * 4 + env.x;

            // Epsilon-greedy action selection
            const action: u8 = if (rng.float(f32) < epsilon) blk: {
                break :blk rng.intRangeAtMost(u8, 0, 3);
            } else blk: {
                // Greedy: select action with max Q-value
                var max_q = Q[state_idx * 4];
                var best_action: u8 = 0;
                for (1..4) |a| {
                    if (Q[state_idx * 4 + a] > max_q) {
                        max_q = Q[state_idx * 4 + a];
                        best_action = @intCast(a);
                    }
                }
                break :blk best_action;
            };

            // Execute action
            const result = env.step(action);
            episode_reward += result.reward;
            episode_steps += 1;

            // Get next state index
            const next_state_idx = env.y * 4 + env.x;

            // Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
            var max_next_q: f32 = Q[next_state_idx * 4];
            for (1..4) |a| {
                max_next_q = @max(max_next_q, Q[next_state_idx * 4 + a]);
            }

            const target = if (result.done)
                result.reward
            else
                result.reward + gamma * max_next_q;

            const q_idx = state_idx * 4 + action;
            Q[q_idx] = Q[q_idx] + alpha * (target - Q[q_idx]);

            if (result.done) {
                if (result.reward > 0) {
                    successes += 1;
                }
                total_episodes += 1;
                break;
            }
        }

        // Decay epsilon
        epsilon = @max(epsilon_min, epsilon * epsilon_decay);

        // Log progress every 50 episodes
        if (episode % 50 == 0) {
            const success_rate = if (total_episodes > 0)
                @as(f32, @floatFromInt(successes)) / @as(f32, @floatFromInt(total_episodes)) * 100.0
            else
                0.0;

            std.debug.print("Episode {d:4}/1000 | Steps: {d:2} | Reward: {d:5.2} | ε: {d:.3} | Success: {d:.1}%\n", .{
                episode,
                episode_steps,
                episode_reward,
                epsilon,
                success_rate,
            });

            // Print Q-values for start state
            const start_idx: usize = 0;
            std.debug.print("  Q(start): Up={d:.3}, Right={d:.3}, Down={d:.3}, Left={d:.3}\n", .{
                Q[start_idx * 4 + 0],
                Q[start_idx * 4 + 1],
                Q[start_idx * 4 + 2],
                Q[start_idx * 4 + 3],
            });
        }
    }

    const final_success_rate = @as(f32, @floatFromInt(successes)) / @as(f32, @floatFromInt(total_episodes)) * 100.0;
    std.debug.print("\n=== Training Complete ===\n", .{});
    std.debug.print("Total episodes: {d}\n", .{total_episodes});
    std.debug.print("Successes: {d}\n", .{successes});
    std.debug.print("Final success rate: {d:.1}%\n", .{final_success_rate});
}
