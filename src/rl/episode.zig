const std = @import("std");
const env_mod = @import("../env.zig");

const CubeState = env_mod.CubeState;
const Move = env_mod.Move;

/// Episode configuration
pub const EpisodeConfig = struct {
    max_steps: usize = 100,
    scramble_depth: usize = 5,
    reward_solved: f32 = 1.0,
    reward_step: f32 = -0.01, // Small penalty per step
    reward_timeout: f32 = -1.0, // Penalty for timeout
};

/// Episode state and management
pub const Episode = struct {
    state: CubeState,
    step_count: usize,
    total_reward: f32,
    done: bool,
    config: EpisodeConfig,

    /// Start a new episode with scrambled cube
    pub fn reset(config: EpisodeConfig, rng: std.Random) Episode {
        var state = CubeState.initSolved();
        state.scramble(config.scramble_depth, rng);

        return Episode{
            .state = state,
            .step_count = 0,
            .total_reward = 0.0,
            .done = false,
            .config = config,
        };
    }

    /// Take a step in the environment
    /// Returns: (next_state, reward, done)
    pub fn step(self: *Episode, action: u8) struct { state: CubeState, reward: f32, done: bool } {
        if (self.done) {
            // Episode already finished
            return .{
                .state = self.state,
                .reward = 0.0,
                .done = true,
            };
        }

        // Apply action
        const move: Move = @enumFromInt(action);
        self.state.step(move);
        self.step_count += 1;

        // Compute reward
        var reward: f32 = self.config.reward_step;
        var done = false;

        if (self.state.isSolved()) {
            // Solved!
            reward = self.config.reward_solved;
            done = true;
        } else if (self.step_count >= self.config.max_steps) {
            // Timeout
            reward = self.config.reward_timeout;
            done = true;
        }

        self.total_reward += reward;
        self.done = done;

        return .{
            .state = self.state.clone(),
            .reward = reward,
            .done = done,
        };
    }

    /// Check if episode is done
    pub fn isDone(self: Episode) bool {
        return self.done;
    }

    /// Get total accumulated reward
    pub fn getTotalReward(self: Episode) f32 {
        return self.total_reward;
    }

    /// Get current step count
    pub fn getStepCount(self: Episode) usize {
        return self.step_count;
    }
};

test "episode reset and step" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const config = EpisodeConfig{
        .max_steps = 10,
        .scramble_depth = 3,
    };

    var episode = Episode.reset(config, rng);

    try std.testing.expect(!episode.isDone());
    try std.testing.expectEqual(@as(usize, 0), episode.getStepCount());

    // Take a step
    const result = episode.step(0); // U move
    try std.testing.expectEqual(@as(f32, -0.01), result.reward);
    try std.testing.expectEqual(@as(usize, 1), episode.getStepCount());
}

test "episode timeout" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const config = EpisodeConfig{
        .max_steps = 2,
        .scramble_depth = 5,
    };

    var episode = Episode.reset(config, rng);

    // Take steps until timeout
    _ = episode.step(0);
    _ = episode.step(1);

    try std.testing.expect(episode.isDone());
}

test "episode solved" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const config = EpisodeConfig{
        .max_steps = 10,
        .scramble_depth = 1, // Just one move
    };

    var episode = Episode.reset(config, rng);

    // The cube was scrambled with 1 random move
    // We'd need to find the inverse move to solve it
    // For this test, just verify the mechanics work
    try std.testing.expect(!episode.state.isSolved());

    // Take steps
    for (0..5) |_| {
        if (episode.isDone()) break;
        _ = episode.step(0);
    }
}
