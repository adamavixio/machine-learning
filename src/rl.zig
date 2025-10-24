/// Reinforcement Learning module for DQN on Rubik's Cube
///
/// This module provides:
/// - ReplayBuffer: Ring buffer for experience replay
/// - QNetwork: Online and target Q-networks with update utilities
/// - DQNAgent: Complete DQN agent with epsilon-greedy and training
/// - Episode: Environment interaction and reward shaping

const std = @import("std");

pub const replay = @import("rl/replay.zig");
pub const qnetwork = @import("rl/qnetwork.zig");
pub const dqn = @import("rl/dqn.zig");
pub const episode = @import("rl/episode.zig");

// Re-export common types
pub const ReplayBuffer = replay.ReplayBuffer;
pub const Experience = replay.Experience;
pub const QNetwork = qnetwork.QNetwork;
pub const DQNAgent = dqn.DQNAgent;
pub const DQNConfig = dqn.DQNConfig;
pub const Episode = episode.Episode;
pub const EpisodeConfig = episode.EpisodeConfig;

test {
    std.testing.refAllDecls(@This());
}
