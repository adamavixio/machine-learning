/// Rubik's Cube environment for reinforcement learning
///
/// This module provides:
/// - CubeState: 54-facelet cube representation
/// - Move tables: Precomputed permutations for fast step()
/// - Helpers: isSolved, clone, hash, toOneHot for NN input
/// - Scramble: Random move generation for training

const std = @import("std");

pub const cube = @import("env/cube.zig");

// Re-export common types
pub const CubeState = cube.CubeState;
pub const Move = cube.CubeState.Move;

test {
    std.testing.refAllDecls(@This());
}
