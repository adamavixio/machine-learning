/// Rubik's Cube environment for reinforcement learning
///
/// This module provides:
/// - CubeState: 54-facelet 3x3 cube representation
/// - Cube2x2State: 24-facelet 2x2 pocket cube (more tractable)
/// - Move tables: Precomputed permutations for fast step()
/// - Helpers: isSolved, clone, hash, toOneHot for NN input
/// - Scramble: Random move generation for training

const std = @import("std");

pub const cube = @import("env/cube.zig");
pub const cube2x2 = @import("env/cube2x2.zig");
pub const gridworld = @import("env/gridworld.zig");

// Re-export common types (3x3)
pub const CubeState = cube.CubeState;
pub const Move = cube.CubeState.Move;

// Re-export 2x2 types
pub const Cube2x2State = cube2x2.Cube2x2State;
pub const Move2x2 = cube2x2.Cube2x2State.Move;

// Re-export gridworld
pub const GridWorld = gridworld.GridWorld;

test {
    std.testing.refAllDecls(@This());
}
