const std = @import("std");

/// Rubik's Cube state representation
/// 54 facelets: 6 faces × 9 stickers each
/// Face order: U (Up), D (Down), F (Front), B (Back), L (Left), R (Right)
/// Each facelet stores its color (0-5)
pub const CubeState = struct {
    /// 54 facelets, each storing a color index (0-5)
    /// Layout: U(0-8), D(9-17), F(18-26), B(27-35), L(36-44), R(45-53)
    facelets: [54]u8,

    /// Face indices for easy access
    pub const Face = enum(u8) {
        U = 0, // Up (white)
        D = 1, // Down (yellow)
        F = 2, // Front (green)
        B = 3, // Back (blue)
        L = 4, // Left (orange)
        R = 5, // Right (red)
    };

    /// Cube moves
    pub const Move = enum(u8) {
        U, // Up clockwise
        Up, // Up counter-clockwise (U')
        D, // Down clockwise
        Dp, // Down counter-clockwise (D')
        F, // Front clockwise
        Fp, // Front counter-clockwise (F')
        B, // Back clockwise
        Bp, // Back counter-clockwise (B')
        L, // Left clockwise
        Lp, // Left counter-clockwise (L')
        R, // Right clockwise
        Rp, // Right counter-clockwise (R')

        pub fn count() usize {
            return 12;
        }
    };

    /// Create a solved cube
    pub fn initSolved() CubeState {
        var state: CubeState = undefined;

        // Each face starts with its own color
        for (0..6) |face| {
            const start = face * 9;
            for (0..9) |i| {
                state.facelets[start + i] = @intCast(face);
            }
        }

        return state;
    }

    /// Check if the cube is solved
    pub fn isSolved(self: CubeState) bool {
        for (0..6) |face| {
            const start = face * 9;
            const color = self.facelets[start];

            // All facelets on this face should have the same color
            for (1..9) |i| {
                if (self.facelets[start + i] != color) {
                    return false;
                }
            }
        }

        return true;
    }

    /// Clone the cube state
    pub fn clone(self: CubeState) CubeState {
        return CubeState{
            .facelets = self.facelets,
        };
    }

    /// Hash the cube state for deduplication in replay buffer
    pub fn hash(self: CubeState) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(&self.facelets);
        return hasher.final();
    }

    /// Apply a move to the cube
    pub fn step(self: *CubeState, move: Move) void {
        const move_table = &MOVE_TABLES[@intFromEnum(move)];

        // Create temporary copy
        const old = self.facelets;

        // Apply permutation
        for (0..54) |i| {
            self.facelets[i] = old[move_table[i]];
        }
    }

    /// Scramble the cube with random moves
    pub fn scramble(self: *CubeState, num_moves: usize, rng: std.Random) void {
        for (0..num_moves) |_| {
            const move_idx = rng.intRangeAtMost(u8, 0, 11);
            const move: Move = @enumFromInt(move_idx);
            self.step(move);
        }
    }

    /// Convert cube state to one-hot tensor input for neural network
    /// Output shape: [54 × 6] = 324 values
    /// Each facelet is one-hot encoded with 6 possible colors
    pub fn toOneHot(self: CubeState, output: []f32) void {
        std.debug.assert(output.len == 324); // 54 facelets × 6 colors

        @memset(output, 0.0);

        for (self.facelets, 0..) |color, i| {
            const offset = i * 6 + color;
            output[offset] = 1.0;
        }
    }
};

/// Precomputed move tables
/// Each table is a permutation of 0-53 indicating where each facelet moves
const MOVE_TABLES: [12][54]u8 = initMoveTables();

/// Initialize move tables at compile time
fn initMoveTables() [12][54]u8 {
    var tables: [12][54]u8 = undefined;

    // Initialize identity permutations
    for (0..12) |m| {
        for (0..54) |i| {
            tables[m][i] = @intCast(i);
        }
    }

    // U move (Up clockwise)
    // Rotate U face clockwise
    tables[0][0] = 6;
    tables[0][1] = 3;
    tables[0][2] = 0;
    tables[0][3] = 7;
    tables[0][4] = 4;
    tables[0][5] = 1;
    tables[0][6] = 8;
    tables[0][7] = 5;
    tables[0][8] = 2;
    // Rotate edge pieces
    tables[0][18] = 36; // F top → L top
    tables[0][19] = 37;
    tables[0][20] = 38;
    tables[0][45] = 18; // R top → F top
    tables[0][46] = 19;
    tables[0][47] = 20;
    tables[0][27] = 45; // B top → R top
    tables[0][28] = 46;
    tables[0][29] = 47;
    tables[0][36] = 27; // L top → B top
    tables[0][37] = 28;
    tables[0][38] = 29;

    // U' move (Up counter-clockwise) - inverse of U
    tables[1][0] = 2;
    tables[1][1] = 5;
    tables[1][2] = 8;
    tables[1][3] = 1;
    tables[1][4] = 4;
    tables[1][5] = 7;
    tables[1][6] = 0;
    tables[1][7] = 3;
    tables[1][8] = 6;
    tables[1][18] = 45;
    tables[1][19] = 46;
    tables[1][20] = 47;
    tables[1][36] = 18;
    tables[1][37] = 19;
    tables[1][38] = 20;
    tables[1][27] = 36;
    tables[1][28] = 37;
    tables[1][29] = 38;
    tables[1][45] = 27;
    tables[1][46] = 28;
    tables[1][47] = 29;

    // D move (Down clockwise)
    tables[2][9] = 15;
    tables[2][10] = 12;
    tables[2][11] = 9;
    tables[2][12] = 16;
    tables[2][13] = 13;
    tables[2][14] = 10;
    tables[2][15] = 17;
    tables[2][16] = 14;
    tables[2][17] = 11;
    tables[2][24] = 42;
    tables[2][25] = 43;
    tables[2][26] = 44;
    tables[2][42] = 33;
    tables[2][43] = 34;
    tables[2][44] = 35;
    tables[2][33] = 51;
    tables[2][34] = 52;
    tables[2][35] = 53;
    tables[2][51] = 24;
    tables[2][52] = 25;
    tables[2][53] = 26;

    // D' move (Down counter-clockwise)
    tables[3][9] = 11;
    tables[3][10] = 14;
    tables[3][11] = 17;
    tables[3][12] = 10;
    tables[3][13] = 13;
    tables[3][14] = 16;
    tables[3][15] = 9;
    tables[3][16] = 12;
    tables[3][17] = 15;
    tables[3][24] = 51;
    tables[3][25] = 52;
    tables[3][26] = 53;
    tables[3][51] = 33;
    tables[3][52] = 34;
    tables[3][53] = 35;
    tables[3][33] = 42;
    tables[3][34] = 43;
    tables[3][35] = 44;
    tables[3][42] = 24;
    tables[3][43] = 25;
    tables[3][44] = 26;

    // F move (Front clockwise)
    tables[4][18] = 24;
    tables[4][19] = 21;
    tables[4][20] = 18;
    tables[4][21] = 25;
    tables[4][22] = 22;
    tables[4][23] = 19;
    tables[4][24] = 26;
    tables[4][25] = 23;
    tables[4][26] = 20;
    tables[4][6] = 44;
    tables[4][7] = 41;
    tables[4][8] = 38;
    tables[4][38] = 9;
    tables[4][41] = 10;
    tables[4][44] = 11;
    tables[4][9] = 45;
    tables[4][10] = 48;
    tables[4][11] = 51;
    tables[4][45] = 6;
    tables[4][48] = 7;
    tables[4][51] = 8;

    // F' move (Front counter-clockwise)
    tables[5][18] = 20;
    tables[5][19] = 23;
    tables[5][20] = 26;
    tables[5][21] = 19;
    tables[5][22] = 22;
    tables[5][23] = 25;
    tables[5][24] = 18;
    tables[5][25] = 21;
    tables[5][26] = 24;
    tables[5][6] = 45;
    tables[5][7] = 48;
    tables[5][8] = 51;
    tables[5][45] = 9;
    tables[5][48] = 10;
    tables[5][51] = 11;
    tables[5][9] = 38;
    tables[5][10] = 41;
    tables[5][11] = 44;
    tables[5][38] = 6;
    tables[5][41] = 7;
    tables[5][44] = 8;

    // B move (Back clockwise)
    tables[6][27] = 33;
    tables[6][28] = 30;
    tables[6][29] = 27;
    tables[6][30] = 34;
    tables[6][31] = 31;
    tables[6][32] = 28;
    tables[6][33] = 35;
    tables[6][34] = 32;
    tables[6][35] = 29;
    tables[6][0] = 47;
    tables[6][1] = 50;
    tables[6][2] = 53;
    tables[6][47] = 17;
    tables[6][50] = 16;
    tables[6][53] = 15;
    tables[6][15] = 42;
    tables[6][16] = 39;
    tables[6][17] = 36;
    tables[6][36] = 0;
    tables[6][39] = 1;
    tables[6][42] = 2;

    // B' move (Back counter-clockwise)
    tables[7][27] = 29;
    tables[7][28] = 32;
    tables[7][29] = 35;
    tables[7][30] = 28;
    tables[7][31] = 31;
    tables[7][32] = 34;
    tables[7][33] = 27;
    tables[7][34] = 30;
    tables[7][35] = 33;
    tables[7][0] = 36;
    tables[7][1] = 39;
    tables[7][2] = 42;
    tables[7][36] = 15;
    tables[7][39] = 16;
    tables[7][42] = 17;
    tables[7][15] = 53;
    tables[7][16] = 50;
    tables[7][17] = 47;
    tables[7][47] = 0;
    tables[7][50] = 1;
    tables[7][53] = 2;

    // L move (Left clockwise)
    tables[8][36] = 42;
    tables[8][37] = 39;
    tables[8][38] = 36;
    tables[8][39] = 43;
    tables[8][40] = 40;
    tables[8][41] = 37;
    tables[8][42] = 44;
    tables[8][43] = 41;
    tables[8][44] = 38;
    tables[8][0] = 35;
    tables[8][3] = 32;
    tables[8][6] = 29;
    tables[8][18] = 0;
    tables[8][21] = 3;
    tables[8][24] = 6;
    tables[8][9] = 18;
    tables[8][12] = 21;
    tables[8][15] = 24;
    tables[8][29] = 15;
    tables[8][32] = 12;
    tables[8][35] = 9;

    // L' move (Left counter-clockwise)
    tables[9][36] = 38;
    tables[9][37] = 41;
    tables[9][38] = 44;
    tables[9][39] = 37;
    tables[9][40] = 40;
    tables[9][41] = 43;
    tables[9][42] = 36;
    tables[9][43] = 39;
    tables[9][44] = 42;
    tables[9][0] = 18;
    tables[9][3] = 21;
    tables[9][6] = 24;
    tables[9][18] = 9;
    tables[9][21] = 12;
    tables[9][24] = 15;
    tables[9][9] = 35;
    tables[9][12] = 32;
    tables[9][15] = 29;
    tables[9][29] = 6;
    tables[9][32] = 3;
    tables[9][35] = 0;

    // R move (Right clockwise)
    tables[10][45] = 51;
    tables[10][46] = 48;
    tables[10][47] = 45;
    tables[10][48] = 52;
    tables[10][49] = 49;
    tables[10][50] = 46;
    tables[10][51] = 53;
    tables[10][52] = 50;
    tables[10][53] = 47;
    tables[10][2] = 20;
    tables[10][5] = 23;
    tables[10][8] = 26;
    tables[10][20] = 11;
    tables[10][23] = 14;
    tables[10][26] = 17;
    tables[10][11] = 33;
    tables[10][14] = 30;
    tables[10][17] = 27;
    tables[10][27] = 2;
    tables[10][30] = 5;
    tables[10][33] = 8;

    // R' move (Right counter-clockwise)
    tables[11][45] = 47;
    tables[11][46] = 50;
    tables[11][47] = 53;
    tables[11][48] = 46;
    tables[11][49] = 49;
    tables[11][50] = 52;
    tables[11][51] = 45;
    tables[11][52] = 48;
    tables[11][53] = 51;
    tables[11][2] = 27;
    tables[11][5] = 30;
    tables[11][8] = 33;
    tables[11][27] = 17;
    tables[11][30] = 14;
    tables[11][33] = 11;
    tables[11][11] = 20;
    tables[11][14] = 23;
    tables[11][17] = 26;
    tables[11][20] = 2;
    tables[11][23] = 5;
    tables[11][26] = 8;

    return tables;
}

test "cube init solved" {
    const cube = CubeState.initSolved();
    try std.testing.expect(cube.isSolved());

    // Check that each face has its own color
    for (0..6) |face| {
        const start = face * 9;
        const color = cube.facelets[start];
        try std.testing.expectEqual(@as(u8, @intCast(face)), color);
    }
}

test "cube clone" {
    var cube = CubeState.initSolved();
    cube.facelets[0] = 5;

    const cloned = cube.clone();
    try std.testing.expectEqual(@as(u8, 5), cloned.facelets[0]);
    try std.testing.expectEqual(cube.facelets, cloned.facelets);
}

test "cube hash" {
    const cube1 = CubeState.initSolved();
    var cube2 = CubeState.initSolved();

    const hash1 = cube1.hash();
    const hash2 = cube2.hash();
    try std.testing.expectEqual(hash1, hash2);

    // Change one facelet
    cube2.facelets[0] = 5;
    const hash3 = cube2.hash();
    try std.testing.expect(hash1 != hash3);
}

test "cube U move and inverse" {
    var cube = CubeState.initSolved();
    const original = cube.clone();

    // Apply U move
    cube.step(.U);
    try std.testing.expect(!cube.isSolved());

    // Apply U' (inverse)
    cube.step(.Up);

    // Should return to solved state
    try std.testing.expect(cube.isSolved());
    try std.testing.expectEqual(original.facelets, cube.facelets);
}

test "cube to one-hot" {
    const cube = CubeState.initSolved();
    var one_hot: [324]f32 = undefined;

    cube.toOneHot(&one_hot);

    // Check first facelet (should be color 0)
    try std.testing.expectEqual(@as(f32, 1.0), one_hot[0]);
    for (1..6) |i| {
        try std.testing.expectEqual(@as(f32, 0.0), one_hot[i]);
    }

    // Check a middle facelet (facelet 13 should be color 1)
    const offset = 13 * 6 + 1;
    try std.testing.expectEqual(@as(f32, 1.0), one_hot[offset]);
}
