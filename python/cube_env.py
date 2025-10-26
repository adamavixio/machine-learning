"""
2x2 Rubik's Cube Environment

Matches the Zig implementation in src/env/cube2x2.zig
- 24 stickers (6 faces × 4 stickers each)
- 6 actions: U, U', R, R', F, F' (Up, Up-inverse, Right, Right-inverse, Front, Front-inverse)
- One-hot state representation: 144 dims (24 positions × 6 colors)
"""

import numpy as np
from typing import Tuple, List

class Cube2x2:
    """2x2x2 Rubik's Cube environment"""

    # Colors
    WHITE, YELLOW, RED, ORANGE, BLUE, GREEN = 0, 1, 2, 3, 4, 5

    # Actions
    U, U_INV, R, R_INV, F, F_INV = 0, 1, 2, 3, 4, 5
    ACTION_NAMES = ["U", "U'", "R", "R'", "F", "F'"]

    def __init__(self):
        # Initialize to solved state
        # Faces: U(top), D(bottom), F(front), B(back), L(left), R(right)
        # Each face has 4 stickers indexed 0-3 in row-major order
        self.state = np.array([
            # Up face (white)
            self.WHITE, self.WHITE, self.WHITE, self.WHITE,
            # Down face (yellow)
            self.YELLOW, self.YELLOW, self.YELLOW, self.YELLOW,
            # Front face (red)
            self.RED, self.RED, self.RED, self.RED,
            # Back face (orange)
            self.ORANGE, self.ORANGE, self.ORANGE, self.ORANGE,
            # Left face (blue)
            self.BLUE, self.BLUE, self.BLUE, self.BLUE,
            # Right face (green)
            self.GREEN, self.GREEN, self.GREEN, self.GREEN,
        ], dtype=np.int32)

    def reset(self):
        """Reset to solved state"""
        self.__init__()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get one-hot encoded state (144 dims)"""
        one_hot = np.zeros(24 * 6, dtype=np.float32)
        for i, color in enumerate(self.state):
            one_hot[i * 6 + color] = 1.0
        return one_hot

    def is_solved(self) -> bool:
        """Check if cube is in solved state"""
        for face_start in range(0, 24, 4):
            face = self.state[face_start:face_start + 4]
            if not np.all(face == face[0]):
                return False
        return True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action and return (next_state, reward, done)

        Reward: +1 if solved, -0.01 otherwise (step penalty)
        """
        # Execute move
        if action == self.U:
            self._move_u()
        elif action == self.U_INV:
            self._move_u_inv()
        elif action == self.R:
            self._move_r()
        elif action == self.R_INV:
            self._move_r_inv()
        elif action == self.F:
            self._move_f()
        elif action == self.F_INV:
            self._move_f_inv()
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check if solved
        done = self.is_solved()
        reward = 1.0 if done else -0.01

        return self.get_state(), reward, done

    def scramble(self, depth: int, seed: int = None):
        """Scramble cube with random moves"""
        if seed is not None:
            np.random.seed(seed)

        for _ in range(depth):
            action = np.random.randint(0, 6)
            self.step(action)

    # Move implementations (90-degree rotations)

    def _move_u(self):
        """Up face clockwise"""
        # Rotate Up face
        self._rotate_face_cw(0)
        # Cycle edge stickers: F→R→B→L→F
        temp = self.state[8:10].copy()  # Front top edge
        self.state[8:10] = self.state[16:18]  # Left top edge
        self.state[16:18] = self.state[12:14]  # Back top edge
        self.state[12:14] = self.state[20:22]  # Right top edge
        self.state[20:22] = temp

    def _move_u_inv(self):
        """Up face counter-clockwise (3x clockwise)"""
        for _ in range(3):
            self._move_u()

    def _move_r(self):
        """Right face clockwise"""
        # Rotate Right face
        self._rotate_face_cw(20)
        # Cycle edge stickers: U→F→D→B→U
        temp = self.state[[1, 3]].copy()  # Up right edge
        self.state[[1, 3]] = self.state[[13, 15]]  # Back right edge
        self.state[[13, 15]] = self.state[[5, 7]]  # Down right edge
        self.state[[5, 7]] = self.state[[9, 11]]  # Front right edge
        self.state[[9, 11]] = temp

    def _move_r_inv(self):
        """Right face counter-clockwise"""
        for _ in range(3):
            self._move_r()

    def _move_f(self):
        """Front face clockwise"""
        # Rotate Front face
        self._rotate_face_cw(8)
        # Cycle edge stickers: U→R→D→L→U
        temp = self.state[[2, 3]].copy()  # Up bottom edge
        self.state[[2, 3]] = self.state[[18, 16]]  # Left right edge (note reversal)
        self.state[[18, 16]] = self.state[[4, 5]]  # Down top edge
        self.state[[4, 5]] = self.state[[20, 22]]  # Right left edge
        self.state[[20, 22]] = temp

    def _move_f_inv(self):
        """Front face counter-clockwise"""
        for _ in range(3):
            self._move_f()

    def _rotate_face_cw(self, start_idx: int):
        """Rotate a face 90 degrees clockwise"""
        face = self.state[start_idx:start_idx + 4].copy()
        # 0 1    2 0
        # 2 3 -> 3 1
        self.state[start_idx] = face[2]
        self.state[start_idx + 1] = face[0]
        self.state[start_idx + 2] = face[3]
        self.state[start_idx + 3] = face[1]


if __name__ == "__main__":
    # Test cube environment
    cube = Cube2x2()
    print("Initial state (solved):", cube.is_solved())

    # Test scrambling
    cube.scramble(depth=3, seed=42)
    print("After 3 scrambles:", cube.is_solved())

    # Test one-hot encoding
    state = cube.get_state()
    print("State shape:", state.shape)
    print("State sum:", state.sum())  # Should be 24 (one-hot)
