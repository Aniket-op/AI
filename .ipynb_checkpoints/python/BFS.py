from collections import deque

class BlockWorld:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.visited_states = set()

    def is_goal_state(self, state):
        return state == self.goal_state

    def generate_possible_moves(self, state):
        possible_moves = []
        for i in range(len(state)):
            if state[i]:  # Check if stack is not empty
                for j in range(len(state)):
                    if i != j:  # Don't move to the same stack
                        new_state = [list(stack) for stack in state]
                        block = new_state[i].pop()
                        new_state[j].append(block)
                        possible_moves.append(new_state)
        return possible_moves

    def bfs(self):
        queue = deque([(self.initial_state, [self.initial_state])])

        while queue:
            current_state, path = queue.popleft()

            if self.is_goal_state(current_state):
                return path

            self.visited_states.add(tuple(map(tuple, current_state)))

            for next_state in self.generate_possible_moves(current_state):
                if tuple(map(tuple, next_state)) not in self.visited_states:
                    new_path = path + [next_state]
                    queue.append((next_state, new_path))

        return None

    def solve(self):
        result = self.bfs()
        if result:
            print("Solution found:")
            for i, state in enumerate(result):
                print(f"Step {i+1}: {state}")
        else:
            print("No solution found.")


# Example usage:
initial_state = [['A'], ['B', 'C'], []]  # Additional empty stack
goal_state = [['A', 'B', 'C'], [], []]   # Additional empty stack
block_world = BlockWorld(initial_state, goal_state)
block_world.solve()
