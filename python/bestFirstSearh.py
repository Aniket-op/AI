from collections import deque

# Class for the Puzzle node
class PuzzleNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(str(self.state))

# Function to get possible actions and corresponding new states
def get_possible_actions(state):
    actions = []
    for i in range(len(state)):
        if state[i] == 0:
            row, col = divmod(i, 3)
            if row > 0:
                actions.append(('up', i, i - 3))
            if row < 2:
                actions.append(('down', i, i + 3))
            if col > 0:
                actions.append(('left', i, i - 1))
            if col < 2:
                actions.append(('right', i, i + 1))
            break
    return actions

# Function to perform actions on the state
def apply_action(state, action):
    state = list(state)
    _, i, j = action
    state[i], state[j] = state[j], state[i]
    return tuple(state)

# Function to perform BFS to find the solution
def bfs(start_state, goal_state):
    frontier = deque([PuzzleNode(start_state)])
    explored = set()

    while frontier:
        node = frontier.popleft()
        explored.add(node.state)

        if node.state == goal_state:
            return get_solution_path(node)

        actions = get_possible_actions(node.state)
        for action in actions:
            new_state = apply_action(node.state, action)
            if new_state not in explored:
                new_node = PuzzleNode(new_state, node, action)
                frontier.append(new_node)
                explored.add(new_state)

    return None

# Function to get the solution path
def get_solution_path(node):
    path = []
    while node.parent:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return path

# Function to print the solution path
def print_solution_path(start_state, path):
    current_state = start_state
    print("Start State:")
    print_state(current_state)
    print("Solution Steps:")
    for action in path:
        print(action[0].capitalize() + ":")
        current_state = apply_action(current_state, action)
        print_state(current_state)

# Function to print the state
def print_state(state):
    for i in range(0, 9, 3):
        print(state[i:i+3])

# Example usage
start_state = (1,2,3,8,0,4,7,6,5)  # Initial state of the puzzle
goal_state = (2,8,1,0,4,3,7,6,5)     # Goal state of the puzzle

solution = bfs(start_state, goal_state)

if solution:
    print_solution_path(start_state, solution)
else:
    print("No solution found.")
