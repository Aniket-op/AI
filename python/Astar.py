
from queue import PriorityQueue

# Calculate the number of correctly placed tiles heuristic
def correctly_placed_tiles_heuristic(current_state, goal_state):
    count = 0
    for i in range(3):
        for j in range(3):
            if current_state[i][j] != goal_state[i][j]:
                count += 1
    return count

# Swap two tiles in the puzzle
def swap_tiles(state, i1, j1, i2, j2):
    new_state = [row[:] for row in state]
    new_state[i1][j1], new_state[i2][j2] = new_state[i2][j2], new_state[i1][j1]
    return new_state

# Get possible moves for the empty tile
def get_possible_moves(state):
    moves = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                if i > 0:
                    moves.append(swap_tiles(state, i, j, i - 1, j))
                if i < 2:
                    moves.append(swap_tiles(state, i, j, i + 1, j))
                if j > 0:
                    moves.append(swap_tiles(state, i, j, i, j - 1))
                if j < 2:
                    moves.append(swap_tiles(state, i, j, i, j + 1))
    return moves



def a_star_search(initial_state, goal_state, heuristic):
    priority_queue = PriorityQueue()
    priority_queue.put((0 + heuristic(initial_state, goal_state), 0, initial_state, []))
    visited = set()

    while not priority_queue.empty():
        total_cost, current_cost, current_state, path = priority_queue.get()

        if current_state == goal_state:
            return path

        if tuple(map(tuple, current_state)) in visited:
            continue
        visited.add(tuple(map(tuple, current_state)))

        # Generate all possible moves
        moves = get_possible_moves(current_state)
        for move in moves:
            move_cost = current_cost + 1
            move_heuristic = heuristic(move, goal_state)
            total_cost = move_cost + move_heuristic
            priority_queue.put((total_cost, move_cost, move, path + [move]))

    return None  # Return None if no solution found

# Example usage:
initial_state = [[2, 0, 3], [1, 8, 4], [7, 6, 5]]
goal_state = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]

result = a_star_search(initial_state, goal_state, correctly_placed_tiles_heuristic)
# print("Moves to reach the goal state using A* search:")
# for state in result:
#     print(state)
if result:
    print("Moves to reach the goal state using A* search:")
    for state in result:
        print("State:\n",state)
else:
    print("No solution found.")
