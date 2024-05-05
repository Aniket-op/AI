import heapq

class Node:
    def __init__(self, state, cost, parent=None):
        self.state = state
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def uniform_cost_search(initial_state, goal_state, successors, heuristic=None):
    visited = set()
    priority_queue = []
    heapq.heappush(priority_queue, Node(initial_state, 0))

    while priority_queue:
        current_node = heapq.heappop(priority_queue)

        if current_node.state == goal_state:
            return current_node

        visited.add(current_node.state)

        for successor, cost in successors(current_node.state):
            if successor not in visited:
                total_cost = current_node.cost + cost
                heapq.heappush(priority_queue, Node(successor, total_cost, current_node))

    return None

# Example usage
def successors(state):
    # Define the successors for each state along with their costs
    # This is just a simple example, adjust according to your problem
    successors_dict = {
        'S': [('A', 1), ('B', 5), ('C', 15)],
        'A': [('S', 1), ('G', 10)],
        'B': [('S', 5), ('G', 5)],
        'C': [('S', 15), ('G', 5)],
        'G': [('A', 10), ('B', 5), ('C', 5)]
    }
    return successors_dict.get(state, [])

initial_state = 'S'
goal_state = 'G'

result_node = uniform_cost_search(initial_state, goal_state, successors)
if result_node:
    # Trace back to get the path
    path = []
    while result_node:
        path.append(result_node.state)
        result_node = result_node.parent
    path.reverse()
    print("Path found:", path)
else:
    print("No path found")
