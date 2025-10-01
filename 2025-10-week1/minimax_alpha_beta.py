class Node:
    def __init__(self, name, children=None, value=None):
        self.name = name
        self.children = children
        self.value = value

def is_terminal(node):
    return node.value is not None

def evaluate(node):
    return node.value

def get_children(node):
    return node.children

def alpha_beta_pruning(node, depth, alpha, beta, isMaximizing):

    if is_terminal(node) or depth==0:
        return evaluate(node)
    
    children = get_children(node)

    if isMaximizing:
        max_eval = float("-inf")
        for child_node in children:
            score = alpha_beta_pruning(child_node, depth-1, alpha, beta, False)
            alpha = max(alpha, score)
            max_eval = max(max_eval, score)

            if alpha >= beta:
                break

        return max_eval
    else:
        min_eval = float("inf")
        for child_node in children:
            score = alpha_beta_pruning(child_node, depth-1, alpha, beta, True)
            beta = min(beta, score)
            min_eval = min(min_eval, score)

            if beta <= alpha:
                break
        
        return min_eval
    

D = Node('D', value=-9)
E = Node('E', value=5)
F = Node('F', value=6)
G = Node('G', value=9)
H = Node('H', value=1)
I = Node('I', value=2)

B = Node("B", children=[I,E,F])
C = Node("C", children=[G,H,D])

A = Node("A", children=[B,C])


optimal_score = alpha_beta_pruning(A, 3, float("inf"), float("inf"), True)

print("The optimal value is", optimal_score)