from mindware.blocks.alternating_block import AlternatingBlock
from mindware.blocks.conditioning_block import ConditioningBlock
from mindware.blocks.joint_block import JointBlock


# Define different execution plans
def get_execution_tree(execution_id):
    # Each node represents (parent_id, node_type)
    trees = {0: [('joint', [])],
             1: [('condition', [1]), ('joint', [])],  # Default strategy
             2: [('condition', [1]), ('alternate', [2, 3]), ('joint', []), ('joint', [])],
             3: [('alternate', [1, 2]), ('joint', []), ('joint', [])],
             4: [('alternate', [1, 2]), ('joint', []), ('condition', [3]), ('joint', [])]}
    return trees[execution_id]


def get_node_type(node_list, index):
    if node_list[index][0] == 'joint':
        root_class = JointBlock
    elif node_list[index][0] == 'condition':
        root_class = ConditioningBlock
    else:
        root_class = AlternatingBlock

    return root_class
