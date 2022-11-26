import copy
import numpy as np


class Node:
    def __init__(self, weights, layer, neuron):
        self._weights = weights
        self._layer = layer
        self._neuron = neuron

    def add_left_child(self, left):
        self._left = left

    def add_right_child(self, right):
        self._right = right


class LabelNode:
    def __init__(self, label):
        self._label = label

    def label(self):
        return self._label


class ObliqueTreeC:

    def get_node(self, depth):
        for i in range(len(self._dims[1:])):
            if depth - np.sum(self._dims[1:(i+2)]) >= 0:
                continue
            else:
                return int(depth - np.sum(self._dims[1:(i+1)])), i+1

    def get_str(self, node, level=0, digits=2):
        if isinstance(node, LabelNode):
            self._str += ('|   ' * level + "|--- class " + str(node.label()) + '\n')
        elif node is not None:
            neuron, layer = self.get_node(level)
            self._str += ('|   ' * level + "|--- " + str(np.around(node._weights[f'OB{layer}'][neuron][0], digits)) + " + " + str(np.around(node._weights[f'OW{layer}'][neuron], digits)) + " * x <= 0\n")
            self.get_str(node._left, level + 1)
            self._str += ('|   ' * level + "|--- " + str(np.around(node._weights[f'OB{layer}'][neuron][0], digits)) + " + " + str(np.around(node._weights[f'OW{layer}'][neuron], digits)) + " * x > 0\n")
            self.get_str(node._right, level + 1)

    def __str__(self):
        self._str = ""
        self.get_str(self._root)
        return self._str

    def build_tree(self, root):

        if root._layer > 1:
            root._weights[f'OW{root._layer}'][root._neuron - 1] = np.dot(root._weights[f'W{root._layer}'][root._neuron - 1], root._weights[f'OW{root._layer - 1}'])
            root._weights[f'OB{root._layer}'][root._neuron - 1] = np.dot(root._weights[f'W{root._layer}'][root._neuron - 1], root._weights[f'OB{root._layer - 1}']) + root._weights[f'B{root._layer}'][root._neuron - 1]

        if root._layer == len(self._dims) - 1 and root._neuron == self._dims[-1]:
            label_left = LabelNode(0)
            label_right = LabelNode(1)

            root.add_left_child(label_left)
            root.add_right_child(label_right)
            return

        left_weights = copy.deepcopy(root._weights)
        right_weights = copy.deepcopy(root._weights)

        left_weights[f'OW{root._layer}'][root._neuron - 1] = 0
        left_weights[f'OB{root._layer}'][root._neuron - 1] = 0

        if root._neuron == self._dims[root._layer]:
            left_child = Node(left_weights, root._layer + 1, 1)
            right_child = Node(right_weights, root._layer + 1, 1)
        else:
            left_child = Node(left_weights, root._layer, root._neuron + 1)
            right_child = Node(right_weights, root._layer, root._neuron + 1)

        root.add_left_child(left_child)
        root.add_right_child(right_child)

        self.build_tree(left_child)
        self.build_tree(right_child)

    def induce_oblique_tree(self, weights, dims):
        self._dims = dims

        for i in range(1, len(dims)):
            weights[f'OW{i}'] = np.zeros((dims[i], dims[0]))
            weights[f'OB{i}'] = np.zeros((dims[i], 1))

        weights[f'OW{1}'] = copy.deepcopy(weights[f'W{1}'])
        weights[f'OB{1}'] = copy.deepcopy(weights[f'B{1}'])

        self._root = Node(weights, 1, 1)
        self.build_tree(self._root)

    def classify_instance(self, root, x):
        if isinstance(root, LabelNode):
            return root.label()

        value_node = np.dot(root._weights[f'OW{root._layer}'][root._neuron - 1], x) + root._weights[f'OB{root._layer}'][root._neuron - 1]

        if value_node < 0:
            label = self.classify_instance(root._left, x)
        else:
            label = self.classify_instance(root._right, x)

        return label

    def classify(self, x):
        return self.classify_instance(self._root, x)


class ObliqueTreeR:

    def get_node(self, depth):
        for i in range(len(self._dims[1:])):
            if depth - np.sum(self._dims[1:(i+2)]) >= 0:
                continue
            else:
                return int(depth - np.sum(self._dims[1:(i+1)])), i+1

    def get_str(self, node, level=0, digits=2):
        if level == np.sum(self._dims[1:-1]):
            for a in range(self._dims[-1]):
                self._str += ('|  ' * level + "|--- " + 'a' + str(a) + ": " + str(np.round(node._weights[f'OB{len(self._dims)-1}'][a][0], digits)) + " + " + str(np.round(node._weights[f'OW{len(self._dims)-1}'][a], digits)) + " * x \n")
        elif node is not None:
            neuron, layer = self.get_node(level)
            self._str += ('|  ' * level + "|--- " + str(np.round(node._weights[f'OB{layer}'][neuron][0], digits)) + " + " + str(np.round(node._weights[f'OW{layer}'][neuron], digits)) + " * x <= 0\n")
            self.get_str(node._left, level + 1)
            self._str += ('|  ' * level + "|--- " + str(np.round(node._weights[f'OB{layer}'][neuron][0], digits)) + " + " + str(np.round(node._weights[f'OW{layer}'][neuron], digits)) + " * x > 0\n")
            self.get_str(node._right, level + 1)

    def __str__(self):
        self._str = ""
        self.get_str(self._root)
        return self._str

    def build_tree(self, root):

        # Output layer
        if root._layer == len(self._dims) - 1:
            for n in range(self._dims[-1]):
                root._weights[f'OW{root._layer}'][n] = np.dot(root._weights[f'W{root._layer}'][n], root._weights[f'OW{root._layer - 1}'])
                root._weights[f'OB{root._layer}'][n] = np.dot(root._weights[f'W{root._layer}'][n], root._weights[f'OB{root._layer - 1}']) + root._weights[f'B{root._layer}'][n]
            return

        # Hidden layers
        if root._layer > 1:
            root._weights[f'OW{root._layer}'] = np.dot(root._weights[f'W{root._layer}'][root._neuron - 1], root._weights[f'OW{root._layer - 1}'])
            root._weights[f'OB{root._layer}'] = np.dot(root._weights[f'W{root._layer}'][root._neuron - 1], root._weights[f'OB{root._layer - 1}']) + root._weights[f'B{root._layer}']

        left_weights = copy.deepcopy(root._weights)
        right_weights = copy.deepcopy(root._weights)

        left_weights[f'OW{root._layer}'][root._neuron - 1] = 0
        left_weights[f'OB{root._layer}'][root._neuron - 1] = 0

        if root._neuron == self._dims[root._layer]:
            left_child = Node(left_weights, root._layer + 1, 1)
            right_child = Node(right_weights, root._layer + 1, 1)
        else:
            left_child = Node(left_weights, root._layer, root._neuron + 1)
            right_child = Node(right_weights, root._layer, root._neuron + 1)

        root.add_left_child(left_child)
        root.add_right_child(right_child)

        self.build_tree(left_child)
        self.build_tree(right_child)

    def induce_oblique_tree(self, weights, dims):
        self._dims = dims

        for i in range(1, len(dims)-1):
            weights[f'OW{i}'] = np.zeros((dims[i], dims[0]))
            weights[f'OB{i}'] = np.zeros((dims[i], 1))

        # First layer
        weights[f'OW{1}'] = copy.deepcopy(weights[f'W{1}'])
        weights[f'OB{1}'] = copy.deepcopy(weights[f'B{1}'])

        # Output layer
        weights[f'OB{len(dims)-1}'] = np.zeros((dims[-1], 1)) * dims[-1]
        weights[f'OW{len(dims)-1}'] = np.zeros((dims[-1], dims[0])) * dims[-1]

        self._root = Node(weights, 1, 1)
        self.build_tree(self._root)

    def classify_instance(self, root, x):
        if root._layer == len(self._dims) - 1:
            return np.dot(root._weights[f'OW{root._layer}'], x) + root._weights[f'OB{root._layer}'].flatten()
            #return np.diagonal(np.matmul(root._weights[f'OW{root._layer}'], x) + root._weights[f'OB{root._layer}'])

        value_node = np.dot(root._weights[f'OW{root._layer}'][root._neuron - 1], x) + root._weights[f'OB{root._layer}'][root._neuron - 1]

        if value_node < 0:
            label = self.classify_instance(root._left, x)
        else:
            label = self.classify_instance(root._right, x)

        return label

    def classify(self, x):
        return self.classify_instance(self._root, x)
