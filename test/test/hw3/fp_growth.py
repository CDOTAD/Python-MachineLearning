from collections import defaultdict, namedtuple

from itertools import imap

def find_frequent_itemsets(transactions, minimum_support, include_support=False):


    items = defaultdict(lambda: 0) 



    for transaction in transactions:
        for item in transaction:
            items[item] += 1

    items = dict((item, support) for item, support in items.iteritems()
        if support >= minimum_support)


    def clean_transaction(transaction):
        transaction = filter(lambda v: v in items, transaction)
        transaction.sort(key=lambda v: items[v], reverse=True)
        return transaction

    master = FPTree()

    for transaction in imap(clean_transaction, transactions):
        master.add(transaction)



    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)

            if support >= minimum_support and item not in suffix:
                found_set = [item] + suffix
                yield (found_set, support) if include_support else found_set

                cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))

                for s in find_with_suffix(cond_tree, found_set):
                    yield s

    for itemset in find_with_suffix(master, []):
        yield itemset

class FPTree(object):

    Route = namedtuple('Route', 'head tail')


    def __init__(self):

        self._root = FPNode(self, None, None)

        self._routes = {}



    @property

    def root(self):
        return self._root



    def add(self, transaction):
        point = self._root



        for item in transaction:

            next_point = point.search(item)

            if next_point:
                next_point.increment()

            else:

                next_point = FPNode(self, item)
                point.add(next_point)

                self._update_route(next_point)



            point = next_point



    def _update_route(self, point):

        assert self is point.tree



        try:

            route = self._routes[point.item]

            route[1].neighbor = point # route[1] is the tail

            self._routes[point.item] = self.Route(route[0], point)

        except KeyError:

            self._routes[point.item] = self.Route(point, point)



    def items(self):

        for item in self._routes.iterkeys():
            yield (item, self.nodes(item))



    def nodes(self, item):

        try:
            node = self._routes[item][0]

        except KeyError:
            return



        while node:
            yield node

            node = node.neighbor



    def prefix_paths(self, item):
      



        def collect_path(node):

            path = []

            while node and not node.root:

                path.append(node)

                node = node.parent

            path.reverse()

            return path



        return (collect_path(node) for node in self.nodes(item))



    def inspect(self):

        print 'Tree:'

        self.root.inspect(1)



        print

        print 'Routes:'

        for item, nodes in self.items():

            print '  %r' % item

            for node in nodes:

                print '    %r' % node

def conditional_tree_from_paths(paths):

    """Build a conditional FP-tree from the given prefix paths."""

    tree = FPTree()

    condition_item = None

    items = set()



    # Import the nodes in the paths into the new tree. Only the counts of the

    # leaf notes matter; the remaining counts will be reconstructed from the

    # leaf counts.

    for path in paths:

        if condition_item is None:

            condition_item = path[-1].item



        point = tree.root

        for node in path:

            next_point = point.search(node.item)

            if not next_point:

                # Add a new node to the tree.

                items.add(node.item)

                count = node.count if node.item == condition_item else 0

                next_point = FPNode(tree, node.item, count)

                point.add(next_point)

                tree._update_route(next_point)

            point = next_point



    assert condition_item is not None



    # Calculate the counts of the non-leaf nodes.

    for path in tree.prefix_paths(condition_item):

        count = path[-1].count

        for node in reversed(path[:-1]):

            node._count += count



    return tree

class FPNode(object):

    """A node in an FP tree."""



    def __init__(self, tree, item, count=1):

        self._tree = tree

        self._item = item

        self._count = count

        self._parent = None

        self._children = {}

        self._neighbor = None



    def add(self, child):

        """Add the given FPNode `child` as a child of this node."""



        if not isinstance(child, FPNode):

            raise TypeError("Can only add other FPNodes as children")



        if not child.item in self._children:

            self._children[child.item] = child

            child.parent = self



    def search(self, item):

        """

        Check whether this node contains a child node for the given item.

        If so, that node is returned; otherwise, `None` is returned.

        """

        try:

            return self._children[item]

        except KeyError:

            return None



    def __contains__(self, item):

        return item in self._children



    @property

    def tree(self):

        """The tree in which this node appears."""

        return self._tree



    @property

    def item(self):

        """The item contained in this node."""

        return self._item



    @property

    def count(self):

        """The count associated with this node's item."""

        return self._count



    def increment(self):

        """Increment the count associated with this node's item."""

        if self._count is None:

            raise ValueError("Root nodes have no associated count.")

        self._count += 1



    @property

    def root(self):

        """True if this node is the root of a tree; false if otherwise."""

        return self._item is None and self._count is None



    @property

    def leaf(self):

        """True if this node is a leaf in the tree; false if otherwise."""

        return len(self._children) == 0



    @property

    def parent(self):

        """The node's parent"""

        return self._parent



    @parent.setter

    def parent(self, value):

        if value is not None and not isinstance(value, FPNode):

            raise TypeError("A node must have an FPNode as a parent.")

        if value and value.tree is not self.tree:

            raise ValueError("Cannot have a parent from another tree.")

        self._parent = value



    @property

    def neighbor(self):

        """

        The node's neighbor; the one with the same value that is "to the right"

        of it in the tree.

        """

        return self._neighbor



    @neighbor.setter

    def neighbor(self, value):

        if value is not None and not isinstance(value, FPNode):

            raise TypeError("A node must have an FPNode as a neighbor.")

        if value and value.tree is not self.tree:

            raise ValueError("Cannot have a neighbor from another tree.")

        self._neighbor = value



    @property

    def children(self):

        """The nodes that are children of this node."""

        return tuple(self._children.itervalues())



    def inspect(self, depth=0):

        print ('  ' * depth) + repr(self)

        for child in self.children:

            child.inspect(depth + 1)



    def __repr__(self):

        if self.root:

            return "<%s (root)>" % type(self).__name__

        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)

