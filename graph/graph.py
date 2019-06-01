from ops.input import Input


class Graph(object):

    def __init__(self, feed_dict):
        self.feed_dict = feed_dict
        self.ordered_nodes = self.topological_sort()

    def topological_sort(self):
        """
        Topological sort all nodes in the graph
        :return:  Topologically sorted list of nodes
        """
        input_nodes = [n for n in self.feed_dict.keys()]

        G = {}
        nodes = [n for n in input_nodes]
        while len(nodes) > 0:
            n = nodes.pop(0)
            if n not in G:
                G[n] = {'in': set(), 'out': set()}
            for m in n.output_nodes:
                if m not in G:
                    G[m] = {'in': set(), 'out': set()}
                G[n]['out'].add(m)
                G[m]['in'].add(n)
                nodes.append(m)

        L = []
        S = set(input_nodes)
        while len(S) > 0:
            n = S.pop()

            if isinstance(n, Input):
                n.output = self.feed_dict[n]

            L.append(n)
            for m in n.output_nodes:
                G[n]['out'].remove(m)
                G[m]['in'].remove(n)
                if len(G[m]['in']) == 0:
                    S.add(m)
        return L

    def compute_gradients(self):

        # compute output
        for node in self.ordered_nodes:
            node.compute()

        # compute gradients
        for node in self.ordered_nodes[::-1]:
            node.backpass()

    def loss(self):
        """
        Show final loss of graph.  Must have completed one forward pass.
        :return: Loss of graph
        """
        return self.ordered_nodes[-1].output
