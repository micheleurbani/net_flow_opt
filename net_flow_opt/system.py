import networkx as nx
import matplotlib.pyplot as plt


class Component:
    """
    The object contains the parameters relevant to the the component and to its
    failure distribution.

    :param float cc: the cost of a corrective intervention.
    :param float cp: the cost of a preventive intervention.
    :param float alpha: the scale factor of the Weibull 2-paramters distribution.
    :param float beta: the shape factor of the Weibull 2-paramters distribution.
    :param float capacity: the amount of material processed per unit of time.


    """

    def __init__(self, cc, cp, alpha, beta, capacity):
        self.cc = cc
        self.cp = cp
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.x_star = self.alpha * (self.cp / (self.cc * (self.beta - 1))) ** \
            (1 / self.beta)
        self.phi_star = (self.cp * self.beta) / (self.x_star * (self.beta - 1))

    def __str__(self):
        return f"C{id(self)}"

    def info(self):
        message = "{} {}\n".format(self.__class__.__name__, self.id)
        for attribute in dir(self):
            if "__" not in attribute:
                message += "\t{}: {:.3f}\n".format(
                    attribute, getattr(self, attribute)
                )


class System:
    """The object contains the information about the structure of the system.

    :param object structure: a :class:`networkx.DiGraph` object.
    :param int resources: the number of available repairmen staff.

    """

    def __init__(self, structure, resources, components):
        # Sanity cheks
        assert type(structure) is nx.DiGraph
        assert type(components) is list
        for c in components:
            assert type(c) is Component

        # Attributes
        self.structure = structure
        self.resources = resources
        self.components = self.indexing(components)
        self.regular_flow = nx.maximum_flow_value(
            self.from_node_to_edge_capacity(self.structure), "s", "t"
        )
        self.N = len(self.components)

    def __str__(self):
        message = "System with {} resources.\n".format(self.resources)
        for c in self.components:
            message += "{}\n".format(c)
        return message

    def plot_system_structure(self, structure=None):
        """Draw the graph representing the srtucture of the system."""
        if structure is None:
            structure = self.structure
        nx.draw_networkx(
            nx.DiGraph(structure),
            pos=nx.spectral_layout(nx.DiGraph(structure))
        )
        plt.savefig('graph')

    @staticmethod
    def indexing(components):
        for i, c in enumerate(components):
            assert type(c) is Component
            c.idx = i
        return components

    @staticmethod
    def from_node_to_edge_capacity(structure):
        """
        The function returns the edge capacited version of a node capacited
        network.

        :param object system_structure: the networkx.DiGraph object that
        describes the original structure of the network.

        :return: a networkx.DiGraph object with twinned nodes and edge
        capacitiees. The capacity of the old edges is set to infinite.
        :rtype: object
        """
        e_capacited_graph = nx.DiGraph()
        for n in structure.nodes:
            # Only if the node n is of type Component, it is twinned
            if type(n) is Component:
                # Twin the node and create an input and an output node
                e_capacited_graph.add_nodes_from(
                    ["C" + str(id(n)) + s for s in ["_in", "_out"]]
                )
                # Create a capacited edge between the two nodes: the capacity
                # is that of the component.
                e_capacited_graph.add_edge(
                    "C" + str(id(n)) + "_in",
                    "C" + str(id(n)) + "_out",
                    capacity=n.capacity
                )
            else:
                e_capacited_graph.add_node(n)

        # Now all the "_in" edges must be connected to the "_out" edges
        # according to the original structure of the network.
        for n in structure.nodes:
            # Create edges (with infinite capacity) with predecessors
            if type(n) is Component:
                for n1 in structure.predecessors(n):
                    if type(n1) is Component:
                        e_capacited_graph.add_edge(
                            "C" + str(id(n1)) + "_out",
                            "C" + str(id(n)) + "_in"
                        )
                    elif type(n1) is str:
                        e_capacited_graph.add_edge(
                            n1,
                            "C" + str(id(n)) + "_in"
                        )

            elif type(n) is str:
                # Since only the sink t presents predecessors, we do not need
                # to check the type of the node
                e_capacited_graph.add_edges_from(
                    [(
                        "C" + str(id(e)) + "_out",
                        n
                    ) for e in structure.predecessors(n)]
                )

        return e_capacited_graph
