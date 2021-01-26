import networkx as nx


from .system import Component


components = [
    Component(
        cc=40,
        cp=15,
        alpha=35,
        beta=2.1,
        capacity=20,
    ),
    Component(50, 35, 46, 1.9, 15),
    Component(27, 15, 45, 2.1, 9),
    Component(46, 22, 55, 1.4, 11),
    Component(29, 15, 34, 2.5, 7),
    Component(31, 21, 55, 1.9, 8),
    Component(55, 26, 61, 1.6, 15),
    Component(61, 27, 58, 2.2, 21),
    Component(45, 18, 36, 2.4, 10),
    Component(50, 17, 39, 2.0, 9),
]

structure = nx.DiGraph()
# Add source and sink nodes
structure.add_nodes_from(["s", "t"])
structure.add_nodes_from(components)
structure.add_edges_from([
    ("s", components[0]),
    ("s", components[1]),
    (components[0], components[2]),
    (components[0], components[3]),
    (components[1], components[4]),
    (components[1], components[5]),
    (components[2], components[6]),
    (components[3], components[6]),
    (components[3], components[7]),
    (components[4], components[7]),
    (components[5], components[7]),
    (components[6], "t"),
    (components[7], components[8]),
    (components[7], components[9]),
    (components[8], "t"),
    (components[9], "t"),
])
