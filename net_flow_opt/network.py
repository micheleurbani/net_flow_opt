import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from plotly.graph_objects import Figure, Scatter
import plotly.figure_factory as ff
from scipy.optimize import Bounds, LinearConstraint, minimize
import matplotlib.pyplot as plt


class Component:

    def __init__(self, cc, cp, alpha, beta, capacity, service_time):
        self.cc = cc
        self.cp = cp
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.d = service_time
        self.x_star = self.alpha * (self.cp / (self.cc * (self.beta - 1))) ** \
            (1 / self.beta)
        self.phi_star = (self.cp * self.beta) / (self.x_star * (self.beta - 1))


class System:

    def __init__(self, structure, components):
        # Sanity cheks
        assert type(structure) is nx.DiGraph
        assert type(components) is list
        for c in components:
            assert type(c) is Component

        # Attributes
        self.structure = structure
        self.components = self.indexing(components)
        self.regular_flow = nx.maximum_flow_value(
            self.from_node_to_edge_capacity(self.structure), "s", "t"
        )
        self.N = len(self.components)

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
            # self.resources = resources
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


class Activity:

    def __init__(self, component):
        self.component = component
        self.t = self.component.x_star
        self.d = self.component.d
        self.r = None

    def expectedCost(self, x):
        assert self.t + x > 0
        return self.component.cp + self.component.cc * \
            (x / self.component.alpha) ** self.component.beta

    def h(self, delta_t):
        """Penalty function to defer the component from its PM date."""
        return self.expectedCost(self.component.x_star + delta_t) - \
            self.expectedCost(self.component.x_star) - delta_t * \
            self.component.phi_star

    def dh(self, delta_t):
        """Derivative of the penalty function."""
        return self.component.cc * self.component.alpha ** \
            (-self.component.beta) * (self.component.x_star + delta_t) ** \
            (self.component.beta - 1) * self.component.beta - \
            self.component.phi_star

    def ddh(self, delta_t):
        """Second derivative of the penalty function."""
        return self.component.cc * self.component.alpha ** \
            (-self.component.beta) * self.component.beta * \
            (self.component.beta - 1) * (self.component.x_star + delta_t) \
            ** (self.component.beta - 2)


class Group:
    """
    Describes a group of maintenance activities that are part of a maintenance
    plan. The objects should be used only to find the *optimal* execution date
    ---i.e., the one that minimizes :math:`IC`.
    Only the method :class:`core.scheduler.Group.minimize` changes the property
    `t_opt` of the activities in the group.
    """

    def __init__(self, activities):
        self.activities = activities
        self.size = len(self.activities)
        self.IC = None

    def H(self, x):
        """Return the expected cost of corrective maintenance of the group."""
        return sum((a.h(x - a.t) for a in self.activities))

    def dH(self, x):
        return sum((a.dh(x - a.t) for a in self.activities))

    def ddH(self, x):
        return sum((a.ddh(x - a.t) for a in self.activities))

    def is_feasible(self):
        earliest_start = max((a.t - a.component.x_star for a in
                              self.activities))
        is_feasible = True
        for a in self.activities:
            if a.t + a.component.x_star < earliest_start:
                is_feasible = False
        return is_feasible

    def minimize(self):
        """
        Implement the Newton method to find the optimal execution date for the
        group, and the `t_opt` of each component is set to the found date.
        """
        x = [sum((a.t for a in self.activities)) / self.size]
        diff = 1e5
        while diff > 1e-3:
            x.append(x[-1] - self.dH(x[-1]) / self.ddH(x[-1]))
            diff = x[-2] - x[-1]
        # Store the extra expected cost of maintenance
        self.IC = self.H(x[-1])
        # Update the execution date of the activities
        for a in self.activities:
            a.t = x[-1]
        return x[-1]


class Plan:

    def __init__(self, activities, system):
        self.activities = activities
        self.system = system
        self.IC = 0
        self.LF = 0

    def eval(self, x):
        IC = self.evaluate_IC(x)
        LF = self.evaluate_LF(x)
        return (IC, LF)

    def compute_dates(self, grouping_structure):
        """
        Implement the whole optimization procedure: firstly, activities are
        scheduled at group date, and subsequently they are ordered according to
        group date. Finally, the trust region constrained algorithm is used to
        resolve conflicts about the use of resources.
        The IC of the plan is updated when activities are deferred.

        :param np.array grouping_structure: the array encoding the assignment
        of activities to groups and to resources (one-hot encoding).
        """
        # Add index to activities
        for i, a in enumerate(self.activities):
            a.idx = i
        # Lists to store coefficients of the optimization problem
        A, lb, ub = [], [], []
        ##################################
        # PARTITION ACTIVITIES INTO GROUPS
        ##################################
        # Squeeze the matrix containing the grouping structure to containing only
        # the information about assignment to a group
        # Empty list to store group dates
        groups = []
        # Iterate over the columns of the grouping structure
        for j in range(grouping_structure.shape[1]):
            # Avoid calculation for empty groups
            if grouping_structure[:, j].any():
                # Use Group objects to set activities' dates
                g = Group(
                    activities=[
                        self.activities[i] for i in range(self.system.N)
                        if grouping_structure[i, j] == 1
                    ]
                )
                groups.append(
                    {
                        "date": g.minimize(),
                        "duration": max([a.d for a in g.activities]),
                        "id_activity": g.activities[
                            np.argmax([a.d for a in g.activities])
                        ].idx,
                    }
                )
                # Sort activities in ascending order of duration
                g.activities.sort(key=lambda x: x.d, reverse=True)
                # Define the constraint to guarantee the nested interval
                if len(g.activities) > 1:
                    for i in range(len(g.activities) - 1):
                        c = np.zeros(self.system.N)
                        c[g.activities[i].idx] = -1
                        c[g.activities[i + 1].idx] = 1
                        A.append(c)
                        lb.append(0.0)
                        ub.append(
                            g.activities[i].d - g.activities[i + 1].d
                        )
        ################################################
        # FORMULATE THE CONSTRAINED OPTIMIZATION PROBLEM
        ################################################
        # Define constraints to avoid the superposition of groups
        groups.sort(key=lambda x: x["date"])
        for i in range(len(groups) - 1):
            c = np.zeros(self.system.N)
            c[groups[i]["id_activity"]] = 1
            c[groups[i + 1]["id_activity"]] = -1
            A.append(c)
            lb.append(-1e5)
            ub.append(-groups[i]["duration"])
        # Define the LinearConstraint object
        linear_constraints = LinearConstraint(
            A=np.stack(A),
            lb=np.array(lb),
            ub=np.array(ub),
        )
        # Define variable bounds
        bounds = Bounds(
            lb=np.array([a.t - a.component.x_star + 1e-3 for a in
                         self.activities]),
            ub=np.repeat(1e5, len(self.activities)),
        )
        # Define the objective function

        def obj(t):
            return sum((a.h(t[i] - a.t) for i, a in
                        enumerate(self.activities)))

        # Define the Jacobian of the objective function

        def jacobian(t):
            return np.array([a.dh(t[i] - a.t) for i, a in
                             enumerate(self.activities)])

        # Define and solve the problem
        solution = minimize(
            fun=obj,
            x0=[a.t for a in self.activities],
            method='trust-constr',
            constraints=[linear_constraints],
            bounds=bounds,
            jac=jacobian,
        )
        return [solution.x[a.idx] for a in self.activities]

    def generate_structure_history(self, dates):
        """
        The method returns a list of system configurations and their start
        date.

        :return: a list of tuples containing a :class:`networkx.DiGraph`
        object, a float with the duration of the configuration, the start date
        of the configuration.
        :rtype: list
        """
        history = []
        # Gather events' dates
        history = [{"date": t} for t in dates] + \
            [{"date": dates[i] + self.activities[i].d} for i in range(len(dates))]
        # Sort events in ascending order of date
        history.sort(key=lambda a: a["date"])
        # Iterate over all events to find the set of non-active nodes
        for event in history:
            # Create a deep copy of the DiGraph object
            event["structure"] = deepcopy(self.system.structure)
            # Collect non-active nodes' idxs
            node_ids = [
                    a.component.idx for i, a in enumerate(self.activities)
                    if dates[i] + a.d > event["date"] and dates[i] <= event["date"]
                ]
            # Remove non-active nodes from the graph
            event["structure"].remove_nodes_from(
                [
                    n for n in event["structure"].nodes if type(n) is not str
                    and n.idx in node_ids
                ]
            )
        return history

    def generate_flow_history(self, t):
        """
        The method returns the dictionary obtained by evaluating
        :method:`core.scheduler.generate_structure_history` enriched with the
        maximum flow value under the specific system configurations.
        The maximum flow value is always calculated from source `s` to sink
        `t`.

        :return: a list of dictionaries. Each dictionary represents an event
        and it stores the following information:

            - ``date``: the date of the event at which there is a change of
            system configuration:
            - ``structure``: a :class:`networkx.DiGraph` object representing
            the structure of the system;
            - ``flow``: the maximum flow value with the given system
            configuration.

        """
        history = self.generate_structure_history(t)
        for config in history:
            config["flow"] = nx.maximum_flow_value(
                self.system.from_node_to_edge_capacity(config["structure"]),
                "s",
                "t"
            )
        return history

    def evaluate_LF(self, t):
        """
        The method returns the total reduction of flow with respect to the
        nominal capacity; the latter is the rate of material, or work, that is
        processed per unit of time.

        The reduction of flow is calculated using the following equation:

        .. math:

            LF(P) = \\int_0^T \\phi(A_t, t) dt

        where :math:`\\phi(A_t,t)` is the instantaneous loss of flow at time
        :math:`t`.
        """
        history = self.generate_flow_history(t)
        lf = 0.0
        for i in range(len(history) - 1):
            lf += (self.system.regular_flow - history[i]["flow"]) * \
                (history[i + 1]["date"] - history[i]["date"])
        return lf

    def evaluate_IC(self, t):
        return np.sum(
                [a.h(t[i] - a.component.x_star) for i, a in enumerate(self.activities)]
            )

    def plot_gantt_chart(self, t):
        fig, ax = plt.subplots()
        order = np.argsort(t)
        j = 0
        for i in order:
            ax.barh(y=j, width=self.activities[i].d, left=t[i], color='tab:blue')
            j += 1
        ax.set_title("Gantt")
        ax.set_xlabel("Time")
        ax.set_yticks(np.arange(self.system.N))
        ax.set_yticklabels(["Component %s" % i for i in order])
        return fig
        # """
        # The method returns a :class:`plotly.express.timeline` object.
        # """
        # try:
        #     getattr(self, 'grouping_structure')
        # except AttributeError:
        #     gs = np.zeros((self.system.N, self.system.N))
        #     gs[:, 0] = np.ones(self.system.N)
        #     self.grouping_structure = gs
        #     for i, a in enumerate(self.activities):
        #         a.idx = i
        # plan_data = pd.DataFrame(
        #     [
        #         {
        #             "Activity": str(id(a)),
        #             "Task": "Component {}".format(a.component.idx),
        #             "Start": t[i],
        #             "Finish": t[i] + a.d,
        #             "Group": "Group {}".format(
        #                 np.argmax(
        #                     self.grouping_structure[a.idx, :],
        #                 )
        #             ),
        #         } for i, a in enumerate(self.activities)
        #     ]
        # )
        # # Sort activities in descending order
        # plan_data.sort_values(by=["Start"], inplace=True, ascending=False)
        # # Save data in CSV for exporting
        # # plan_data.to_csv(os.path.join("gantt_data.csv"), index=False)
        # # Convert float dates to datetime
        # plan_data["Start"] = pd.to_datetime(
        #     plan_data["Start"] * 1e14,
        # )
        # plan_data["Finish"] = pd.to_datetime(
        #     plan_data["Finish"] * 1e14,
        # )
        # # Create figure
        # gantt = ff.create_gantt(
        #     plan_data,
        #     index_col="Group",
        #     show_colorbar=True,
        # )
        # return gantt

    def plot_flow_history(self):
        """ Return a Plotly Dash figure object representing the flow value as
        a function of time."""
        history = self.generate_flow_history()
        # Append start and end date
        history.append({
            "date": min([i["date"] for i in history]) * 0.93,
            "flow": self.system.regular_flow,
        })
        history.append({
            "date": max([i["date"] for i in history]) * 1.03,
            "flow": self.system.regular_flow,
        })
        history.sort(key=lambda x: x["date"])
        df = pd.DataFrame(history)
        df.to_csv("flow_data.csv", index=False)
        # Change real values used for dates with datetime strings
        fig = Figure()
        fig.add_trace(
            Scatter(
                x=pd.to_datetime(df["date"] * 1e14),
                y=df.flow,
                line_shape="hv",
            )
        )
        fig.update_layout(
            yaxis={
                "range": (0, self.system.regular_flow + 3)
            }
        )
        return fig