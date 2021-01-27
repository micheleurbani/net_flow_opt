import copy
import numpy as np
import pandas as pd
import networkx as nx
from plotly.graph_objects import Figure, Scatter
import plotly.figure_factory as ff
from scipy.optimize import Bounds, LinearConstraint, minimize


class Activity(object):
    """
    The object desribes a maintenance activity; this is always associated to a
    group when the activity is added to a plan.

    :param object component: an instance of :class:`core.system.Component`
    object.
    :param float date: the due date of the activity.
    :param floa duration: the duration of the activity.

    """

    def __init__(self, component, date, duration):
        self.component = component
        self.t = date
        self.d = duration
        self.r = None

    def __str__(self):
        return "Component {},\tt={:.3f},\td={:.3f}.".format(id(self.component),
                                                            self.t, self.d)

    def expectedCost(self, x):
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


class Group(object):
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

    def __str__(self):
        message = "Group with {} activities:\n".format(self.size)
        for a in self.activities:
            message += "{}\n".format(a)
        return message

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


class Plan(object):
    """
    The object describes a generic maintenance plan, i.e. a list of activities
    with the associated maintenance date.

    :param list activities: a list of :class:`core.scheduler.Activity` objects.
    :param objecet system: a :class:`core.system.System` object.

    """

    def __init__(
        self,
        activities,
        system,
        grouping_structure=None,
        original_plan=None,
    ):
        self.activities = activities
        self.system = system
        self.IC = 0.0
        self.LF = self.evaluate_flow_reduction()
        self.N = len(self.activities)
        if grouping_structure is not None and original_plan is not None:
            # Save the grouping structure as attribute (for future use in E/T
            # minimization)
            self.grouping_structure = grouping_structure
            # Assign activities to resources
            self.set_resources()
            # Set activities' dates according to the grouping structure.
            # set_dates updates also the IC of the plan.
            self.set_dates(original_plan)
            # Evaluate the new LF
            self.LF = self.evaluate_flow_reduction()

    def __str__(self):
        message = "Plan with {} resources.\n".format(self.system.resources)
        for activity in sorted(self.activities, key=lambda x: x.t):
            message += "{}\n".format(activity)
        return message

    def plot_gantt_chart(self):
        """
        The method returns a :class:`plotly.express.timeline` object.
        """
        try:
            getattr(self, 'grouping_structure')
        except AttributeError:
            gs = np.zeros((self.N, self.N, self.system.resources))
            gs[:, 0, 0] = np.ones(self.N)
            self.grouping_structure = gs
            for i, a in enumerate(self.activities):
                a.idx = i
        plan_data = pd.DataFrame(
            [
                {
                    "Activity": str(id(a)),
                    "Task": "Component {}".format(a.component.idx),
                    "Start": a.t,
                    "Finish": a.t + a.d,
                    "Group": "Group {}".format(
                        np.argmax(
                            np.sum(
                                self.grouping_structure[a.idx, :, :],
                                axis=1,
                            )
                        )
                    ),
                    "Resource": "Crew {}".format(a.r),
                } for a in self.activities
            ]
        )
        # Sort activities in descending order
        plan_data.sort_values(by=["Start"], inplace=True, ascending=False)
        # Convert float dates to datetime
        plan_data["Start"] = pd.to_datetime(plan_data["Start"] * 1e14,
                                            format="%Y-%m-%d")
        plan_data["Finish"] = pd.to_datetime(plan_data["Finish"] * 1e14,
                                             format="%Y-%m-%d")
        # Create figure
        gantt = ff.create_gantt(
            plan_data,
            index_col="Group",
            show_colorbar=True,
        )
        return gantt

    def plot_flow_history(self):
        """ Return a Plotly Dash figure object representing the flow value as
        a function of time."""
        history = self.generate_flow_history()
        df = pd.DataFrame(history)
        # Change real values used for dates with datetime strings
        fig = Figure()
        fig.add_trace(
            Scatter(
                x=pd.to_datetime(df["date"] * 1e14, format="%Y-%m-%d"),
                y=df.flow,
                line_shape="hv",
            )
        )
        return fig

    def set_dates(self, original_plan):
        """
        Implement the whole optimization procedure: firstly, activities are
        scheduled at group date, and subsequently they are ordered according to
        group date. Finally, the trust region constrained algorithm is used to
        resolve conflicts about the use of resources.
        The IC of the plan is updated when activities are deferred.

        :param np.array grouping_structure: the array encoding the assignment
        of activities to groups and to resources.
        """
        # Initialize IC to 0
        self.IC = 0.0
        # Add index to activities
        for i, a in enumerate(self.activities):
            a.idx = i
        # Lists to store coefficients of the optimization problem
        A, lb, ub = [], [], []
        ##################################
        # PARTITION ACTIVITIES INTO GROUPS
        ##################################
        # Squeeze the matrix containing the grouping structure to containg only
        # the information about assignment to a group
        grouping_structure = np.sum(self.grouping_structure, axis=-1)
        # Empty list to store group dates
        groups = []
        # Iterate over the columns of the grouping structure
        for j in range(grouping_structure.shape[1]):
            # Avoid calculation for empty groups
            if grouping_structure[:, j].any():
                # Use Group objects to set activities' dates
                g = Group(
                    activities=[self.activities[i] for i in
                                range(self.system.N) if
                                grouping_structure[i, j] == 1]
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
        # Define the LinearConstraint objecet
        linear_constraints = LinearConstraint(
            A=np.stack(A),
            lb=np.array(lb),
            ub=np.array(ub),
        )
        # Define variable bounds
        bounds = Bounds(
            lb=np.array([a.t - a.component.x_star + 1e-3 for a in
                         original_plan.activities]),
            ub=np.repeat(1e5, len(self.activities)),
        )
        # Define the objective function

        def obj(t):
            return sum((a.h(t[i] - a.t) for i, a in
                        enumerate(original_plan.activities)))

        # Define the Jacobian of the objective function

        def jacobian(t):
            return np.array([a.dh(t[i] - a.t) for i, a in
                             enumerate(original_plan.activities)])

        # Define and solve the problem
        solution = minimize(
            fun=obj,
            x0=[a.t for a in self.activities],
            method='trust-constr',
            constraints=[linear_constraints],
            bounds=bounds,
            jac=jacobian,
        )
        for a in self.activities:
            a.t = solution.x[a.idx]
        self.IC = sum([a.h(a.t - original_plan.activities[a.idx].t) for a in
                       self.activities])

    def set_resources(self):
        """Store the resource id in each activity."""
        for i, a in enumerate(self.activities):
            # Perform the assignment
            a.r = np.argmax(
                np.sum(
                    self.grouping_structure[i],
                    axis=0
                )
            )

    def generate_random_assignment_matrix(self):
        """ The method generates a random assignment matrix, which might
        generate an infeasible maintenance plan."""
        sgm = []
        for i in range(self.N):
            x = np.zeros((self.N, self.system.resources))
            x[np.random.randint(self.system.N),
              np.random.randint(self.system.resources)] = 1
            sgm.append(x)
        sgm = np.stack(sgm)
        # print(sgm)
        return sgm

    def generate_structure_history(self):
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
        history = [{"date": a.t} for a in self.activities] + \
            [{"date": a.t + a.d} for a in self.activities]
        # Sort events in ascending order of date
        history.sort(key=lambda a: a["date"])
        # Iterate over all events to find the set of non-active nodes
        for event in history:
            # Create a deep copy of the DiGraph object
            event["structure"] = copy.deepcopy(self.system.structure)
            # Collect non-active nodes' idxs
            node_ids = [
                    a.component.idx for a in self.activities if
                    a.t + a.d > event["date"] and
                    a.t <= event["date"]
                ]
            # Remove non-active nodes from the graph
            event["structure"].remove_nodes_from(
                [
                    n for n in event["structure"].nodes if type(n) is not str
                    and n.idx in node_ids
                ]
            )
        return history

    def generate_flow_history(self):
        """
        The method retuns the dictionary obtained by evaluating
        :method:`core.scheduler.generate_structure_history` enriched with the
        value maximum flow value under the specific system configurations.
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
        history = self.generate_structure_history()
        for config in history:
            config["flow"] = nx.maximum_flow_value(
                self.system.from_node_to_edge_capacity(config["structure"]),
                "s",
                "t"
            )
        return history

    def evaluate_flow_reduction(self):
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
        history = self.generate_flow_history()
        lf = 0.0
        for i in range(len(history) - 1):
            lf += (self.system.regular_flow - history[i]["flow"]) * \
                (history[i + 1]["date"] - history[i]["date"])
        return lf
