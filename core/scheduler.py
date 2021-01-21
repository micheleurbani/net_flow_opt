import copy
import pandas as pd
import networkx as nx
import plotly.express as px
import matplotlib.pyplot as plt


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
        for a in self.activities:
            a.t = x[-1]


class Plan(object):
    """
    The object describes a generic maintenance plan, i.e. a list of activities
    with the associated maintenance date.

    :param list activities: a list of :class:`core.scheduler.Activity` objects.
    :param objecet system: a :class:`core.system.System` object.

    """

    def __init__(self, activities, system, grouping_structure=None):
        self.activities = activities
        self.system = system
        if grouping_structure is not None:
            # Save the grouping structure as attribute (for future use in E/T
            # minimization)
            self.grouping_structure = grouping_structure
            # Set activities' dates according to the grouping structure
            self.set_dates(self.grouping_structure)

    def __str__(self):
        message = "Plan with {} resources.\n".format(self.system.resources)
        for activity in sorted(self.activities, key=lambda x: x.t):
            message += "{}\n".format(activity)
        return message

    def gantt_chart(self):
        """
        The method returns a :class:`plotly.express.timeline` object.
        """
        plan_data = pd.DataFrame(
            [
                {
                    "Activity": str(id(a)),
                    "Component": str(id(a.component)),
                    "Start": a.t,
                    "End": a.t + a.d,
                } for a in self.activities
            ]
        )
        # Sort activities in descending order
        plan_data.sort_values(by=["Start"], inplace=True, ascending=False)
        # Convert float dates to datetime
        plan_data["Start"] = pd.to_datetime(plan_data["Start"] * 1e14,
                                            format="%Y-%m-%d")
        plan_data["End"] = pd.to_datetime(plan_data["End"] * 1e14,
                                          format="%Y-%m-%d")
        # Create figure
        gantt = px.timeline(
            data_frame=plan_data,
            x_start="Start",
            x_end="End",
            y="Component",
        )
        # gantt.show()
        return gantt

    def set_dates(self, grouping_structure):
        """
        Implement the whole optimization procedure: firstly, activities are
        scheduled at group date, and subsequently they are ordered according to
        group date. Finally, the trust region constrained algorithm is used to
        resolve conflicts about the use of resources.

        :param np.array grouping_structure: the array encoding the assignment
        of activities to groups and to resources.
        """
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
                g.minimize()

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
        dates = [{"date": a.t} for a in self.activities] + \
            [{"date": a.t + a.d} for a in self.activities]
        # Add the start date at time 0
        dates.append({"date": 0.0})
        # Sort events in ascending order of date
        dates.sort(key=lambda a: a["date"])
        # Iterate over all events to find the set of non-active nodes
        for event in dates:
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

    def total_flow_reduction(self):
        """
        The method returns the total reduction of flow with respect to the
        nominal capacity; the latter is the rate of material, or work, that is
        processed per unit of time.

        The information in reduction must be processed before to perform the
        calculation: the ``nx.DiGraph`` object contained in each tuple must be
        converted to a flow reduction value.
        Therefore, the reduction of flow is:

        .. math::

            \\sum \\left( \\mu(N) - \\mu(A_i) \\right) \\cdot d_i

        where :math:`i` denotes a system configuration, and :math:`A_i` denotes
        the set of active nodes under configuration :math:`i`.

        :return: a scalar, which quantifies the total reduction of flow
        generated by following the maintenance plan.
        :rtype: float

        """
        reductions_history = self.generate_structure_history()
        reductions = []
        for interval in reductions_history:
            reduction = self._regular_flow - \
                nx.maximum_flow_value(
                    self.from_node_to_edge_capacity(interval[0]),
                    "s", "t"
                )
            assert reduction >= 0
            assert interval[1] >= 0
            reductions.append(reduction * interval[1])
        return sum(reductions)
