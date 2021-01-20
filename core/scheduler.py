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
        self.t_opt = self.t

    def expectedCost(self, x):
        return self.component.cp + self.component.cc * \
            (x / self.component.alpha) ** self.component.beta

    def h(self, delta_t):
        """Penalty function to defer the component from its PM date."""
        return self.expectedCost(self.component.xStar + delta_t) - \
            self.expectedCost(self.component.xStar) - delta_t * \
            self.component.phiStar

    def dh(self, delta_t):
        """Derivative of the penalty function."""
        return self.component.cc * self.component.alpha ** \
            (-self.component.beta) * (self.component.xStar + delta_t) ** \
            (self.component.beta - 1) * self.compoennt.beta - \
            self.component.phiStar

    def ddh(self, delta_t):
        """Second derivative of the penalty function."""
        return self.component.cc * self.component.alpha ** \
            (-self.component.beta) * self.component.beta * \
            (self.component.beta - 1) * (self.component.xStar + delta_t) \
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

    def H(self, x):
        """Return the expected cost of corrective maintenance of the group."""
        return sum((a.h(x - a.t) for a in self.activities))

    def dH(self, x):
        return sum((a.dh(x - a.t) for a in self.activities))

    def ddH(self, x):
        return sum((a.ddh(x - a.t) for a in self.activities))


class Plan(object):

    pass
