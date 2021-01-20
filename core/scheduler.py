class Activity(object):

    def __init__(self, component, date, duration):
        self.component = component
        self.t = date
        self.d = duration
        self.t_opt = None


class Group(object):

    def __init__(self, components):
        self.components = components


class Plan(object):

    pass
