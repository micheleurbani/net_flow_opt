class InfeasibleGroup(Exception):

    def __init__(self, *args, **kwargs):
        self.message = "The group is infeasible due to uncompatible date \
        ranges."

    def __str__(self):
        return self.message
