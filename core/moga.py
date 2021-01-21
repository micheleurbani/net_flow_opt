class Individual:
    """
    The object contains useful information for the MOGA about the individual.

    :param network_system: a :class:`core.system.System` object containing
    system information

    """

    def __init__(self, system, sgm_matrix=None):
        self.system = system
        self.dominated_solutions = []
        self.dominator_counter = 0
        self.rank = 0
        self.score = None
        self.crowding_distance = 0

    def __str__(self):
        if self.score is not None:
            return "Individual, score: U = {:.3f}, H = {:.3f}, rank: {}, cd: \
                {}.".format(self.score[0], self.score[1], self.rank,
                            self.crowding_distance)
        else:
            return "Individual, score: {}, rank: {}, cd: {}."\
                .format(self.score, self.rank, self.crowding_distance)
