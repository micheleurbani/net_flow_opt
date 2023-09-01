import numpy as np

from pymoo.indicators.hv import HV
from pymoo.core.callback import Callback


class TrackPerformance(Callback):

    def __init__(self, ref_point) -> None:
        super().__init__()
        self.HV = HV(
            ref_point=ref_point,
            zero_to_one=True,
            nadir=ref_point,
            ideal=np.array([0, 0])
        )
        self.data["hv"] = []

    def notify(self, algorithm):
        self.data["hv"].append(self.HV.do(algorithm.pop.get("F")))


if __name__ == '__main__':
    pass
