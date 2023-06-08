"""
Реализация класса для различных видов аподизации.
"""

import numpy as np
import scipy


class Apodization:

    def __init__(self):
        """self.p = p
        self.Np = Np
        self.v = v
        self.W = W"""

    @staticmethod
    def none(**kwargs):
        Pm = np.array([x % 2 for x in range(kwargs["Np"])])

        return Pm, kwargs["W"]

    @staticmethod
    def sinc(**kwargs):
        lam = 2 * kwargs["p"]
        f_0 = kwargs["v"] / (2 * kwargs["p"])

        dt = 0.5 * lam / kwargs["v"]  # TODO Почему 0.5?
        ZM = np.arange(0, kwargs["Np"]) * kwargs["v"] * dt

        x0 = kwargs["p"] * kwargs["Np"] / 2
        XM = 2 * np.pi * (kwargs["bandwidth"] * f_0) * (ZM - x0) / kwargs["v"]  # TODO Почему 2?

        Wm = np.array([kwargs["W"] * np.sinc(x / np.pi) for x in XM])

        Pm = np.array([x % 2 for x in range(kwargs["Np"])]) * (Wm / kwargs["W"]) * scipy.signal.windows.kaiser(
            kwargs["Np"], 4)  # TODO Почему 4?

        return Pm, Wm
