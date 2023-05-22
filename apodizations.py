import numpy as np
import scipy


class Apodization:

    def __init__(self):
        """self.p = p
        self.Np = Np
        self.v = v
        self.W = W"""

    # TODO: Что-то сделать с тем, что функция "none" берет лишние аргументы (это необходимо, чтобы применение аподизации было универсальным)

    @staticmethod
    def none(p, Np, v, W, bandwidth):
        Pm = np.array([x % 2 for x in range(Np)])

        return Pm, W

    @staticmethod
    def sinc(p, Np, v, W, bandwidth):
        lam = 2 * p
        f_0 = v / (2 * p)

        dt = 0.5 * lam / v  # TODO Почему 0.5?
        ZM = np.arange(0, Np) * v * dt

        x0 = p * Np / 2
        XM = 2 * np.pi * (bandwidth * f_0) * (ZM - x0) / v  # TODO Почему 2?

        Wm = np.array([W * np.sinc(x / np.pi) for x in XM])

        Pm = np.array([x % 2 for x in range(Np)]) * (Wm / W) * scipy.signal.windows.kaiser(Np, 4)  # TODO Почему 4?

        return Pm, Wm
