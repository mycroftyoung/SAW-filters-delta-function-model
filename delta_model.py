import os

import numpy as np
import scipy
import matplotlib.pyplot as plt
from material_constants import e0
from material_constants import Material
from apodizations import Apodization

"""
Interdigital Transducer (IDT) = Встречно-Штыревой преобразователь (ВШП)
Pitch - Расстояние между электродами (м)
Np - количество электродов в ВШП
Апертура - Длина электрода. В случае аподизированного ВШП задаётся функцией.
Sample rate - частота дискретизации

"""


class IDT:
    """
    Describes IDT with delta-function model.
    """

    def __init__(self, p: float = 10e-6,
                 Np: int = 100,
                 apodization: callable = Apodization.none,
                 aperture: int = 100,
                 material=Material(),
                 freq_sampling: float = 0.1e6,
                 band: str = 'wide',
                 bandwidth: float = 0.05
                 ):
        """

        :param p: Pitch is the distance between electrodes
        :param Np: Quantity of electrodes
        :param apodization: Apodization type (function)
        :param aperture: Aperture (length of a single electrode) length in wavelengths, defines max aperture in apodized filter
        :param material: Material type, required for some constants
        :param freq_sampling: Frequency sampling rate
        :param band: Allows you to configure whether the simulation will be in a wide or in a narrow band
        :param bandwidth: Target bandwidth width as a percentage of main frequency

        """
        self.p = p
        self.Np = Np
        self.material = material
        self.apodization = apodization
        self.a = p / 2

        self.lam = 2 * self.p
        self.f_0 = self.material.v / (2 * self.p)
        self.freq = np.arange(freq_sampling, ((5.5 if band == 'wide' else 1.5) * self.f_0), freq_sampling)
        self.omega = 2 * np.pi * self.freq
        self.Ew_const = 1.694j * self.material.dvv
        self.W = aperture * self.lam
        self.bandwidth = bandwidth
        self.band = band

        self.Kw = self.omega / self.material.v  # Вектор волновых чисел (для свободной поверхности)
        self.Kws = self.omega / self.material.vs  # Для металлизированной поверхности
        self.X = np.array([x * self.p for x in range(self.Np)])  # Координаты точечных излучателей

        # Применение аподизации
        self.Pm, self.Wm = self.apodization(p=self.p, Np=self.Np, v=self.material.v, W=self.W, bandwidth=bandwidth)

        # Расчёт множителя элемента (element factor)
        self.Ew = np.array([self.element_factor(k) for k in self.Kw])

        # Расчёт множителя массива элементов (array factor)
        self.A = np.sum(np.array([self.Pm * np.exp(- self.X * self.Kw[i] * 1j) for i in range(len(self.Kw))]),
                        axis=1)

        # Отклик преобразователя (transducer response)
        self.H = - 1j * self.Ew * self.A * np.sqrt(self.omega * self.W * self.material.einf / self.material.dvv)

    @staticmethod
    def _legendre(x: float, v: int, M: int = 100) -> float:
        """
        This function computes the Legendre polynomial P(x)v
        :param x: argument
        :param v: lowercase parameter
        :param M: sampling
        :return: Legendre function value with given parameters
        """

        N = np.ones(M)
        for i in range(1, M):
            m = i
            am = (m - 1 - v) * (m + v) * (1 - x) / 2 / m / m
            N[i] = am * N[i - 1]
        P = np.sum(N)
        return P

    def _rho_f(self, beta) -> float:
        delta = np.pi * self.a / self.p

        m = np.floor(beta * self.p / (2 * np.pi))
        s = (beta * self.p / (2 * np.pi)) - m

        P1 = self.__class__._legendre(np.cos(delta), m)
        P2 = self.__class__._legendre(-np.cos(delta), -s)

        rho_f = self.material.einf * (2 * np.sin(np.pi * s)) * P1 / P2

        return rho_f

    def element_factor(self, beta) -> np.ndarray:
        ew = 1j * self.material.dvv * self._rho_f(beta) / self.material.einf
        return ew


class Filter:
    """
    A class representing a filter composed of two interdigital transducers (IDTs).

    Main functionalities:
    The Filter class describes a filter consisting of two Interdigital Transducers (IDTs) with a given distance between them. The class calculates the S-parameters of the filter and provides a method to plot the frequency response of the filter.

    Methods:
    - __init__(self, idt_1: IDT, idt_2: IDT, d: float = 10e-3): initializes the Filter object with two IDTs and the distance between them. Calculates the Y-parameters of the filter.
    - s_params(self): calculates and returns the S-parameters of the filter.
    - plot(self, true_freq: bool = False, param: str = "S21"): plots the frequency response of the filter. The parameter 'true_freq' determines whether the x-axis is in MHz or normalized to the center frequency of the IDTs. The parameter 'param' determines which S-parameter to plot.

    Fields:
    - Y11, Y12, Y21, Y22: the Y-parameters of the filter.
    - material: the material of the IDTs.
    - f_0: the center frequency of the IDTs.
    - freq: the frequency range of the IDTs.
    - omega: the angular frequency range of the IDTs.
    - band: the band of the IDTs.
    """

    @staticmethod
    def _check_idts(idt_1, idt_2) -> None:
        """
        Checks if the two IDTs are compatible for use in a filter.

        :param idt_1: The first IDT.
        :param idt_2: The second IDT.
        :raises ValueError: If the IDTs have different bands, sample rates, or materials.
        """

        if idt_2.band != idt_1.band:
            raise ValueError("IDTs have different band")
        if not np.isclose(idt_2.freq, idt_1.freq).all():
            raise ValueError('IDTs have different band or sample rate')
        if idt_2.material != idt_1.material:
            raise ValueError("IDTs have different materials")

        return None

    def __init__(self, idt_1: IDT, idt_2: IDT,
                 d: float = 10e-3, ) -> None:
        """
        Initializes a Filter object.

        :param idt_1: The first IDT.
        :param idt_2: The second IDT.
        :param d: The distance between the two IDTs.
        """
        self.__class__._check_idts(idt_1, idt_2)

        self.material = idt_1.material
        self.f_0 = idt_1.f_0
        self.freq = idt_1.freq
        self.omega = idt_1.omega
        self.band = idt_1.band

        # Ёмкостная связь
        Cio = 0.01e-12
        YCio = -1j * self.omega * Cio

        # Y21 и Y12 считаем по 2 (принимающему) преобразователю
        # TODO: Добавить затухание!!
        self.Y21 = idt_1.H * idt_2.H * np.exp(-1j * idt_2.Kw * d) + YCio
        self.Y12 = self.Y21

        # Y11 и Y22 считаем по 1 (испускающему) преобразователю
        self.Ga = np.abs(idt_1.H ** 2)
        self.Ba = -(scipy.signal.hilbert(self.Ga)).imag
        self.Ct = np.sum(idt_1.Wm * idt_1.material.einf)
        self.Y11 = self.Ga + 1j * self.Ba + 1j * self.omega * self.Ct - YCio
        self.Y22 = self.Y11

    @property
    def s_params(self) -> tuple[float, float, float, float]:
        """
        Calculates the scattering parameters of the filter.

        :return: A tuple containing the S11, S12, S21, and S22 parameters.
        """

        Z0 = 50
        Y11, Y12, Y21, Y22 = self.Y11, self.Y12, self.Y21, self.Y22

        delta = (1 + Z0 * Y11) * (1 + Z0 * Y22) - Z0 ** 2 * Y12 * Y21
        S11 = ((1 - Z0 * Y11) * (1 + Z0 * Y22) + Z0 ** 2 * Y12 * Y21) / delta
        S12 = (-2 * Z0 * Y12) / delta
        S21 = (-2 * Z0 * Y21) / delta
        S22 = ((1 + Z0 * Y11) * (1 - Z0 * Y22) + Z0 ** 2 * Y12 * Y21) / delta

        return S11, S12, S21, S22

    def plot(self, true_freq: bool = False, param: str = "S21") -> None: # TODO: добавить Yпараметры, фазовую хар-ку с наложением на Y\S
        """
        Plots the frequency response of the filter.

        :param true_freq: If True, plots the frequency in MHz. Otherwise, plots the frequency in units of the fundamental frequency.
        :param param: The scattering parameter to plot (S11, S12, S21, or S22).
        """

        fig, ax = plt.subplots()

        s_param_dict = {"S11": 0, "S12": 1, "S21": 2, "S22": 3}
        s_param_index = s_param_dict[param]
        ax.plot((a.freq / 1e6 if true_freq else a.freq / a.f_0), 20 * np.log10(abs(self.s_params[s_param_index])))

        if self.band == "narrow" and not true_freq:
            ax.set_ylim(-90, 0)
            ax.set_xlim(0.85, 1.15)

        ax.set_title(f'АЧХ {param} -- параметров фильтра')
        if true_freq:
            ax.set_xlabel('Частота, MHz')
        ax.set_ylabel(f'|{param}|, дБ')
        ax.grid()
        fig.show()


if __name__ == '__main__':
    a = IDT(Np=100, apodization=Apodization.sinc, band='wide')
    b = IDT(Np=10, band='wide')

    c = Filter(a, b, d=100e-3)
    c.plot(true_freq=True)

