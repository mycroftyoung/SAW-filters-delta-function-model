"""
Some physics constants.
"""

e0 = 8.85e-12

"""
Material constants as classes for easy acces and use.
"""
# TODO Create better architecture for materials!!!

class MYZNiobate:
    v = 3488
    dvv = 0.024
    einf_e0 = 46
    TCD = 94
    einf = einf_e0 * e0


class M128YXNiobate:
    v = 3979
    dvv = 0.027
    einf_e0 = 56
    TCD = 75
    einf = einf_e0 * e0
