from pandas import read_csv

"""
Some physics constants.
"""

# Диэлектрическая проницаемость в вакууме :)
e0 = 8.85e-12

"""
Material class for loading and easy use of different material constants.
"""

# Путь к файлу с константами для материалов
PATH_TO_MATERIALS = 'materials/material_constants.csv'


class Material:
    def __init__(self, name: str = 'YZNiobate'):
        material_data = read_csv(PATH_TO_MATERIALS, index_col='name').loc[name, :]
        self.v = material_data[0]
        self.dvv = material_data[1]
        self.einf_e0 = material_data[2]
        self.TCD = material_data[3]
        self.einf = self.einf_e0 * e0
        self.vs = self.v * (1 - self.dvv)
