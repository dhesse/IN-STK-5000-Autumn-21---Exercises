import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

class Ex2:

    """We will solve various tasks using a data set containing various
    information on the inner planets. To get you started, task 1 is
    completed. You will have to write the code to complete the
    others. The data set is loaded for you, no need to do anything
    with this."""

    planets = pd.read_csv("inner_planets.csv", index_col=0)
    G = 6.67E-11

    def task1(self):

        """Display the first 2 rows of the inner plantes dataset."""

        self.planets.head(2)

    def task2(self):

        """Display the last 3 rowas of the inner planets dataset."""

        self.planets.tail(3)

    def task3(self):

        """Return a Series containing all information on Venus."""

        return self.planets['VENUS']

    def task3_1(self):

        """Return a DataFrame containing all information on Earth."""

        return self.planets[['EARTH']]

    def task4(self):

        """Return a series containing the masses of all inner
        planets."""

        return self.planets.loc['Mass (10e24kg)']
        
    def task5(self):

        """Return a dataframe containing the mass, diameter, and density of
        Mercury and Venus, accessing the data by label."""
        
        return self.planets.loc['Mass (10e24kg)':'Density (kg/m3)',
                                'MERCURY':'VENUS']

    def task5_1(self):

        """Return a dataframe containing the diameter, density, and gravity of
        Mercury and Mars, accessing the data by position."""
        
        return self.planets.iloc[1:4,[0, 3]]

    def task6(self):

        """The escape velocity of an object of mass M, with radius r, can be
        calculated using the formula

          v_e = sqrt(2 * G * M / r),

        where G is the gravitational constant. You can use self.G to
        access it. More informaiton can be found at 
        https://en.wikipedia.org/wiki/Escape_velocity.

        Write a method to calculate the escape velocity and return it
        in a series, with the planet names as index.
        """

        return np.sqrt(
            self.G
            * self.planets.loc['Mass (10e24kg)'] * 1e24
            / self.planets.loc['Diameter (km)'] / 1000 * 4) / 1000

    def task6_1(self):

        """Calculate the expected escape velocity and subtract the
        actual escape velocity given in the inner planets dataset. Why
        do you think there is such a difference?"""


        # Differences can come from how diameter/radius is measured,
        # planets are not perfect spheres.
        
        return np.sqrt(
            self.G
            * self.planets.loc['Mass (10e24kg)'] * 1e24
            / self.planets.loc['Diameter (km)'] / 1000 * 4
        ) / 1000 - self.planets.loc['Escape Velocity (km/s)']
    
    def task7(self):

        """Use the given dictionary to create a data frame containing
        information about Jupiter and Saturn. Join it with the inner planet
        data and return it."""

        jus_data = {'JUPITER': {'Mass (10e24kg)': 1898,
                      'Diameter (km)': 142984,
                      'Density (kg/m3)': 1326,
                      'Gravity (m/s2)': 23.1,
                      'Escape Velocity (km/s)': 59.5},
                    'SATURN': {'Mass (10e24kg)': 568,
                      'Diameter (km)': 120536,
                      'Density (kg/m3)': 687,
                      'Gravity (m/s2)': 9.0,
                      'Escape Velocity (km/s)': 35.5}}
        
        right = pd.DataFrame(jus_data)
        return self.planets.join(right)
        
    def task8(self):

        """Use the information about the number of moons given in the dictionary
        and append it to the planets data frame. Return the resulting dataframe.
        """

        moons = {"MERCURY": {"Moons": None},
                 "VENUS": {"Moons": None},
                 "EARTH": {"Moons": 1},
                 "MARS": {"Moons": 2}}
        moons_df = DataFrame(moons)
        return self.planets.append(moons_df)

    def task8_1(self):

        """Appending the moons data leaves NaN values in the data frame. Drop 
        those and return the resultng dataframe."""

        return self.task8().dropna()

    def task8_2(self):

        """Appending the moons data leaves NaN values in the data frame. In this
        context, NaN actually means 'no moons'. Thus replace those NaN values
        with 0 and return the resulting dataframe."""

        return self.task8().fillna(0)