from math import factorial
from math import e
import numpy as np
import matplotlib.pyplot as mpl

def program():

    print("Welcome to the comparison between Binomial and Poisson distributions!")
    print("")

    n = round(float(input("Insert a value (integer) for n (between 0 and 170): ")))

    p = float(input("Insert a value for the probability: "))

    while p < 0 or p > 1:
        print("Insert a value between 0 and 1")
        p = float(input("Insert a value for p: "))

    def nCr(n, r):
        return factorial(n) // factorial(r) // factorial(n - r)

    def poisson(n, k, p):
        probability = (e ** (-n * p) * (n * p) ** k) / factorial(k)
        return probability

    def binomial(n, k, p):
        probability = nCr(n, k) * p ** k * (1 - p) ** (n - k)
        return probability

    list_bin = []
    list_poi = []
    axis_x = []

    for k in np.arange(0, n, 1):

        list_poi.append(poisson(n, k, p))
        list_bin.append(binomial(n, k, p))

    for x in np.arange(0, n, 1):
        axis_x.append(x)

    fig, grf = mpl.subplots()

    grf.plot(axis_x, list_poi, label = 'Poisson')
    grf.plot(axis_x, list_bin, label = 'Binomial')
    grf.legend()
    mpl.xlabel("k")
    mpl.ylabel("probability")
    mpl.show()

    restart = input("Run the program again (y/n)?:  ")

    while restart != "y" and restart != "n":
        restart = input("Run the program again (y/n)?:  ")

    if restart == "y":
        programa()
    elif restart == "n":
        print("Thanks for playing!")

program()


















