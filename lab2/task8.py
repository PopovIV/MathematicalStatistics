import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2, t, norm, moment
import scipy.stats as stats

gamma = 0.95
alpha = 0.05

def student_mo(samples, alpha):
    n = len(samples)
    q_1 = np.mean(samples) - np.std(samples) * t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n - 1)
    q_2 = np.mean(samples) + np.std(samples) * t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n - 1)
    return q_1, q_2

def chi_sigma(samples, alpha):
    n = len(samples)
    q_1 =  np.std(samples) * np.sqrt(n) / np.sqrt(chi2.ppf(1 - alpha / 2, n - 1))
    q_2 = np.std(samples) * np.sqrt(n) / np.sqrt(chi2.ppf(alpha / 2, n - 1))
    return q_1, q_2


def as_mo(samples, alpha):
    n = len(samples)
    q_1 = np.mean(samples) - np.std(samples) * norm.ppf(1 - alpha / 2) / np.sqrt(n)
    q_2 = np.mean(samples) + np.std(samples) * norm.ppf(1 - alpha / 2) / np.sqrt(n)
    return q_1, q_2


def as_sigma(samples, alpha):
    n = len(samples)
    s = np.std(samples)
    U = norm.ppf(1 - alpha / 2) * np.sqrt((moment(samples, 4) / (s * s * s * s) + 2) / n)
    q_1 = s / np.sqrt(1 + U)
    q_2 = s / np.sqrt(1 - U)
    return q_1, q_2


def task8():
  samples20 = np.random.normal(0, 1, size=20)
  samples100 = np.random.normal(0, 1, size=100)
  student_20 = student_mo(samples20, alpha)
  student_100 = student_mo(samples100, alpha)
  chi_20 = chi_sigma(samples20, alpha)
  chi_100 = chi_sigma(samples100, alpha)
  as_mo_20 = as_mo(samples20, alpha)
  as_mo_100 = as_mo(samples100, alpha)
  as_d_20 = as_sigma(samples20, alpha)
  as_d_100 = as_sigma(samples100, alpha)

  print(f"Classic:\n"
        f"n = 20 \n"
        f"\t\t m: " + str(student_20) + " \t sigma: " + str(chi_20) + "\n"
        f"n = 100 \n"
        f"\t\t m: " + str(student_100) + " \t sigma: " + str(chi_100) + "\n")

  print(f"Asymptotic:\n"
        f"n = 20 \n"
        f"\t\t m: " + str(as_mo_20) + " \t sigma: " + str(as_d_20) + "\n"
        f"n = 100 \n"
        f"\t\t m: " + str(as_mo_100) + " \t sigma: " + str(as_d_100) + "\n")