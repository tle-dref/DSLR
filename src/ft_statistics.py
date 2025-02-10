from typing import Any

def ft_median(lst: list):
    """Calculate the median of a given list."""
    lst.sort()
    mid = len(lst) // 2
    if len(lst) % 2 == 0:
        return (lst[mid - 1] + lst[mid]) / 2
    else:
        return lst[mid]

def ft_mean(lst: list):
    """Calculate the mean (average) of a given list."""
    return (sum(lst) / len(lst))

def ft_quartile(lst: list):
    """Calculate the first and third quartiles of a given list. (25% and 75%)"""
    lst = sorted(lst)
    mid = len(lst) // 2
    if (len(lst) % 2 == 0):
        lower = lst[:mid]
        upper = lst[mid:]
    else:
        lower = lst[:mid]
        upper = lst[mid + 1:]
    return (ft_median(lower), ft_median(upper))

def ft_variance(lst: list):
    """Calculate the variance of a given list."""
    return (sum((x - sum(lst) / len(lst)) ** 2 for x in lst) / len(lst))

def ft_ecart_type(lst: list):
    """Calculate the std of a given list."""
    return (ft_variance(lst) ** 0.5)


def ft_min_max(lst: list):
    """Return the minimum and the maximum of a given list."""
    min = lst[0]
    max = lst[0]
    for i in lst:
        if (i >= max):
            max = i
        elif (i < min):
            min = i
    return (min, max)
