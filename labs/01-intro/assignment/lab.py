# lab.py


import os
import time
import io
import pandas as pd
import numpy as np
import doctest

from sphinx.addnodes import index


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.
    :Example:
    >>> consecutive_ints([5, 3, 6, 4, 9, 8])
    True
    >>> consecutive_ints([1, 3, 5, 7, 9])
    False
    >>> consecutive_ints([])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    '''
    median_vs_mean takes in a non-empty list of numbers
    and returns a Boolean of whether the median is
    less than or equal to the mean.
    :param nums: a non-empty list of numbers.
    :returns: bool, whether the median is less than or equal to the mean.
    
    :Example:
    >>> median_vs_mean([6, 5, 4, 3, 2])
    True
    >>> median_vs_mean([50, 20, 15, 40])
    True
    >>> median_vs_mean([1, 8, 9])
    False
    '''
    srt = sorted(nums)
    median = srt[int((len(srt) + 1) / 2) - 1]
    mean = sum(srt) / len(srt)
    return median <= mean




# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i positions apart, whose absolute 
    difference as integers is also i.
    :param ints: a list of integers
    :returns: a bool, describing whether or not the 
    above condition is satisfied
    :Example:
    >>> same_diff_ints([5, 3, 1, 5, 9, 8])
    True
    >>> same_diff_ints([1, 3, 5, 7, 9])
    False
    >>> same_diff_ints([1, 1, 1, -1])
    True
    >>> time1_start = time.time()
    >>> _ = same_diff_ints([1]*1000 + [2])
    >>> time1_end = time.time()
    >>> time1 = time1_end - time1_start
    >>> time2_start = time.time()
    >>> _ = same_diff_ints([1]*1000 + [1001])
    >>> time2_end = time.time()
    >>> time2 = time2_end - time2_start
    >>> time2 - 10 * time1 > 0
    True
    """


    if len(ints) == 0:
        return False

    for i in range(len(ints)):
        a = i + 1
        while a < len(ints):
            # print(i, a, ints[i], ints[a], i - a, ints[i] - ints[a])
            diff = abs(ints[i] - ints[a])
            num_diff = abs(i - a)
            if num_diff == diff:
                return True
            a += 1
    return False




# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    """
    n_prefixes returns a string of n
    consecutive prefix of the input string.

    :param s: a string.
    :param n: an integer

    :returns: a string of n consecutive prefixes of s backwards.
    :Example:
    >>> n_prefixes('Billy', 4)
    'BillBilBiB'
    >>> n_prefixes('Marina', 3)
    'MarMaM'
    >>> n_prefixes('aaron', 2)
    'aaa'
    >>> n_prefixes('Justin', 5)
    'JustiJustJusJuJ'
    >>> n_prefixes('', 1)
    ''
    """
    if len(s) == 0:
        return ""
    new_str = ""
    while n > 0:
        for i in range(0, n):
            new_str += s[i]
        n -= 1
    return new_str


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    """
    exploded_numbers returns a list of strings of numbers from the
    input array each exploded by n.
    Each integer is zero padded.

    :param ints: a list of integers.
    :param n: a non-negative integer.

    :returns: a list of strings of exploded numbers. 
    :Example:
    >>> exploded_numbers([3, 8, 15], 2)
    ['01 02 03 04 05', '06 07 08 09 10', '13 14 15 16 17']
    >>> exploded_numbers([9, 99], 3)
    ['006 007 008 009 010 011 012', '096 097 098 099 100 101 102']
    """

    exploded = []
    a = 0
    max_len = len(str(max(ints) + n))
    while a < len(ints):
        str_explode = ""
        lowest = ints[a] - n
        highest = ints[a] + n
        x = lowest
        while x <= highest:
            str_x = str(x)
            zeroes = ""
            while (len(str_x) + len(zeroes)) < max_len:
                zeroes += "0"
            str_x = zeroes + str_x
            str_explode += str_x + " "
            x += 1
        str_explode = str_explode[:-1]
        exploded.append(str_explode)
        a += 1
    return exploded


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of each line.
    :param fh: a file object to read from.
    :returns: a string of last characters from fh
    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """

    ret_str = ""
    with open(fh.name, "r") as f:
        for line in f:
            ret_str += line[-2]
    return ret_str



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def add_root(A):
    """
    add_root takes in a numpy array and
    adds to each element the square-root of
    the index of each element.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = add_root(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    >>> np.isclose(out[3], 7 + np.sqrt(3))
    True
    """
    ...

def where_square(A):
    """
    where_square takes in a numpy array of integers
    and returns an array of Booleans
    whose ith element is True if and only if the ith element
    of the input array is a perfect square.
    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.
    :Example:
    >>> out = where_square(np.array([1, 2, 16, 17, 32, 49]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    >>> out[2]
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    """
    growth_rates takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = growth_rates(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """
    ...

def with_leftover(A):
    """
    Create a function with_leftover that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: the day on which you can buy at least one share from 'left-over' money
    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = with_leftover(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------



def salary_stats(salary):
    """
    salary_stats returns a series as specified in the notebook.
    :param salary: a DataFrame of NBA salaries as found in `salary.csv`
    :return: a series with index specified in the notebook.
    :Example:
    >>> salary_fp = os.path.join('data', 'salary.csv')
    >>> salary = pd.read_csv(salary_fp)
    >>> out = salary_stats(salary)
    >>> isinstance(out, pd.Series)
    True
    >>> 'total_highest' in out.index
    True
    >>> isinstance(out.loc['duplicates'], bool)
    True
    """
    stats_series = pd.Series()
    stats_series["num_players"] = salary.shape[0]
    stats_series["num_teams"] = salary.get("Team").nunique()
    stats_series["total_salary"] = sum(salary.get("Salary"))
    stats_series["highest_salary"] = salary.get("Salary").max()
    stats_series["avg_bos"] = round(salary[salary.get("Team") == "BOS"].get("Salary").mean(), 2)
    stats_series["third_lowest"] = (
    salary.sort_values("Salary").get("Player").iloc[2], salary.sort_values("Salary").get("Team").iloc[2])
    last_names = []
    for name in salary.get("Player"):
        first_last = name.split(" ")
        last_names.append(first_last[1])
    last_counts = {}
    for name in last_names:
        if name in last_counts.keys():
            last_counts[name] += 1
        else:
            last_counts[name] = 1
    for value in last_counts.values():
        if value != 0:
            stats_series["duplicates"] = True
            break
    else:
        stats_series["duplicates"] = False
    highest_paid = salary.sort_values("Salary").get("Team").iloc[-1]
    stats_series["total_highest"] = salary[salary.get("Team") == highest_paid].get("Salary").sum()
    return stats_series


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    """
    Parses and loads the malformed .csv file into a 
    properly formatted DataFrame (as described in 
    the question).
    :param fh: file handle for the malformed .csv file.
    :returns: a pandas DataFrame of the data, 
    as specified in the question statement.
    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
    formed = pd.DataFrame()
    i = 0
    cols = []
    with open(fp, "r") as f:
        for line in f:
            if line[-1] != '"':
                line = line[:-1]
            line = line.replace("\n", "").replace(",,", ",")
            items = line.split(",")
            if len(items) > 6:
                items = items[0:-1]
            if i == 0:
                for item in items:
                    formed[item] = 0
                cols = items
            else:
                geo = str(items[-2:])[1:-1]
                items = items[0:-2]
                geo = geo.replace('"', "").replace("'", "").replace(" ", "")
                items2 = []
                for item in items:
                    item = item.replace('"', "").replace("'", "")
                    items2.append(item)
                items = items2
                items.append(geo)
                df_add = pd.DataFrame(items, index=cols)
                df_add = df_add.transpose()
                formed = formed.append(df_add)
            i += 1
    formed = formed.reset_index().drop(columns=["index"])
    formed = formed.astype({"first": str, "last": str, "weight": float, "height": float, "geo": str})
    return formed