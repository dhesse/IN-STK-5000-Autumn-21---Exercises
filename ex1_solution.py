

def task1(input_list, n):

    """Return the n-th element of a list, or None if the list is shorter than n
    elements."""

    if len(input_list) <= n:
        return None
    return input_list[n]
