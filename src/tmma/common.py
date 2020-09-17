import pandas as pd

def validate_series_indices(*multiple_series):
    """
    Validates indices of provided series

    :param multiple_series:
    :return:
    """

    # No `pd.Series` provided, nothing to do
    if not any(map(lambda x: isinstance(x, pd.Series), multiple_series)):
        return None

    # If one Series was given then all should be series
    if not all(map(lambda x: isinstance(x, pd.Series), multiple_series)):
        raise ValueError("You must provide either all `pd.Series` or all numpy arrays. "
                         "But not mix and match.")

    # Verify that indices are equal
    index = None
    for series in multiple_series:
        if index is None:
            index = series.index
            continue
        else:
            if not series.index.equals(index):
                raise ValueError("Indices do not match")

    # If we're at this point then all ok, return index
    return index
