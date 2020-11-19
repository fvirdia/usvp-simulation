"""
Example file to run experiments using the `experiments.py` module.
"""


from experiments import win_beta_distribution


if __name__ == "__main__":
    import warnings

    # ignore a DeprecationWarning for @parallel in Sagemath 9.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Progressive BKZ
        win_beta_distribution(1, 1, label="test-pbkz-experiments", parallel=True, progressive=True, progressive_skip=1)

        # BKZ
        # win_beta_distribution(25, 8, label="test-bkz-experiments", parallel=True, progressive=False)
