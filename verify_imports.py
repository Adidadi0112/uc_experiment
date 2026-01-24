try:
    import catboost
    import fancyimpute
    import imblearn
    import matplotlib
    import numpy
    import pandas
    import sdv
    import seaborn
    import sklearn
    import torch
    print("All imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
