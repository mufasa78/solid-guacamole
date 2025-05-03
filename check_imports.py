try:
    import jieba
    import matplotlib
    import numpy
    import pandas
    import sklearn
    import seaborn
    import streamlit
    import torch
    print("All packages imported successfully!")
except ImportError as e:
    print(f"Error importing packages: {e}")
