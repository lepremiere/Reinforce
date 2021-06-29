import requests
import pandas as pd
import numpy as np
import polars as pl

df = pl.read_csv("D:/TS/SP500_M5.csv").to_pandas()
