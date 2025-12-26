import pandas as pd
import os


import llm_clients
import loaders
import metrics_calculators

tables_dict = loaders.load_excel_all_sheets("../data/Benchmark_Questions.xlsx")
science, argumentation, ethics, histoire = tables_dict.values()

print(science.head(5))

