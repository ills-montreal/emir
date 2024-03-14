import datasets
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime



dataset =datasets.load_dataset('tasksource/seahorse_summarization_evaluation')


_, val, test = dataset['train'], dataset['validation'], dataset['test']

# to pandas
val, test = val.to_pandas(), test.to_pandas()
# merge val and test
df = pd.concat([val, test])

# get list of unique models

