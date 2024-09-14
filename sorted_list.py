#%%
import os
import sys
import torch as t
from pathlib import Path

# Make sure exercises are in the path
# chapter = r"chapter1_transformer_interp"
# exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
# section_dir = exercises_dir / "monthly_algorithmic_problems" / "october23_sorted_list"
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from dataset import SortedListDataset
from model import create_model
from plotly_utils import hist, bar, imshow

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
# %%
