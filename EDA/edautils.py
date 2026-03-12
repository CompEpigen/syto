import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from typing import List, Literal, Optional, Callable, Union, Dict, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from scipy import stats
import warnings
import pysam
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import re
from matplotlib.gridspec import GridSpec
import os
from sklearn.metrics import confusion_matrix










