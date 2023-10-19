# Run these in venv terminal:
# $export NUMEXPR_NUM_THREADS=1
# $export OMP_NUM_THREADS=1

from .plotting import *
from .fileHandling import *
from .basicSolver import *
from .conformal import *
from .laplaceCoriolis import *
from .specialFunctions import *
from .timeSolver import *
from .rotationPDE import *


import dedalus
matplotlib.rcParams.update({'font.size': 16})
print('Using Dedalus v', dedalus.__version__)
