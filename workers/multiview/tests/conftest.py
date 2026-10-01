import os
import sys

# Import multiview_worker and the vendored mvadapter from the worker folder without installing them
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
