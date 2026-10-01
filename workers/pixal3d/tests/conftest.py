import os
import sys

# Import pixal3d_worker, and the TRELLIS.2 worker's forge3d_worker it builds on, without installing them
HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.dirname(HERE)
sys.path.insert(0, WORKER)
sys.path.insert(0, os.path.join(os.path.dirname(WORKER), "trellis2"))
