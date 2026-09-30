import os
import sys

# Import job_api and modal_app from the workers folder without installing them
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
