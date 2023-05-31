import os
from .process import ParkingProcessing

print("DIRECTORY: " + os.getcwd())

ParkingProcessing().run_processing()