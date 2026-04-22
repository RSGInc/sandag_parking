from parking.process import ParkingProcessing
import argparse

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    ParkingProcessing(settings_file=args.config).run_processing()