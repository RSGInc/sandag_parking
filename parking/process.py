from .reduction import ReduceRawParkingData
from .imputation import ImputeParkingCosts
from .districts import CreateDistricts
from .estimate_spaces import EstimateStreetParking
from .expected_cost import ExpectedParkingCost

class ParkingProcessing(
    ReduceRawParkingData,
    ImputeParkingCosts,
    CreateDistricts,
    EstimateStreetParking,
    ExpectedParkingCost,
):
    def run_processing(self):
        
        # Runs models listed in settings.yaml
        for model_name in self.settings.get("models"):
            getattr(self, model_name)()
            
        # 1. Reduce/organize dataset and estimate model fit
        # 2. Impute missing values
        # 3. Find the parking districts
        # 4. Estimate spaces
        # 5. Calculate expected costs


if __name__ == "__main__":
    ParkingProcessing().run_processing()