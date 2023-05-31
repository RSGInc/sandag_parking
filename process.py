from parking.reduction import ReduceRawParkingData
from parking.imputation import ImputeParkingCosts
from parking.districts import CreateDistricts
from parking.estimate_spaces import EstimateStreetParking
from parking.expected_cost import ExpectedParkingCost


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

if __name__ == "__main__":
    ParkingProcessing().run_processing()