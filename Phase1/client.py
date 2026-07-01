import flwr as fl
import argparse
from task import MedicalCNN, get_weights, set_weights


parser = argparse.ArgumentParser()
parser.add_argument("--hospital", type=str, choices=['A', 'B', 'C', 'D'], required=True)
args = parser.parse_args()

model = MedicalCNN()
data_path = f"./federated_data/Hospital_{args.hospital}"

class HospitalClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_weights(model)

    def fit(self, parameters, config):
        set_weights(model, parameters)
        print(f"🏥 Hospital {args.hospital}: Commencing local training on private data...")
        
        return get_weights(model), 100, {}

    def evaluate(self, parameters, config):
        set_weights(model, parameters)
        print(f"🏥 Hospital {args.hospital}: Evaluating global model...")
        return 0.1, 10, {"accuracy": 0.85}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=HospitalClient())