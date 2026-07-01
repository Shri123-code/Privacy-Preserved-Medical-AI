import flwr as fl


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=4,         
    min_available_clients=4,
    min_evaluate_clients=2,
)

print("📡 Central Server starting on port 8080...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)