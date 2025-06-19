import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="experiment", version_base=None)
def print_experiment(cfg: DictConfig):
    print("=== Experiment Configuration ===")
    for key, value in cfg.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    print_experiment()
