from experiments.experiment_runner import (
    run_all_experiments
)

from visualization.visualization_runner import (
    generate_all_visualizations
)


def main():
    
    # ====================================
    # RUN EXPERIMENTS
    # ====================================

    run_all_experiments()

    # ====================================
    # GENERATE VISUALIZATIONS
    # ====================================

    generate_all_visualizations()


if __name__ == "__main__":

    main()