"""Gradient Descent Visualizer — entry point."""
from python.gdv_vis import runner


if __name__ == "__main__":
    try:
        runner_instance = runner.Runner()
        runner_instance.run()
    except KeyboardInterrupt:
        print("\n\nThe program ends!")
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")