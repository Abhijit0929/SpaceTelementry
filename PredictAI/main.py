import sys
import argparse
from src import train, predict, threshold, detector, evaluate, visualize


def main():
    parser = argparse.ArgumentParser(description="Spacecraft Intelligence Pipeline Runner")
    parser.add_argument("channel", nargs="?", default="A-1", help="Target telemetry channel (default: A-1)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run inference with existing model")
    args = parser.parse_args()

    channel = args.channel

    print("=" * 60)
    print("SPACECRAFT INTELLIGENCE PIPELINE RUNNER")
    print("=" * 60)
    print(f"Target Channel: {channel}")

    # 1. Training
    if not args.skip_train:
        print("\n[Step 1/6] Running Model Training...")
        trainer = train.Trainer(channel)
        trainer.train()
    else:
        print("\n[Step 1/6] Skipping Model Training (using existing trained weights)...")

    # 2. Prediction (Reconstruction Errors)
    print("\n[Step 2/6] Running Prediction / Inference...")
    predict.main(channel)

    # 3. Anomaly Threshold
    print("\n[Step 3/6] Computing Anomaly Threshold...")
    threshold.main(channel)

    # 4. Anomaly Detection
    print("\n[Step 4/6] Running Anomaly Detector...")
    detector.main(channel)

    # 5. Evaluation
    print("\n[Step 5/6] Evaluating Anomaly Detection...")
    evaluate.main(channel)

    # 6. Visualization
    print("\n[Step 6/6] Generating Anomaly Visualizations...")
    visualize.main(channel)

    print("\n" + "=" * 60)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
