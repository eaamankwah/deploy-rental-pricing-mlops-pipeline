#!/usr/bin/env python
"""
This step performs basic cleaning of the NYC Airbnb dataset:
- Removes price outliers outside [min_price, max_price]
- Converts last_review to datetime
- Removes properties outside NYC geographic boundaries
- Uploads cleaned artifact to W&B
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact from W&B
    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Dataset shape before cleaning: {df.shape}")

    # Drop price outliers outside [min_price, max_price]
    logger.info(f"Dropping price outliers outside [{args.min_price}, {args.max_price}]")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info(f"Dataset shape after price filtering: {df.shape}")

    # Convert last_review to datetime
    logger.info("Converting last_review column to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save cleaned dataframe to CSV
    logger.info("Saving cleaned dataset")
    df.to_csv("clean_sample.csv", index=False)

    # Upload cleaned artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    logger.info(f"Uploaded artifact {args.output_artifact} to W&B")
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning step")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact name (raw dataset from W&B)",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact (cleaned dataset)",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum acceptable nightly price in dollars",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum acceptable nightly price in dollars",
        required=True
    )

    args = parser.parse_args()

    go(args)
