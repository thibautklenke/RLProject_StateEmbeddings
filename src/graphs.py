import numpy as np
import pandas as pd
from tbparse import SummaryReader
import matplotlib.pyplot as plt
import seaborn as sns


def graphs() -> None:
    log_dir = "./logs/MlpPolicy/"
    reader = SummaryReader(log_dir, extra_columns={"dir_name"})
    df = reader.scalars

    df["ALG_ENV"] = df["dir_name"].apply(lambda x: "-".join(x.split("-")[0:2]))
    algs_envs = df["ALG_ENV"].unique()

    def percentile5(x: pd.Series) -> pd.DataFrame:
        return np.percentile(x, 5)

    def percentile95(x: pd.Series) -> pd.DataFrame:
        return np.percentile(x, 95)

    df = df[df["tag"] == "eval/mean_reward"].pivot_table(
        index="step",
        columns="ALG_ENV",
        values="value",
        aggfunc=[np.median, percentile5, percentile95],
    )

    # Start plot
    plt.figure(figsize=(8, 5))

    # Get color palette
    palette = sns.color_palette("tab10", len(algs_envs))

    # Loop through each group
    for i, group in enumerate(algs_envs):
        # Plot line using seaborn
        sns.lineplot(x=df.index, y=df["median"][group], label=group, color=palette[i])

        # Add error bounds
        plt.fill_between(
            df.index,
            df["percentile5"][group],
            df["percentile95"][group],
            alpha=0.2,
            color=palette[i],
        )

    # Final touches
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Title")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title="Algorithms")
    plt.show()

    plt.savefig("output.png")


if __name__ == "__main__":
    graphs()
