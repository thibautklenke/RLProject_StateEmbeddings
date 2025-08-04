from tbparse import SummaryReader
import matplotlib.pyplot as plt

def graphs() -> None:
    log_dir = "./logs/MlpPolicy/"
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    df = reader.scalars
    print(df[df["tag"] == "eval/mean_reward"])

    df[df["tag"] == "eval/mean_reward"].pivot(index="step", columns="dir_name", values="value").plot()

    fig = plt.gcf()
    fig.savefig('output.png')


if __name__ == "__main__":
    graphs()