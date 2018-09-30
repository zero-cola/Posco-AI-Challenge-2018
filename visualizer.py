import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(**kwargs):
    metrics_dict = {}
    for c0 in kwargs:
        for c1 in kwargs[c0]:
            metrics_dict[c0 + '_' + c1] = kwargs[c0][c1]
    df = pd.DataFrame(metrics_dict)
    df.index.name='epoch'
    df_2 = df.loc[:, df.columns.str.contains('score')].copy()
    df_1 = df.drop(columns=df_2.columns)

    if len(df_1.columns) != 0:
        ax = df_1.plot(xticks=range(len(df_1)),grid=True)
        df_2.plot(ax=ax, style=':', secondary_y=True, xticks=range(len(df_1)), grid=True)
    else:
        ax = df_2.plot(xticks=range(len(df_2)),grid=True)

    plt.show()
