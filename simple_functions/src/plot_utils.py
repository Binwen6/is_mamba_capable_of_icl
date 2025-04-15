import os

import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

# sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "gkr_gpt2_fast": [
        # "S4",
        # "Mamba",
        'Transformer', 
        '3-Nearest Neighbors', 
        'Averaging'
    ]
    # "linear_regression": [
    #     "S4",
    #     "Mamba",
    #     "Transformer",
    #     "Least Squares",
    #     # "3-Nearest Neighbors",
    #     # "Averaging",
    # ],
    # "sparse_linear_regression": [
    #     "S4",
    #     "Mamba",
    #     "Transformer",
    #     "Least Squares",
    #     # "3-Nearest Neighbors",
    #     # "Averaging",
    #     "Lasso (alpha=0.01)",
    # ],
    # "decision_tree": [
    #     "S4",
    #     "Mamba",
    #     "Transformer",
    #     # "3-Nearest Neighbors",
    #     # "2-layer NN, GD",
    #     # "Greedy Tree Learning",
    #     "XGBoost",
    # ],
    # "relu_2nn_regression": [
    #     "S4",
    #     "Mamba",
    #     "Transformer",
    #     "Least Squares",
    #     # "3-Nearest Neighbors",
    #     "2-layer NN, GD",
    # ],
    # "gaussian_kernel_regression": [
    #     "S4",
    #     "Mamba",
    #     "Transformer",
    #     # "3-Nearest Neighbors",
    #     # "Averaging",
    # ],
    # "nonlinear_dynamical_system": [
    #     "S4",
    #     "Mamba",
    #     "Transformer",
    #     # "3-Nearest Neighbors",
    #     # "Averaging",
    # ]
}


def basic_plot(metrics, size, models=None, trivial=1.0, title=None, n_train_points=None, size_mult=1.):
    fig, ax = plt.subplots(1, 1, figsize=size)

    if models is not None:
        metrics = {k: metrics[k] for k in models}

    ax.set_title(title)
    color_index = 0
    ax.axhline(trivial, ls="--", color="gray")
    max_x = 0
    color_match = dict()
    for name, vs in metrics.items():
        if name.split('_')[0] in color_match.keys(): 
            color = color_match[name.split('_')[0]]
        else:
            color = palette[color_index % 10]
            color_match[name.split('_')[0]] = color

        ax.plot(vs["mean"], "-", label=name, color=color, lw=2, zorder=100-color_index)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        max_x = max((len(vs["mean"]), max_x))
        color_index += 1
    
    if n_train_points is not None:
        plt.axvline(n_train_points-1, linestyle="dotted", color="black")

    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-1, max_x + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    legend.set_zorder(102)
    fig.set_size_inches(size_mult*4, size_mult*3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None, xlim=None,
                    no_recompute=False):
    all_metrics = {}

    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True, no_recompute=no_recompute)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}

            for model_name, m in results.items():
                original_model_name = model_name

                # 保留原始名字（用于 rename_model 阶段）
                if "gpt2" in model_name or "mamba" in model_name or "s4" in model_name:
                    pass  # 不修改
                else:
                    model_name = baseline_names(model_name)

                m_processed = {}
                n_dims = conf.model.n_dims

                # 自动设置 xlim（只设置一次）
                local_xlim = xlim
                if local_xlim is None:
                    if r.task in ["relu_2nn_regression", "decision_tree"]:
                        local_xlim = 200 
                    else:
                        local_xlim = 2 * n_dims + 1

                # loss 归一化（例如除以维度）
                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:local_xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v

                processed_results[original_model_name] = m_processed  # ✅ 保留原始 key
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)

    return all_metrics
