import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve



seeds = [
  2297140167,
  406998175,
  2359957438,
  1069493441,
  3502404296,
  221703882,
  864257415,
  2411092152,
  2701685854,
  375368243
]

n_seeds = len(seeds)

dfs_wo_model = list()
dfs_with_model = list()


for s in seeds:
    df = pd.read_csv(f"outputs/2025-06-28/14-15-10/evaluation_data_seed_{s}_without_model.csv")
    df['seed'] = s
    dfs_wo_model.append(df)

    df = pd.read_csv(f"outputs/2025-06-28/14-20-43/evaluation_data_seed_{s}_with_model.csv")
    df['seed'] = s
    dfs_with_model.append(df)


steps = dfs_wo_model[0]['steps'].to_list()

df_wo_model = pd.concat(dfs_wo_model, ignore_index=True)
df_with_model = pd.concat(dfs_with_model, ignore_index=True)

train_scores = {
    "PPO":      df_wo_model["mean_returns"].to_numpy().reshape((n_seeds, -1)),
    "Dyna-PPO": df_with_model["mean_returns"].to_numpy().reshape((n_seeds, -1))
}



iqm = lambda scores: np.array(  # noqa: E731
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)

iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)

# avg = lambda scores: np.array(  # noqa: E731
#     [np.mean(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
# )
# 
# avg_scores, avg_cis = get_interval_estimates(
#     train_scores,
#     avg,
#     reps=2000,
# )


final_ppo_r = iqm_scores['PPO'][-1]

dyna_ppo_step = None
ppo_step = None
for i, step in enumerate(steps):
    dyna_ppo_r = iqm_scores['Dyna-PPO'][i]
    ppo_r = iqm_scores['PPO'][i]

    if dyna_ppo_step is None and dyna_ppo_r >= 0.8 * final_ppo_r:
        print("Dyna-PPO", step, dyna_ppo_r)
        dyna_ppo_step = step
    
    if ppo_step is None and ppo_r >= 0.8 * final_ppo_r:
        print("PPO", step, ppo_r)
        ppo_step = step

print("step diff", ppo_step - dyna_ppo_step)


plot_sample_efficiency_curve(
    steps,
    iqm_scores,
    iqm_cis,
    algorithms=["PPO", "Dyna-PPO"],
    xlabel="Real Steps",
    ylabel="Avg Return",
)

plt.legend()
plt.title("Plain PPO vs Dyna-PPO on CartPole-v1")
plt.tight_layout()
plt.savefig("rl_exercises/week_9/Level1_PPO_vs_Dyna-PPO.pdf")
plt.show()


