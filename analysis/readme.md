## 📊 Calculating Metrics and indicators  

Each experiment outputs set of raw records, which are then processed with the script in this folder for a set of performance indicators which we report and several additional metrics that track the quality of the solution and its impact to the system.

#### Usage

To use the analysis script, you have to provide in the command line the following command:

```bash
python analysis/metrics.py --id <exp_id> --verbose <verbose> --results-folder <results-folder> --skip-collecting <skip-collecting>
```

that will collect the results from the experiment with identifier ```<exp_id>``` and save them in the folder ```<exp_id>/metrics/```. The ```--verbose``` flag is optional and if set to ```True``` will print additional information about the analysis process. Flag ```--results-folder``` is optional and if set to ```True``` will use the folder ```<results-folder>``` instead of the default one ```results/```. The flag ```--skip-collecting``` is optional and if set to ```True``` will skip collecting the results from the experiment into single ```.csv``` file. This operation has to be done only once, so if you are running the analysis script more than once, you can skip this step.
To process every experiment under `results/`, use `python analysis/metrics.py --all` (skips runs that already have metrics unless you add `--no-skip`).

#### Reported indicators
---

The core metric is the travel time $t$, which is both the core term of the utility for human drivers (rational utility maximizers) and of the CAVs reward.
We report the average travel time for the system $\hat{t}$, human drivers $\hat{t}\_{HDV}$, and autonomous vehicles $\hat{t}\_{CAV}$. We record each during training, dynamic switching, and testing, plus a 50-day pre-mutation baseline ($\hat{t}^{train}, \hat{t}^{dyn}, \hat{t}^{test}, \hat{t}^{pre}$), with start/end windows for training and dynamic phases.

From these, we introduce:

- CAV advantage as $\hat{t}^{test}_{HDV} / \hat{t}^{test}_{CAV}$,
- Effect of changing to CAV as ${\hat{t}^{pre}_{HDV}}/{\hat{t}^{test}_{CAV}}$, and
- Effect of remaining HDV as ${\hat{t}^{pre}_{HDV}}/{\hat{t}^{test}_{HDV}}$.

To understand causes of travel time shifts, we track _Average speed_ and _Average mileage_ (from SUMO), plus dynamic-phase stability metrics:

- _Switch cost_ for all agents and per group: $\hat{t}^{dyn} - \hat{t}^{train}_{end}$, $\hat{t}^{dyn}_{HDV} - \hat{t}^{train}_{HDV,end}$, $\hat{t}^{dyn}_{CAV} - \hat{t}^{train}_{CAV,end}$.
- _Dynamic recovery_: $\hat{t}^{dyn}_{start} - \hat{t}^{dyn}_{end}$.
- _Dynamic volatility_: $\sigma(\hat{t}^{dyn})$ and group-specific versions.
- _Dynamic instability_: per-episode action-change rates for HDVs/CAVs.
- _Dynamic time excess_: average per-episode sum of time lost during switching.

We measure the _Cost of training_, expressed as the average of $\sum_{\tau \in train}(t^\tau_a - \hat{t}^{pre}_a)$ over all agents $a$, i.e. the cumulated disturbance CAVs cause during training. We define $c\_{CAV}$ and $c\_{HDV}$ accordingly. We call an experiment _won_ by CAVs if their policy was on average faster than human drivers' behaviour. A final _winrate_ is the share of training episodes where CAVs are faster.

Finally, we report switch statistics from `shifts.csv` (when available): total switches by direction, switches per event/agent, unique switch churn, and machine-ratio summary (start/end/avg/min/max/std).

#### Units of measurement
---

All the metrics are expressed in the following units:
- Time: minutes
- Distance: kilometers
- Speed: kilometers per hour
