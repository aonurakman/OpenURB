# Open IPPO vs IQL (env config 1)

- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_ippo_1  --alg-conf config1 --env-seed 42 --torch-seed 42
- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_ippo_2  --alg-conf config2 --env-seed 42 --torch-seed 42
- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_ippo_3  --alg-conf config3 --env-seed 42 --torch-seed 42
- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_ippo_4  --alg-conf config4 --env-seed 42 --torch-seed 42
- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_ippo_5  --alg-conf config5 --env-seed 42 --torch-seed 42

- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_iql_1  --alg-conf config1 --env-seed 42 --torch-seed 42
- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_iql_2  --alg-conf config2 --env-seed 42 --torch-seed 42
- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_iql_3  --alg-conf config3 --env-seed 42 --torch-seed 42
- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_iql_4  --alg-conf config4 --env-seed 42 --torch-seed 42
- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config1 --id open_iql_5  --alg-conf config5 --env-seed 42 --torch-seed 42

## Additional folds

- python reproduce/repeat_exp.py --id open_ippo_1 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_1 --torch-seed 44

- python reproduce/repeat_exp.py --id open_ippo_2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_ippo_3 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_3 --torch-seed 44

- python reproduce/repeat_exp.py --id open_ippo_4 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_4 --torch-seed 44

- python reproduce/repeat_exp.py --id open_ippo_5 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_5 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_1 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_1 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_3 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_3 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_4 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_4 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_5 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_5 --torch-seed 44


# Open IPPO vs IQL (env config 2)

- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_ippo_1_env2  --alg-conf config1 --env-seed 42 --torch-seed 42
- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_ippo_2_env2  --alg-conf config2 --env-seed 42 --torch-seed 42
- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_ippo_3_env2  --alg-conf config3 --env-seed 42 --torch-seed 42
- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_ippo_4_env2  --alg-conf config4 --env-seed 42 --torch-seed 42
- python ./scripts/open_ippo.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_ippo_5_env2  --alg-conf config5 --env-seed 42 --torch-seed 42

- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_iql_1_env2  --alg-conf config1 --env-seed 42 --torch-seed 42
- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_iql_2_env2  --alg-conf config2 --env-seed 42 --torch-seed 42
- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_iql_3_env2  --alg-conf config3 --env-seed 42 --torch-seed 42
- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_iql_4_env2  --alg-conf config4 --env-seed 42 --torch-seed 42
- python ./scripts/open_iql.py --net ing_small --task-conf dynamic1 --env-conf config2 --id open_iql_5_env2  --alg-conf config5 --env-seed 42 --torch-seed 42

## Additional folds

- python reproduce/repeat_exp.py --id open_ippo_1_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_1_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_ippo_2_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_2_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_ippo_3_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_3_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_ippo_4_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_4_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_ippo_5_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_ippo_5_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_1_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_1_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_2_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_2_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_3_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_3_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_4_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_4_env2 --torch-seed 44

- python reproduce/repeat_exp.py --id open_iql_5_env2 --torch-seed 43
- python reproduce/repeat_exp.py --id open_iql_5_env2 --torch-seed 44
