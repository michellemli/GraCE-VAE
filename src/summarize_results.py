# Summarize results

import os
import ast
import numpy as np


results_dir = "result/model15_seed"

results = {"Single Evaluation": dict(), "Double Evaluation": dict()}
for i in range(1, 11, 1):
    f = results_dir + str(i) + ("/model15_seed%d.txt" % i)
    print(f)
    assert os.path.exists(f)
    with open(f, "r") as fin:
        for line in fin:
            line = line.strip()
            label, _, res = line.partition(":")
            label = label.strip()
            parsed = ast.literal_eval(res)
            assert label in results
            for k, v in parsed.items():
                if k not in results[label]: results[label][k] = []
                results[label][k].append(v["DE Genes"])

for k in ["Single Evaluation", "Double Evaluation"]:
    print(k)
    for metric in results[k]:
        print(metric, np.mean(results[k][metric]), np.std(results[k][metric]))

