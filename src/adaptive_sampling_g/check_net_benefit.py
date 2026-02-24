#!/usr/bin/env python3
import json,sys,statistics,collections

fn = sys.argv[1]
with open(fn) as f:
    j = json.load(f)

acc = j['per_snr_accuracy']    # dict of snr -> [a_e]
share = j['per_snr_share']     # dict of snr -> [w_e]

out = {}
for snr_str, acc_series in acc.items():
    acc_s = acc_series
    w_s = share[snr_str]
    n = len(acc_s)
    nets_when_dW_pos = []
    nets_all = []
    for i in range(n-2):   # we need e, e+1, e+2
        dW = w_s[i+1] - w_s[i]
        net = acc_s[i+2] - acc_s[i]    # net after one-epoch payoff (A_{e+2}-A_e)
        nets_all.append(net)
        if dW > 0:
            nets_when_dW_pos.append(net)
    def summarize(arr):
        if not arr:
            return {'count':0}
        return {'count':len(arr), 'mean': statistics.mean(arr), 'median': statistics.median(arr),
                'frac_pos': sum(1 for x in arr if x>0)/len(arr)}
    out[snr_str] = {'all': summarize(nets_all), 'when_dW_pos': summarize(nets_when_dW_pos)}

import pprint
pprint.pprint(out)
