#for tf_method in ['MTF', 'STF']:
for tf_method in ['MTF-2']:
    for freq in ['1H']:
        for r in range(1, 11):
            print("python analysis.py {} {} {} &> log-{}-{}-{}.out &".format(tf_method, freq, r, tf_method, freq, r))
