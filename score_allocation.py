def scores(target, revenue, ratings):
    tar_sc = {}
    tar_n = {}
    for i1 in range(0,len(target)):
        i = target[i1]
        j = revenue[i1]
        k = float(ratings[i1])
        if i == 'Someone':
            tar_sc[i] = 0
            continue
        k /= 10;
        if i not in list(tar_sc.keys()):
            tar_sc[i] = j*k
            tar_n[i] = 1
        else:
            tar_sc[i] = (tar_sc[i]*tar_n[i] + j*k)/(tar_n[i] + 1)
            tar_n[i] += 1
    return tar_sc
        