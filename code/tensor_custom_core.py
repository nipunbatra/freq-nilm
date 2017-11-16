import autograd.numpy as np
from autograd import multigrad
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


cases = {
    1: {'HA': 'Ma, Nb -> MNab', 'HAT': 'MNab, Oab -> MNO'},
    2: {'HA': 'Ma, Nab -> MNb', 'HAT': 'MNb, Ob -> MNO'},
    3: {'HA': 'Mab, Na -> MNb', 'HAT': 'MNb, Ob -> MNO'},
    4: {'HA': 'Ma, Na -> MNa', 'HAT': 'MNa, Oa -> MNO'}
}


def multiply_case(H, A, T, case):
    HA = np.einsum(cases[case]['HA'], H, A)
    HAT = np.einsum(cases[case]['HAT'], HA, T)
    return HAT


def cost_abs(H, A, T, E_np_masked, case):
    HAT = multiply_case(H, A, T, case)
    mask = ~np.isnan(E_np_masked)
    error = (HAT - E_np_masked)[mask].flatten()
    return np.sqrt((error ** 2).mean())

def cost_fraction(H, A, T, E_np_masked, case):
    HAT = multiply_case(H, A, T, case)
    num_appliances = len(A)-1
    c = 0
    for appliance_num in range(1, num_appliances + 1):
        gt_appliance_fr = E_np_masked[:, appliance_num, :] / E_np_masked[:, 0, :]
        pred_appliance_fr = HAT[:, appliance_num, :] / E_np_masked[:, 0, :]
        diff_appliance_fr = (pred_appliance_fr - gt_appliance_fr).flatten()
        diff_appliance_fr = diff_appliance_fr[~np.isnan(diff_appliance_fr)]
        c = c + np.sqrt(np.square(diff_appliance_fr).mean())
    return c


def cost_rel_per(H, A, T, E_np_masked, case):
    HAT = multiply_case(H, A, T, case)
    mask = ~np.isnan(E_np_masked)
    error = 100*(HAT - E_np_masked)[mask].flatten() / (1 + E_np_masked[mask].flatten())
    return np.sqrt((error ** 2).mean())



def cost_rel(H, A, T, E_np_masked, case):
    HAT = multiply_case(H, A, T, case)
    mask = ~np.isnan(E_np_masked)
    error = (HAT - E_np_masked)[mask].flatten() / (1 + E_np_masked[mask].flatten())
    return np.sqrt((error ** 2).mean())


def set_known(A, W):
    mask = ~np.isnan(W)
    A[:, :mask.shape[1]][mask] = W[mask]
    return A


def distance(x, y):
    return np.linalg.norm(x - y)




def learn_HBAT_adagrad_graph(tensor, num_home_factors, num_season_factors, num_iter=2000, lr=0.01, dis=False,
                             random_seed=0, eps=1e-8, B_known=None, A_known=None, T_known=None):


    def multiply_HBAT(H, B, A, T):
        return np.einsum('mh, hn, ns, ts ->mnt', H, B, A, T)

    def cost(H, B, A, T, tensor):
        mask = ~np.isnan(tensor)
        HBAT = multiply_HBAT(H, B, A, T)
        error = (HBAT - tensor)[mask].flatten()
        return np.sqrt((error ** 2).mean())

    np.random.seed(random_seed)


    args_num = [0, 1, 2, 3]
    mg = multigrad(cost, argnums=args_num)

    m, n, t = tensor.shape
    h, s = num_home_factors, num_season_factors

    H = np.random.rand(m, h)
    B = np.random.rand(h, n)
    A = np.random.rand(n, s)
    T = np.random.rand(t, s)

    if A_known is not None:
        A = set_known(A, A_known)
    if B_known is not None:
        B = set_known(B, B_known)
    sum_square_gradients_A = np.zeros_like(A)
    sum_square_gradients_B = np.zeros_like(B)
    sum_square_gradients_H = np.zeros_like(H)
    sum_square_gradients_T = np.zeros_like(T)
    Hs = [H.copy()]
    Bs = [B.copy()]
    Ts = [T.copy()]
    As = [A.copy()]
    HBATs = [multiply_HBAT(H, B, A, T)]
    costs = [cost(H, B, A, T, tensor)]

    # GD procedure
    for i in range(num_iter):
        del_h, del_b, del_a, del_t = mg(H, B, A, T,  tensor)

        sum_square_gradients_A += eps + np.square(del_a)
        lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
        A -= lr_a * del_a

        sum_square_gradients_H += eps + np.square(del_h)
        sum_square_gradients_B += eps + np.square(del_b)
        sum_square_gradients_T += eps + np.square(del_t)

        lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
        lr_t = np.divide(lr, np.sqrt(sum_square_gradients_T))

        H -= lr_h * del_h
        T -= lr_t * del_t


        if A_known is not None:
            A = set_known(A, A_known)
        if B_known is not None:
            B = set_known(B, B_known)
        if T_known is not None:
            T = set_known(T, T_known)

        # Projection to non-negative space
        H[H < 0] = 1e-8
        A[A < 0] = 1e-8
        T[T < 0] = 1e-8
        B[B<0] = 1e-8

        As.append(A.copy())
        Ts.append(T.copy())
        Hs.append(H.copy())
        Bs.append(B.copy())

        costs.append(cost(H, B, A, T,  tensor))

        HBATs.append(multiply_HBAT(H, B, A, T))
        if i % 100 == 0:
            if dis:
                print(cost(H, B, A, T, tensor))

    return H, B, A, T, Hs, Bs, As, Ts, HBATs, costs


def stf_4dim(tensor, r, random_seed=0, num_iter=100, eps=1e-8, lr=1):
    np.random.seed(random_seed)
    args_num = [1, 2, 3, 4]

    def cost(tensor, home, appliance, day, hour):
        pred = np.einsum('Hr, Ar, Dr, Tr ->HADT', home, appliance, day, hour)
        mask = ~np.isnan(tensor)
        error = (pred - tensor)[mask].flatten()
        return np.sqrt((error ** 2).mean())

    mg = multigrad(cost, argnums=args_num)
    sizes = [(x, r) for x in tensor.shape]
    home = np.random.rand(*sizes[0])
    appliance = np.random.rand(*sizes[1])
    day = np.random.rand(*sizes[2])
    hour = np.random.rand(*sizes[3])

    sum_home = np.zeros_like(home)
    sum_appliance = np.zeros_like(appliance)
    sum_day = np.zeros_like(day)
    sum_hour = np.zeros_like(hour)

    # GD procedure
    for i in range(num_iter):
        del_home, del_appliance, del_day, del_hour = mg(tensor, home, appliance, day, hour)

        sum_home += eps + np.square(del_home)
        lr_home = np.divide(lr, np.sqrt(sum_home))
        home -= lr_home * del_home

        sum_appliance += eps + np.square(del_appliance)
        lr_appliance = np.divide(lr, np.sqrt(sum_appliance))
        appliance -= lr_appliance * del_appliance

        sum_day += eps + np.square(del_day)
        lr_day = np.divide(lr, np.sqrt(sum_day))
        day -= lr_day * del_day

        sum_hour += eps + np.square(del_hour)
        lr_hour = np.divide(lr, np.sqrt(sum_hour))
        hour -= lr_hour * del_hour
        
        


        # Projection to non-negative space
        home[home <0] = 1e-8
        appliance[appliance < 0] = 1e-8
        day[day < 0] = 1e-8
        hour[hour < 0] = 1e-8

        if i%50==0:
            print cost(tensor, home, appliance, day, hour), i




    return home, appliance, day, hour

def learn_HAT_adagrad_graph(case, tensor, L, num_home_factors, num_season_factors, num_iter=2000, lr=0.01, dis=False,
                            lam=1, random_seed=0, eps=1e-8, A_known = None, T_known = None):
    np.random.seed(random_seed)
    cost = cost_abs
    
    args_num=[0,1,2]
    mg = multigrad(cost, argnums=args_num)

    params = {}
    params['M'], params['N'], params['O'] = tensor.shape
    params['a'] = num_home_factors
    params['b'] = num_season_factors
    H_dim_chars = list(cases[case]['HA'].split(",")[0].strip())
    H_dim = tuple(params[x] for x in H_dim_chars)
    A_dim_chars = list(cases[case]['HA'].split(",")[1].split("-")[0].strip())
    A_dim = tuple(params[x] for x in A_dim_chars)
    T_dim_chars = list(cases[case]['HAT'].split(",")[1].split("-")[0].strip())
    T_dim = tuple(params[x] for x in T_dim_chars)

    H = np.random.rand(*H_dim)
    A = np.random.rand(*A_dim)
    T = np.random.rand(*T_dim)

    if A_known is not None:
        A = set_known(A, A_known)
    sum_square_gradients_A = np.zeros_like(A)
    sum_square_gradients_H = np.zeros_like(H)
    sum_square_gradients_T = np.zeros_like(T)
    Hs = [H.copy()]
    Ts = [T.copy()]
    As = [A.copy()]
    costs = [cost(H, A, T, L, tensor, lam, case)]
    HATs = [multiply_case(H, A, T, case)]

    # GD procedure
    for i in range(num_iter):
        del_h, del_a, del_t = mg(H, A, T, L, tensor, lam, case)
        
        sum_square_gradients_A += eps + np.square(del_a)
        lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
        A -= lr_a * del_a
        
        sum_square_gradients_H += eps + np.square(del_h)
        sum_square_gradients_T += eps + np.square(del_t)

        lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
        lr_t = np.divide(lr, np.sqrt(sum_square_gradients_T))

        H -= lr_h * del_h
        T -= lr_t * del_t

        if T_known is not None:
            T = set_known(T, T_known)
        if A_known is not None:
            A = set_known(A, A_known)

        # Projection to non-negative space
        H[H < 0] = 1e-8
        A[A < 0] = 1e-8
        T[T < 0] = 1e-8

        As.append(A.copy())
        Ts.append(T.copy())
        Hs.append(H.copy())

        costs.append(cost(H, A, T, L, tensor, lam, case))

        HATs.append(multiply_case(H, A, T, case))
        if i % 500 == 0:
            if dis:
                print(cost(H, A, T, L, tensor, lam, case))
    
    return H, A, T, Hs, As, Ts, HATs, costs


def learn_HAT_multiple_source_adagrad(case, source_1_energy, source_2_energy, a, b, num_iter=2000, lr=0.1, dis=False,  H_known_s1=None,
                      A_known=None, T_known_s1=None, H_known_s2=None, T_known_s2 = None,
                        random_seed=0, eps=1e-8, penalty_coeff=0.0, source_ratio=0.5):





    def cost_l21(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, case, source_ratio, lam=0.1):
        HAT_s1 = multiply_case(H_s1, A, T_s1, case)
        HAT_s2 = multiply_case(H_s2, A, T_s2, case)
        mask_s1 = ~np.isnan(source_1_energy)
        mask_s2 = ~np.isnan(source_2_energy)
        error_s1 = (HAT_s1 - source_1_energy)[mask_s1].flatten()
        error_s2 = (HAT_s2 - source_2_energy)[mask_s2].flatten()
        A_shape = A.shape
        A_flat = A.reshape(A_shape[0], A_shape[1]*A_shape[2])
        l1 = 0.
        for j in range(A_shape[1]*A_shape[2]):
            l1 = l1 + np.sqrt(np.square(A_flat[:, j]).sum())
        # return np.sqrt((error ** 2).mean()) + lam*np.sum(A[A!=0])
        return source_ratio*np.sqrt((error_s1 ** 2).mean()) + (1-source_ratio)*np.sqrt((error_s2 ** 2).mean())+ lam * l1



    np.random.seed(random_seed)

    cost = cost_l21


    mg = multigrad(cost, argnums=[0, 1, 2, 3, 4])

    params_s1 = {}
    params_s1['M'], params_s1['N'], params_s1['O'] = source_1_energy.shape
    params_s1['a'] = a
    params_s1['b'] = b

    params_s2 = {}
    params_s2['M'], params_s2['N'], params_s2['O'] = source_2_energy.shape
    params_s2['a'] = a
    params_s2['b'] = b

    H_dim_chars = list(cases[case]['HA'].split(",")[0].strip())
    H_dim_s1 = tuple(params_s1[x] for x in H_dim_chars)
    H_dim_s2 = tuple(params_s2[x] for x in H_dim_chars)

    A_dim_chars = list(cases[case]['HA'].split(",")[1].split("-")[0].strip())
    A_dim_s1 = tuple(params_s1[x] for x in A_dim_chars)
    A_dim_s2 = tuple(params_s2[x] for x in A_dim_chars)

    T_dim_chars = list(cases[case]['HAT'].split(",")[1].split("-")[0].strip())
    T_dim_s1 = tuple(params_s1[x] for x in T_dim_chars)
    T_dim_s2 = tuple(params_s2[x] for x in T_dim_chars)


    H_s1 = np.random.rand(*H_dim_s1)
    H_s2 = np.random.rand(*H_dim_s2)

    A = np.random.rand(*A_dim_s1)
    T_s1 = np.random.rand(*T_dim_s1)
    T_s2 = np.random.rand(*T_dim_s2)

    sum_square_gradients_H_s1 = np.zeros_like(H_s1)
    sum_square_gradients_H_s2 = np.zeros_like(H_s2)

    sum_square_gradients_A = np.zeros_like(A)

    sum_square_gradients_T_s1 = np.zeros_like(T_s1)
    sum_square_gradients_T_s2 = np.zeros_like(T_s2)


    Hs_s1 = [H_s1.copy()]
    Hs_s2 = [H_s2.copy()]
    As = [A.copy()]
    Ts_s1 = [T_s1.copy()]
    Ts_s2 = [T_s2.copy()]
    costs = [cost(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, 2,  penalty_coeff, source_ratio)]
    HATs_s1 = [multiply_case(H_s1, A, T_s1, 2)]
    HATs_s2 = [multiply_case(H_s2, A, T_s2, 2)]

    # GD procedure
    for i in range(num_iter):
        del_h_s1, del_a, del_t_s1, del_h_s2, del_t_s2 = mg(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, case, source_ratio, penalty_coeff)
        sum_square_gradients_H_s1 += eps + np.square(del_h_s1)
        sum_square_gradients_H_s2 += eps + np.square(del_h_s2)

        sum_square_gradients_A += eps + np.square(del_a)

        sum_square_gradients_T_s1 += eps + np.square(del_t_s1)
        sum_square_gradients_T_s2 += eps + np.square(del_t_s2)

        lr_h_s1 = np.divide(lr, np.sqrt(sum_square_gradients_H_s1))
        lr_h_s2 = np.divide(lr, np.sqrt(sum_square_gradients_H_s2))

        lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))

        lr_t_s1 = np.divide(lr, np.sqrt(sum_square_gradients_T_s1))
        lr_t_s2 = np.divide(lr, np.sqrt(sum_square_gradients_T_s2))

        H_s1 -= lr_h_s1 * del_h_s1
        H_s2 -= lr_h_s2 * del_h_s2
        A -= lr_a * del_a

        T_s1 -= lr_t_s1 * del_t_s1
        T_s2 -= lr_t_s2 * del_t_s2
        # Projection to known values
        if H_known_s1 is not None:
            H_s1 = set_known(H_s1, H_known_s1)
        if H_known_s2 is not None:
            H_s2 = set_known(H_s2, H_known_s2)
        if A_known is not None:
            A = set_known(A, A_known)
        if T_known_s1 is not None:
            T = set_known(T, T_known_s1)
        if T_known_s2 is not None:
            T = set_known(T, T_known_s2)
        # Projection to non-negative space
        H_s1[H_s1 < 0] = 1e-8
        H_s2[H_s2 < 0] = 1e-8
        A[A < 0] = 1e-8
        T_s1[T_s1 < 0] = 1e-8
        T_s2[T_s2 < 0] = 1e-8

        As.append(A.copy())
        Ts_s1.append(T_s1.copy())
        Ts_s2.append(T_s2.copy())
        Hs_s1.append(H_s1.copy())
        Hs_s2.append(H_s2.copy())
        costs.append(cost(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, 2,  penalty_coeff,source_ratio))
        HATs_s1.append(multiply_case(H_s1, A, T_s1, 2))
        HATs_s2.append(multiply_case(H_s2, A, T_s2, 2))
        if i % 500 == 0:
            if dis:
                print(cost(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, 2,  penalty_coeff, source_ratio))
    return H_s1, A, T_s1, H_s2, T_s2, Hs_s1, As, Ts_s1, Hs_s2,  Ts_s2, HATs_s1, HATs_s2, costs

def learn_HAT_adagrad_graph_old(case, tensor, L, num_home_factors, num_season_factors, num_iter=2000, lr=0.01, dis=False,
                            lam=1, random_seed=0, eps=1e-8, A_known = None, T_known = None):
    np.random.seed(random_seed)
    cost = cost_graph_laplacian
    if A_known is not None:
        # Don't need to learn A
        args_num = [0, 2]
    else:
        args_num = [0, 1, 2]
    mg = multigrad(cost, argnums=args_num)

    params = {}
    params['M'], params['N'], params['O'] = tensor.shape
    params['a'] = num_home_factors
    params['b'] = num_season_factors
    H_dim_chars = list(cases[case]['HA'].split(",")[0].strip())
    H_dim = tuple(params[x] for x in H_dim_chars)
    A_dim_chars = list(cases[case]['HA'].split(",")[1].split("-")[0].strip())
    A_dim = tuple(params[x] for x in A_dim_chars)
    T_dim_chars = list(cases[case]['HAT'].split(",")[1].split("-")[0].strip())
    T_dim = tuple(params[x] for x in T_dim_chars)

    H = np.random.rand(*H_dim)
    
    if A_known is not None:
        A = A_known
    else:
        A = np.random.rand(*A_dim)
        sum_square_gradients_A = np.zeros_like(A)
    
    T = np.random.rand(*T_dim)
    
    sum_square_gradients_H = np.zeros_like(H)
    sum_square_gradients_T = np.zeros_like(T)
    Hs = [H.copy()]
    Ts = [T.copy()]
    As = [A.copy()]
    costs = [cost(H, A, T, L, tensor, lam, case)]
    HATs = [multiply_case(H, A, T, case)]

    # GD procedure
    for i in range(num_iter):
        if A_known is not None:
            del_h,  del_t = mg(H, A, T, L, tensor, lam, case)
        else:
            del_h, del_a, del_t = mg(H, A, T, L, tensor, lam, case)
            sum_square_gradients_A += eps + np.square(del_a)
            lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
            A -= lr_a * del_a
        
        sum_square_gradients_H += eps + np.square(del_h)
        sum_square_gradients_T += eps + np.square(del_t)

        lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
        lr_t = np.divide(lr, np.sqrt(sum_square_gradients_T))

        H -= lr_h * del_h
        T -= lr_t * del_t

        if T_known is not None:
            T = set_known(T, T_known)

        # Projection to non-negative space
        H[H < 0] = 1e-8
        A[A < 0] = 1e-8
        T[T < 0] = 1e-8

        As.append(A.copy())
        Ts.append(T.copy())
        Hs.append(H.copy())

        costs.append(cost(H, A, T, L, tensor, lam, case))

        HATs.append(multiply_case(H, A, T, case))
        if i % 500 == 0:
            if dis:
                print(cost(H, A, T, L, tensor, lam, case))
    return H, A, T, Hs, As, Ts, HATs, costs
