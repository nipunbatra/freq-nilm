import numpy as np
import sys


def reshape_for_sc(tensor):
    """

    :param tensor: M homes, A appliance, D days, 24 hours
    :return: flattened across the D X 24 dimension to give A X (D X 24) X M
    """
    return tensor.reshape(-1, tensor.shape[1], tensor.shape[2]*tensor.shape[3]).swapaxes(1, 2)

def nnls_autograd(A, B, num_iter=300, eps=1e-8, lr=4):
    """
    Ax = B
    """

    import autograd.numpy as np
    np.random.seed(0)
    from autograd import grad

    def cost(A, B, x):
        #error = A@x - B
        error = np.dot(A, x) - B
        return np.sqrt((error ** 2).mean())

    mg = grad(cost, argnum=2)

    M, N = B.shape
    r = A.shape[1]
    x = np.abs(np.random.randn(r, N))
   
    sum_x = np.zeros_like(x)

    for i in range(num_iter):
        del_x = mg(A, B, x)
        sum_x += eps + np.square(del_x)
        lr_x = np.divide(lr, np.sqrt(sum_x))
        x -= lr_x * del_x

        x[x < 0] = 0.

        if i % 10 == 0:
            # print(cost(A, B, x), i)
            sys.stdout.flush()
        

    return x


class SparseCoding:
    def train(self, Xs, num_latent=2):
        """
        Xs: N appliance X T_time X M homes
        N does not include the aggregate
        num_latent: number of latent factors
        """
        self.N, self.T, self.M = Xs.shape
        self.num_latent = num_latent
        from sklearn.decomposition import NMF
        self.As = []
        self.Bs = []
        for i in range(self.N):
            self.model = NMF(n_components=self.num_latent, init='random', random_state=0)
            self.Bs.append(self.model.fit_transform(Xs[i]))
            self.As.append(self.model.components_)

    def disaggregate_home(self, X_test_agg):
        from scipy.optimize import nnls

        As_learnt = nnls(np.hstack(self.Bs), X_test_agg)[0]
        As_per_appliance = As_learnt.reshape(self.N, self.num_latent)
        out = []
        for appliance_num in range(self.N):
            out.append(self.Bs[appliance_num] @ As_per_appliance[appliance_num])
        return np.array(out)

    def disaggregate(self, X_test):
        """
        X_test: T_time X M homes containing aggregate data for M homes

        """
        self.pred = np.zeros((self.N, self.T, X_test.shape[1]))
        from scipy.optimize import nnls
        num_homes = X_test.shape[1]
        for home in range(num_homes):
            As_learnt = nnls(np.hstack(self.Bs), X_test[:, home])[0]
            As_per_appliance = As_learnt.reshape(self.N, self.num_latent)
            for appliance_num in range(self.N):
                self.pred[appliance_num, :, home] = self.Bs[appliance_num] @ As_per_appliance[appliance_num]
        return self.pred

    def disaggregate_discriminative(self, X_train, X_test, num_iter=2, alpha=1e-5):
        from scipy.optimize import nnls
        self.prediction_iterations = np.zeros((num_iter, self.N, self.T, X_test.shape[1]))

        self.A_star = np.vstack(self.As)
        self.B_tilde = np.hstack(self.Bs)

        for iter_number in range(num_iter):
            self.A_hat = nnls_autograd(self.B_tilde, X_train)
            self.B_tilde = self.B_tilde - alpha * ((X_train - self.B_tilde @ self.A_hat) @ self.A_hat.T - (
                    (X_train - self.B_tilde @ self.A_star) @ self.A_star.T))
            self.B_tilde[self.B_tilde < 0.] = 0.

            num_homes = X_test.shape[1]
            for home in range(num_homes):
                As_learnt = nnls(self.B_tilde, X_test[:, home])[0]
                self.As_per_appliance = As_learnt.reshape(self.N, self.num_latent)

                for appliance_num in range(self.N):
                    self.prediction_iterations[iter_number, appliance_num, :, home] = self.B_tilde[:,
                                                                                      appliance_num * self.num_latent:appliance_num * self.num_latent + self.num_latent] @ \
                                                                                      self.As_per_appliance[
                                                                                          appliance_num]
        return self.prediction_iterations