import torch
import torch.nn.functional as F

class CKA(object):
    '''
    Calculates the linear centered kernel alignment between two sets of representations in pytorch.
    Main function runner is linear_CKA.
    >>> cka = CKA()
    >>> x, y = torch.randn(200, 768), torch.randn(200, 768)
    >>> cka.linear_cka(x, y)
    '''
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        I = torch.eye(n, device=self.device)
        H = I - torch.ones([n, n], device=self.device) / n
        return H @ K @ H 

    def linear_HSIC(self, X, Y):
        #Calculate Gram matrix.
        L_X = X @ X.T
        L_Y = Y @ Y.T
        #Center the two Gram matrices and calculate the HSIC between them i.e. the trace of their product.
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        #Numerator HSIC
        hsic = self.linear_HSIC(X, Y)
        #Denominator HSICs
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)
    
class Procrustes(object):
    def __init__(self, device):
        self.device = device

    def orthogonal_procrustes_distance(self, X, Y, normalize = True):
        if normalize:
            X = X - X.mean()
            X = X / torch.norm(X, p = 'fro')

            Y = Y - Y.mean()
            Y = Y / torch.norm(Y, p = 'fro')
        XY = X.t() @ Y
        XY_norm = torch.linalg.norm(XY, ord = 'nuc')
        d = 2 - 2 * XY_norm
        return d
    
    def orthogonal_procrustes_similarity(self, X, Y, normalize = False):
        d = self.orthogonal_procrustes_distance(X, Y, normalize)
        sim = 1.0 - d if normalize else 2.0 - d
        return sim

def random_projection(activations, target_dim):
    input_dim = activations.size(1)
    projection_matrix = torch.randn(input_dim, target_dim, device=activations.device) / (input_dim ** 0.5)
    return torch.matmul(activations, projection_matrix)

class Ridge(object):
    '''
    Calculates similarity between two sets of representations using fits from a ridge regression in pytorch.
    The code takes the representations, finds the best fit on half the samples, and then tests on the other half.
    Score is calculated using cosine similarity. 
    >>> ridge = Ridge(device = torch.device('cpu'))
    >>> X, y = torch.randn(256, 768), torch.randn(256, 768)
    >>> ridge.linear_regression_similarity(X, y)
    '''
    def __init__(self, device, target_dim = 1024, epsilon = 1e-5, alpha = 10.0):
        self.device = device
        self.target_dim = target_dim
        self.epsilon = epsilon
        self.alpha = alpha

    def split_tensors(self, X, Y):
        split_index = X.size(0) // 2
        X_train, X_test = X[:split_index], X[split_index:]
        Y_train, Y_test = Y[:split_index], Y[split_index:]
        return X_train, X_test, Y_train, Y_test
    
    def standardize(self, X_train, X_test, Y_train, Y_test):
        X_train_mean = X_train.mean(dim = 0, keepdim = True)
        X_train_std = X_train.std(dim = 0, keepdim = True) + self.epsilon
        Y_train_mean = Y_train.mean(dim = 0, keepdim = True)
        Y_train_std = Y_train.std(dim = 0, keepdim = True) + self.epsilon

        X_train_centered = (X_train - X_train_mean) / X_train_std
        Y_train_centered = (Y_train - Y_train_mean) / Y_train_std
        X_test_centered = (X_test - X_train_mean) / X_train_std
        Y_test_centered = (Y_test - Y_train_mean) / Y_train_std
        return X_train_centered, X_test_centered, Y_train_centered, Y_test_centered

    def linear_regression_similarity(self, X, Y):
        if X.shape[1] > self.target_dim:
            X_projected = random_projection(X, self.target_dim)
        else:
            X_projected = X
            
        if Y.shape[1] > self.target_dim:
            with torch.no_grad():
                Y_projected = random_projection(Y, self.target_dim)
        else:
            Y_projected = Y

        X_train, X_test, Y_train, Y_test = self.split_tensors(X_projected, Y_projected)
        X_train_centered, X_test_centered, Y_train_centered, Y_test_centered = self.standardize(X_train, X_test, Y_train, Y_test)
        cov_matrix = torch.matmul(X_train_centered.T, Y_train_centered)
        cov_X = torch.matmul(X_train_centered.T, X_train_centered) + self.alpha * torch.eye(X_train_centered.size(1), device = self.device)

        weights = torch.linalg.solve(cov_X, cov_matrix)
        Y_pred = torch.matmul(X_test_centered, weights)
        similarity = F.cosine_similarity(Y_pred, Y_test_centered, dim = 1).mean()
        return similarity

class CCA(object):
    def __init__(self, device):
        self.device = device

    def zero_mean(self, tensor, dim):
        return tensor - tensor.mean(dim = dim, keepdims = True)

    def svd(self, x, y):
        u_1, s_1, v_1 = torch.linalg.svd(x, full_matrices = False) 
        v_1 = v_1.t()
        u_2, s_2, v_2 = torch.linalg.svd(y, full_matrices = False) 
        v_2 = v_2.t()
        uu = u_1.t() @ u_2

        try:
            u, diag, v = (uu).svd()
        except RuntimeError as e:
            raise e
        
        a = v_1 @ s_1.reciprocal().diag() @ u
        b = v_2 @ s_2.reciprocal().diag() @ u
        return a, b, diag
    
    def cca(self, x, y):
        assert x.size(0) == y.size(0)

        x = self.zero_mean(x, 0)
        y = self.zero_mean(y, 0)
        return self.svd(x, y)

    def svd_reduction(self, tensor, accept_rate = 0.99):
        left, diag, right = torch.linalg.svd(tensor, full_matrices = False)
        full = diag.abs().sum()
        ratio = diag.abs.cumsum(dim = 0) / full
        num = torch.where(ratio < accept_rate, torch.ones(1).to(self.device), torch.zeros(1).to(self.device)).sum()
        return tensor @ right[:, :int(num)]

    def svcca_distance(self, x, y):
        x = self.svd_reduction(x)
        y = self.svd_reduction(y)
        div = min(x.size(1), y.size(1))
        a, b, diag = self.cca(x, y)
        return 1 - diag.sum() / div

    def pwcca_distance(self, x, y):
        a, b, diag = self.cca(x, y)
        alpha = (x @ a).abs().sum(dim = 0)
        alpha = alpha / alpha.sum()
        return 1 - alpha @ diag

class DifferentiableRSA(object):
    '''
    Performs representational similarity analysis between two sets of representations in pytorch.
    Main function runner is rsa.
    >>> rsa = DifferentiableRSA()
    >>> x, y = torch.randn(200, 768), torch.randn(200, 768)
    >>> rsa.rsa(x, y)
    '''
    def __init__(self, device):
       self.device = device

    def compute_similarity_matrix(self, representations):
        #Normalize the representations and compute the distance between every pair of examples
        representations_norm = representations / torch.norm(representations, dim=1, keepdim=True)
        similarity_matrix = torch.mm(representations_norm, representations_norm.T)
        return similarity_matrix

    def pearson_correlation(self, x, y):
        #Flatten and compute pearson correlation
        x_flat = x.flatten()
        y_flat = y.flatten()
        x_centered = x_flat - x_flat.mean()
        y_centered = y_flat - y_flat.mean()

        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
        return numerator / denominator
    
    def rsa(self, representations_1, representations_2):
        similarity_matrix_1 = self.compute_similarity_matrix(representations_1)
        similarity_matrix_2 = self.compute_similarity_matrix(representations_2)

        rsa_value = self.pearson_correlation(similarity_matrix_1, similarity_matrix_2)
        return rsa_value