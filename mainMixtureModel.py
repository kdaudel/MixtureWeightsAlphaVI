import numpy as np
from scipy.stats import multivariate_normal
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
import PowerRenyiDescent as PRD
import EntropicMirrorDescent as EMD


# ### Model ### #

# Auxiliary functions
def _kernel_generate_1d(mean, sd, nb_samples):
    return np.random.normal(mean, sd, nb_samples)


def mixture_prob(y, means, sd, weights, nb_peaks, dim):
    '''
    Compute the pdf of the mixture model evaluated at y
    '''
    pdf_y = 0

    for i in range(nb_peaks):
        pdf_y += weights[i] * multivariate_normal.pdf(y, mean=means[i], cov=sd * np.identity(dim))

    return np.array(pdf_y)

class MVN:
    def __init__(self, target_nb_peaks, target_means, target_weights, target_sd, Z, D):
        '''
        Initialise a Mixture Model
        :param target_nb_peaks: number of modes (int)
        :param target_means: array of means (array)
        :param target_weights: mixture weights (array)
        :param target_sd: standard deviation (float > 0)
        :param Z: normalising constant (float > 0)
        :param D: dimension (int)
        '''
        self.target_nb_peaks = target_nb_peaks
        self.target_means = target_means
        self.target_weights = target_weights
        self.target_sd = target_sd
        self.Z = Z
        self.D = D

        if self.D == 1:
            self._generate_from_multivariate = _kernel_generate_1d
        else:
            self._generate_from_multivariate = self._kernel_generate_nd

    def lnprob(self, theta):
        '''
        Compute the log prob of the mixture model evaluated in theta
        '''
        prob = self.Z * mixture_prob(theta, self.target_means, self.target_sd, self.target_weights,
                                     self.target_nb_peaks, self.D)
        return np.log(prob)

    def _kernel_generate_nd(self, mean, sd, nb_samples):
        return np.random.multivariate_normal(mean=mean, cov=sd * np.identity(self.D), size=nb_samples)

    def sample_from_true_posterior(self, nb_samples_y):
        '''
        Sample according to the mixture model
        '''
        repartition = np.random.multinomial(nb_samples_y, self.target_weights)

        samples = []
        for i in range(self.target_nb_peaks):
            nb = repartition[i]
            u = self._generate_from_multivariate(self.target_means[i], self.target_sd, nb)
            samples.extend(u)
        return np.array(samples)


# ### Main functions ### #

def save_file(j, directory, strname, arraytosave):
    '''
    Save array under the filename directory + strname + str(j) + '.txt'
    '''
    filename = directory + strname + str(j) + '.txt'
    np.savetxt(filename, arraytosave)


def main_function(j):

    # Initialise (theta_1, ..., theta_J_t) according to a normal distribution
    if dim_latent == 1:
        thetas_init = np.random.normal(q0_mean, q0_sd, J_t)
    else:
        thetas_init = np.random.multivariate_normal(q0_mean, q0_sd * np.identity(dim_latent), J_t)

    if not alpha == 1.:
        try:
            # Power Descent
            powerDescent = PRD.PowerRenyiDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N, J_t, h_t,
                                                  nb_samples_y_t, eta_n, False, False, main_on, cte)
            thetas_final, weights_final, renyi_bound_lst = powerDescent._full_algorithm()

            save_file(j, directory, 'mixture_PD_renyi_not_av'+ str_params, renyi_bound_lst)

        except:
            pass

        try:
            # Renyi Descent
            mirrorDescent = PRD.PowerRenyiDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N, J_t, h_t,
                                                  nb_samples_y_t, eta_n, True, False, main_on, cte)
            thetas_final_md, weights_final_md, renyi_bound_lst_md = mirrorDescent._full_algorithm()

            save_file(j, directory, 'mixture_RD_renyi_md_not_av' + str_params, renyi_bound_lst_md)

        except:
            pass

    if main_on:
        try:
            # Entropic Mirror Descent applied to mu mapsto Psi(mu) (reference algorithm)
            mirrorDescent = EMD.EntropicMirrorDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N, J_t, h_t,
                                                  nb_samples_y_t, eta_n, False)
            thetas_final_md, weights_final_md, renyi_bound_lst_md = mirrorDescent._full_algorithm()

            save_file(j, directory, 'mixture_MD_renyi_md_not_av'+ str_params, renyi_bound_lst_md)

        except:
            pass

    return 0

# ### Parameters ### #
T = 20
N = 10
J_t = 100

alpha_list = [0.5]
dim_latent_list = [16]#, 100]
cte_list = [0.]
eta_n_list = [0.3]
nb_samples_y_t_list = [100]#, 500, 1000]
main_on = False # set to False to switch to Exploration step described in Appendix D.3.2, set to True to swich to Exploration step used in [19]
if main_on:
    str_main = 'main'
else:
    str_main = 'appendices'

nb_cores_used = 1  # set > 1 to parallelise the different runs of the algorithm
nb_repeat_exp = 5

for alpha in alpha_list:
    for nb_samples_y_t in nb_samples_y_t_list:
        for dim_latent in dim_latent_list:
            for eta_n in eta_n_list:
                for cte in cte_list:

                    h_t = np.power(J_t, -1/(4 + dim_latent))
                    str_params = '_M' + str(nb_samples_y_t) + 'N' + str(N) + 'T' + str(T) + 'ht' + str(h_t)

                    q0_sd = 5.
                    q0_mean = np.array([0.] * dim_latent)

                    directory = './results/'+ str_main +'/dim' + str(dim_latent) + "/alpha" + str(alpha) + "/eta" + str(
                        eta_n) + 'kappa' + str(cte) + '/'
                    Path(directory).mkdir(parents=True, exist_ok=True)

                    # ### Define the targeted density and initialise model ### #
                    target_means = [[2.] * dim_latent, [-2.] * dim_latent]
                    target_nb_peaks = len(target_means)
                    target_sd = 1.
                    target_weights = [1 / target_nb_peaks] * target_nb_peaks
                    c = 2.

                    model = MVN(target_nb_peaks, target_means, target_weights, target_sd, c, dim_latent)

                    # ### Launch the experiments ### #
                    i_list = range(nb_repeat_exp)
                    
                    #Parallel(nb_cores_used)(delayed(main_function)(i) for i in i_list)
                    for i in i_list:
                        main_function(i)
