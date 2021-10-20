import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os


def read_res(filename):
    f1 = open(filename, 'r')
    fileline = f1.read()
    fileline = re.split(' |\n', fileline)
    return np.array([float(fileline[j]) for j in range(len(fileline)-1)])


def wrapper_function(directory, T, strName2):
    f_renyi = glob.glob(os.path.join(directory, strName2))
    nb_repeat = len(f_renyi)
    summation = np.zeros(T)

    for i in range(nb_repeat):
        file_read_llh = read_res(f_renyi[i])
        summation = summation + np.array(file_read_llh)

    summation = summation / nb_repeat

    return summation


T = 20
N = 10
J_t = 100
main_on = False #True  # set to False to switch to Exploration step described in Appendix D.3.2, set to True to swich to Exploration step used in [19]

if main_on:
    str_main = 'main'
else:
    str_main = 'appendices'

alpha_list = [0.5]
dim_latent_list = [16]#, 100]
cte_list = [0.]
eta_n_list = [0.3]
nb_samples_y_t_list = [100]#, 500, 1000]

plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams["legend.loc"] = 'lower right'

str_used = 'renyi'

for alpha in alpha_list:
    for dim_latent in dim_latent_list:
        for eta_n in eta_n_list:
            for cte in cte_list:
                for nb_samples_y_t in nb_samples_y_t_list:
                    h_t = np.power(J_t, -1/(4 + dim_latent))
                    str_params = '_M' + str(nb_samples_y_t) + 'N' + str(N) + 'T' + str(T) + 'ht' + str(h_t)

                    directory = './results/'+str_main+ '/dim' + str(dim_latent) + "/alpha" + str(alpha) + "/eta" + str(
                        eta_n) + 'kappa' + str(cte) + '/'

                    if main_on:
                        directory_md = './results/'+str_main+ '/dim' + str(dim_latent) + "/alpha" + str(alpha) + "/eta" + str(
                                eta_n) + 'kappa' + str(0.) + '/'
                        summation_MD_renyi_not_av = wrapper_function(directory_md, T * N, 'mixture_MD_'+ str_used +'_md_not_av' + str_params + '*.txt')

                    if not alpha == 1.:
                        summation_PD_renyi_not_av = wrapper_function(directory, T * N, 'mixture_PD_'+ str_used +'_not_av' + str_params + '*.txt')
                        summation_RD_renyi_not_av = wrapper_function(directory, T * N, 'mixture_RD_'+ str_used +'_md_not_av' + str_params + '*.txt')

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.set_title('Dimension ' + str(dim_latent) + r', $\alpha$ = ' + str(alpha) + r', $\eta_0$ = ' + str(eta_n) + ' M = ' + str(
                        nb_samples_y_t) )
                    ax.axhline(y=np.log(2), c='grey', label=r'$\log c$')
                    if not alpha == 1.:
                        ax.plot(summation_PD_renyi_not_av, label='PD')
                        ax.plot(summation_RD_renyi_not_av, label='RD')

                    if main_on:
                        ax.plot(summation_MD_renyi_not_av, label='EMD')

                    if dim_latent == 8:
                        ax.set_ylim(-7.5, 1.5)
                    if dim_latent == 16:
                        ax.set_ylim(-40., 1.5)
                    if dim_latent == 20:
                        ax.set_ylim(-6., 1.5)

                    ax.set_xlabel('Iterations')
                    ax.set_ylabel('Variational Renyi Bound')
                    ax.legend(loc='lower right')

                    fig.savefig('./results/'+ str_main +'ComparisonAlpha' +str(alpha) + 'Dim' + str(dim_latent) + str_used + 'M' + str(
                        nb_samples_y_t) + 'cte' + str(cte) + 'eta' + str(eta_n) + '.png', bbox_inches='tight')
                    plt.close()