import numpy as np
from EM import EM_Circle
from data_loader import preprocess_data, preprocess_dict, read_data
from process_result_utils import show_and_save_result, plot_and_save_likelihood, save_label
import pprint

preprocess_data()
vocab_name_list, vocab_count_list = preprocess_dict()
X, y = read_data()
# X is the data; y is the index

K_list = [10, 20, 30, 50]
results_list = []
label_list = []
log_likelihood_list = []

rng = np.random.default_rng(seed=0)
for K in K_list:
    pi, mu, gamma, L_list = EM_Circle(rng, X, K, gamma=None, eps=1e-5, maxiter=14)
    y = np.argmax(gamma, axis=1)
    label_list.append(y)
    results_list.append(np.argmax(mu, axis=1))
    log_likelihood_list.append(L_list)
save_label(label_list)
show_and_save_result(K_list, results_list, log_likelihood_list, vocab_name_list, vocab_count_list)
# plot_and_save_likelihood(K_list, log_likelihood_list)



