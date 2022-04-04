def get_word_result(results_list, vocab_name_list, vocab_count_list):
    name_result = []
    count_result = []
    for result in results_list:
        word_name = []
        word_count = []
        for index in result:
            word_name.append(vocab_name_list[index])
            word_count.append(vocab_count_list[index])
        name_result.append(word_name)
        count_result.append(word_count)
    return name_result, count_result

def show_and_save_result(K_list, results_list, log_likelihood_list, vocab_name_list, vocab_count_list):
    name_result, count_result = get_word_result(results_list, vocab_name_list, vocab_count_list)

    word_results_dict = {}
    for i, K in enumerate(K_list):
        word_result = {}
        word_result['name'] = name_result[i]
        word_result['count'] = count_result[i]
        word_results_dict[K] = word_result

    import json
    jsonstr = json.dumps(word_results_dict)
    filename = open('./word_results_dict.json','w')#dict转josn
    filename.write(jsonstr)

def plot_and_save_likelihood(K_list, log_likelihood_list):
    import matplotlib.pyplot as plt
    legend_list = ['K='+f'{K}' for K in K_list]
    fig, ax = plt.subplots()
    for log_likelihood in log_likelihood_list:
        ax.plot(log_likelihood)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-likelihood')
    plt.legend(labels=legend_list,loc='best')
    fig.savefig('Log-likelihood.png')