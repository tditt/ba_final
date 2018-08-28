import numpy as np
import time


# Implementation of k-Vector search, as described in the paper by Daniele Mortari and Beny Neta
# Some snippets of this code are taken from https://github.com/elsuizo/K-vector (GNU General Public License)

def construct_k_vector(data):
    print('constructing k-vector, this might take some time!')

    start = time.time()
    n = data.shape[0]
    k = np.zeros(n)
    k[0] = 0
    k[-1] = n

    # get sorting vector (ascending mode, in paper: I)
    sort = np.argsort(data)

    # get smallest and biggest elements
    y_min = data[sort[0]]
    y_max = data[sort[-1]]

    # use floating point machine precision to construct "big epsilon" from the paper
    float_precision = np.finfo(np.float64).eps
    epsilon = float_precision * (max(abs(y_min), abs(y_max)))

    # calculate line parameters m and q
    m = (y_max - y_min + 2 * epsilon) / (n - 1)
    q = y_min - m - epsilon

    # generate k-vector
    j = 0
    for i in range(2, n - 1):
        z = m * i + q
        while j < n - 1 and not (data[sort[j]] <= z < data[sort[j + 1]]):
            j += 1
        k[i - 1] = j

    print('k-vector constructed!')
    print('elapsed time: ', str(time.time() - start))
    return k, sort, q, m


def k_vector_search(indices_mode, y_a, y_b, data, k, sort, q, m):
    if y_a >= y_b: print('error: cancelling search... y_a must be smaller than y_b!')
    y_a_min = data[sort[0]]
    y_b_max = data[sort[-1]]
    if y_a < y_a_min: y_a = y_a_min
    if y_b > y_b_max: y_b = y_b_max
    # print('fetching...')
    # start = time.time()
    max = k.shape[0] - 1
    j_b = int(np.floor((y_a - q) / m))
    j_t = int(np.ceil((y_b - q) / m))
    # if j_b > max:
    #     print('exceeding possible range for j_b, selecting maximum')
    #     print('corresponding value for y_a: ', str(y_a), ' and min and max values for data : ', str(data[sort][0]),
    #           str(data[sort[-1]]))
    #     j_b = max
    # if j_t > max:
    #     print('exceeding possible range for j_t, selecting maximum')
    #     print('corresponding value for y_b: ', str(y_b), ' and min and max values for data : ', str(data[sort][0]),
    #           str(data[sort[-1]]))
    #     j_t = max

    if j_b <= 0:
        print('j_b is 0 or smaller, cancelling...')
        return None
    if j_t <= 0:
        print('j_t is 0 or smaller, cancelling...')
        return None

    # get start and end index
    k_start = int(k[j_b - 1] + 1)
    k_end = int(k[j_t - 1])
    if k_start > max or k_end <= 0:
        print('k_start or k_end out of range, cancelling...')
        print('k_start ', str(k_start), 'k_end', str(k_end))
        return None

    # print(str(k_start))
    # print(str(j_b))
    # print(str(k_end))
    # print(str(j_t))
    # data at k_start and k_end has certain chance of not belonging to range, perform corrections:
    while k_start < 0 or data[sort[k_start]] < y_a:
        k_start += 1
    while k_end > max or data[sort[k_end]] > y_b:
        k_end -= 1

    # get search result
    if indices_mode:
        # output vector with indices
        result = sort[np.arange(k_start, k_end + 1)]
    else:
        # output data directly
        result = data[sort[np.arange(k_start, k_end + 1)]]

    if result.shape[0] == 0:
        print('found nothing...')
        return result

    # print('...successful! found ' + str(result.shape[0]) + ' items!')
    # print('elapsed time: ', str(time.time() - start))
    return result


def verify_k_vector(data, k, sort, q, m, y_a, y_b):
    search = k_vector_search(False, y_a, y_b, data, k, sort, q, m)
    bf_search = []
    for d in data:
        if d < y_a or d > y_b: continue
        bf_search.append(d)
    result = np.array(bf_search)
    sort = np.argsort(result)
    sorted_arr = result[sort]
    if search is None or len(search) == 0:
        print('k-vector search empty')
        if len(bf_search) != 0:
            print('brute-force search contains ', result.shape[0], ' items')
        else:
            print('brute-force search also empty! k-vector functionality verified!')
        return
    if search.shape[0] != sorted_arr.shape[0]:
        print('fail: number of found items in k-vector search is ', search.shape[0], ' while brute-force search gives',
              sorted_arr.shape[0], ' items!')
        if search.shape[0] >= 1:
            print('k-vector search first and last item: ', str(search[0]), str(search[-1]))
        else:
            print('k-vector search returned empty result')
        if sorted_arr.shape[0] >= 1:
            print('brute-force search first and last item: ', str(sorted_arr[0]), str(sorted_arr[-1]))
        else:
            print('brute-force search returned empty result')
        return
    for i, v in enumerate(sorted_arr):
        if search[i] != v:
            print('fail: k-vector search item ' + str(search[i]), ' doesnt match ' + str(v))
            return

    print('k-vector functionality verified!')


def run_test():
    y = np.array(np.random.random(100000), dtype=np.float16)
    k, sort, q, m = construct_k_vector(y)
    test = np.random.randint(0, 100000, 1000)
    for number in test:
        print(str(k[number]))
    verify_k_vector(y, k, sort, q, m, 0.8, 0.9)
    verify_k_vector(y, k, sort, q, m, 0.2, 0.4)
    verify_k_vector(y, k, sort, q, m, 0.123, 0.125)
    verify_k_vector(y, k, sort, q, m, 0.126, 0.128)
    verify_k_vector(y, k, sort, q, m, 0.023, 0.125)
    verify_k_vector(y, k, sort, q, m, 0.25664, 0.9999)
    verify_k_vector(y, k, sort, q, m, 0.383, 0.555)
    verify_k_vector(y, k, sort, q, m, 0.222, 0.6666)
    # y2 = np.random.randint(5000, size=5000000)
    # y1 = np.arange(10000000)
    # np.random.shuffle(y1)
    # k, sort, q, m = construct_k_vector(y1)
    # y3 = np.arange(10000000)
    # np.random.shuffle(y3)
    # k, sort, q, m = construct_k_vector(y3)
    # y4 = np.arange(10000000)
    # np.random.shuffle(y4)
    # y2 = np.concatenate((y1, y3, y4))
    # k, sort, q, m = construct_k_vector(y2)
    # verify_k_vector(False, y2, k, sort, q, m, 1, 5000)
    # verify_k_vector(False, y2, k, sort, q, m, 5687, 8888)
    # verify_k_vector(False, y2, k, sort, q, m, 1256, 1999)
    # verify_k_vector(False, y2, k, sort, q, m, 2122, 7541)
    # verify_k_vector(False, y2, k, sort, q, m, 5522, 5566)
    # verify_k_vector(False, y2, k, sort, q, m, 3358, 7894)
    # verify_k_vector(False, y2, k, sort, q, m, 5, 120)
    # verify_k_vector(False, y2, k, sort, q, m, 1000, 1005)
