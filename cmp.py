import numpy as np
import random
from phe import paillier
import time
import pickle


class Timer:
    def tic(self):
        self.mark = time.time()

    def toc(self):
        return time.time() - self.mark


timer = Timer()


def paillier_encrypt_matrix(matrix, public_key):
    encrypted_matrix = []
    for row in matrix:
        encrypted_row = [public_key.encrypt(int(x)) for x in row]
        encrypted_matrix.append(encrypted_row)
    return encrypted_matrix


def paillier_decrypt_matrix(encrypted_matrix, secret_key):
    decrypted_matrix = []
    for row in encrypted_matrix:
        decrypted_row = [secret_key.decrypt(x) for x in row]
        decrypted_matrix.append(decrypted_row)
    return decrypted_matrix


def paillier_matrix_multiply(matrix_a, matrix_b, public_key, secret_key):
    encrypted_matrix_a = paillier_encrypt_matrix(matrix_a, public_key)
    matrix_b_T = list(zip(*matrix_b))  # Transpose matrix B
    # encrypted_matrix_b_T = paillier_encrypt_matrix(matrix_b_T, public_key)
    encrypted_matrix_b_T = matrix_b_T

    result = []

    for i in range(len(matrix_a)):
        result_row = []
        for j in range(len(matrix_b[0])):
            encrypted_dot_product = encrypted_matrix_a[i][0] * encrypted_matrix_b_T[j][0]
            for k in range(1, len(matrix_a[0])):
                encrypted_dot_product += encrypted_matrix_a[i][k] * encrypted_matrix_b_T[j][k]
            result_row.append(secret_key.decrypt(encrypted_dot_product))
        result.append(result_row)

    return result


def simulate_secure_aggregation(mat_a, mat_b, num_parties=5):
    # ans = mat_a.dot(mat_b)
    # for s in range(num_parties):
    #     np.random.seed(s)
    #     ans += np.random.randint(low=0, high=1024, size=ans.shape)
    # return ans
    mat_a = mat_a.tolist()
    mat_b = mat_b.T.tolist()
    result = []

    for i in range(len(mat_a)):
        result_row = []
        for j in range(len(mat_b[0])):
            dot_prod = mat_a[i][0] * mat_b[j][0]
            for k in range(1, len(mat_a[0])):
                dot_prod += mat_a[i][k] * mat_b[j][k]
            # add mask
            dot_prod += sum([random.randint(0, 1024) for _ in range(num_parties)])
            result_row.append(dot_prod)
        result.append(result_row)

    return result


if __name__ == "__main__":
    results_list = []
    for idx in range(10):
        res = {}
        results_list += [res]
        print(f'Experiment {idx + 1}:')
        public_key, secret_key = paillier.generate_paillier_keypair()
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            print(f'Running HE with batch size {batch_size}...')
            matrix_a = np.random.randint(low=1, high=9, size=(batch_size, 8))
            matrix_b = np.random.randint(low=1, high=9, size=(8, 8))
            timer.tic()
            result = paillier_matrix_multiply(matrix_a.tolist(), matrix_b.tolist(), public_key, secret_key)
            he_time = timer.toc()
            # print(f'\tHE cpu_time = {he_time}')
            total_time = 0
            print(f'Running SA with batch size {batch_size}...')
            for _ in range(1000):
                timer.tic()
                simulate_secure_aggregation(matrix_a, matrix_b)
                total_time += timer.toc()
            sa_time = total_time / 1000
            # print(f'\tSA cpu_time = {sa_time}\n')
            res[batch_size] = (he_time, sa_time)
        print(f'')
    pth = 'temp_exp_results.pkl'
    print(f'Writing results to {pth}')
    with open(pth, 'wb') as f:
        pickle.dump(results_list, f)
