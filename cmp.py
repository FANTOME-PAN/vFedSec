import numpy as np
import random
from phe import paillier
import time


class Timer:
    mark: float
    cnt: int = 0
    total: float = 0.

    def tic(self):
        self.mark = time.time()

    def toc(self):
        dt = time.time() - self.mark
        self.total += dt
        self.cnt += 1
        return dt

    def avg(self):
        return self.total / self.cnt


timer = Timer()


def paillier_encrypt_matrix(matrix, public_key):
    encrypted_matrix = []
    encrypt_timer = Timer()
    for row in matrix:
        encrypted_row = []
        for x in row:

            encrypt_timer.tic()
            res = public_key.encrypt(int(x))
            encrypt_timer.toc()
            encrypted_row += [res]
        encrypted_matrix.append(encrypted_row)
    print(f"Avg encrypt time = {encrypt_timer.avg()} (x {encrypt_timer.cnt} = {encrypt_timer.total})")
    return encrypted_matrix


def paillier_decrypt_matrix(encrypted_matrix, secret_key):
    decrypted_matrix = []
    for row in encrypted_matrix:
        decrypted_row = [secret_key.decrypt(x) for x in row]
        decrypted_matrix.append(decrypted_row)
    return decrypted_matrix


def paillier_matrix_multiply(matrix_a, matrix_b, public_key, secret_key):
    tic = time.time()
    encrypted_matrix_a = paillier_encrypt_matrix(matrix_a, public_key)
    print(f"encrypt time = {time.time() - tic}")

    matrix_b_T = list(zip(*matrix_b))  # Transpose matrix B
    # encrypted_matrix_b_T = paillier_encrypt_matrix(matrix_b_T, public_key)
    encrypted_matrix_b_T = matrix_b_T

    result = []
    prod_timer = Timer()
    add_timer = Timer()
    decrypt_timer = Timer()

    for i in range(len(matrix_a)):
        result_row = []
        for j in range(len(matrix_b[0])):
            prod_timer.tic()
            res = encrypted_matrix_a[i][0] * encrypted_matrix_b_T[j][0]
            prod_timer.toc()
            encrypted_dot_product = res
            for k in range(1, len(matrix_a[0])):
                prod_timer.tic()
                res = encrypted_matrix_a[i][k] * encrypted_matrix_b_T[j][k]
                prod_timer.toc()
                add_timer.tic()
                encrypted_dot_product += res
                add_timer.toc()
            decrypt_timer.tic()
            res = secret_key.decrypt(encrypted_dot_product)
            decrypt_timer.toc()
            result_row.append(res)
        result.append(result_row)
    print(f"Avg product time = {prod_timer.avg()} (x {prod_timer.cnt} = {prod_timer.total})\n"
          f"Avg addition time = {add_timer.avg()} (x {add_timer.cnt} = {add_timer.total})\n"
          f"Avg decryption time = {decrypt_timer.avg()} (x {decrypt_timer.cnt} = {decrypt_timer.total})")
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

    public_key, secret_key = paillier.generate_paillier_keypair()
    print(f"public key n: {public_key.n}")

    for batch_size in [256]:
        print(f'batch size {batch_size}:')
        matrix_a = np.random.randint(low=1, high=1 << 27, size=(256, 2))
        matrix_b = np.random.randint(low=1, high=1 << 27, size=(2, 2))
        timer.tic()
        result = paillier_matrix_multiply(matrix_a.tolist(), matrix_b.tolist(), public_key, secret_key)
        tr = timer.toc()
        print(f'\tHE cpu_time = {tr}')
        total_time = 0
        for _ in range(100):
            timer.tic()
            simulate_secure_aggregation(matrix_a, matrix_b)
            total_time += timer.toc()
        print(f'\tSA cpu_time = {total_time / 100}\n')
