import re
import os
import numpy as np


def extract_client_data(file_path):
    with open(file_path, 'r') as file:
        log_content = file.read()

    client_data_pattern = re.compile(r'========\sclient (\d+):(.+?)========', re.DOTALL)

    clients_data = {}

    for client_match in client_data_pattern.finditer(log_content):
        client_id = int(client_match.group(1))
        client_data_text = client_match.group(2)
        client_data = convert_text_to_dict(client_data_text)

        clients_data[client_id] = client_data

    return clients_data


def convert_text_to_dict(text):
    data_pattern = re.compile(r'(\w+):')
    data_dict = {}

    sections = data_pattern.split(text)
    sections = [section.strip() for section in sections if section.strip()]

    for i in range(0, len(sections), 2):
        key = sections[i]
        values_text = sections[i + 1]

        values = {}
        for line in values_text.strip().split('\n'):
            kv, okv = line.split(', ')
            k, v = kv.split(' = ')
            ov = okv.split(' = ')[1]
            values[k] = float(v)
            values[k + '_overhead'] = float(ov)

        data_dict[key] = values

    return data_dict


def main():
    # Replace 'log_file.txt' with the path to your log file
    client_data_lst = []
    for filename in os.listdir('./'):
        if not filename.startswith('clients'):
            continue
        client_data_lst += [extract_client_data(filename)]
    # print(list(client_data_lst[0][0]['TRAIN'].keys()))
    # print(list(client_data_lst[0][0]['TRAIN'].values()))
    # print(list(client_data_lst[0][0]['TRAIN'].items()))
    for client_data in client_data_lst:
        ap_data = np.array(list(client_data[0]['TRAIN'].values()) + list(client_data[0]['TEST'].values()))
        pp_data = np.zeros_like(ap_data)
        for cid in client_data:
            if cid == 0:
                continue
            pp_data += np.array(list(client_data[cid]['TRAIN'].values()) + list(client_data[cid]['TEST'].values()))
        pp_data /= len(client_data) - 1
        line = '\t'.join([str(o) for o in ap_data] + [str(o) for o in pp_data])
        print(line)

    # client_data = extract_client_data('clients_log[21 23][20-4-2023].log')
    # print(client_data)


if __name__ == '__main__':
    main()
