import os
import numpy as np
import logging
import shutil
from time import time, strftime, localtime
import torch


def read_all_align_dicts(data_path, kgs, threshold):

    def _read_align_dict(data_path):
        if not os.path.exists(data_path):
            return None
        src1_to_src2 = dict()
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                tokens = line.strip().split('\t')
                if len(tokens) == 2:
                    ent1, ent2 = tokens[0], tokens[1]
                    src1_to_src2[ent1] = {'ent': ent2, 'cos': 1.0}
                elif len(tokens) == 3:
                    ent1, ent2, cos = tokens[0], tokens[1], float(tokens[2])
                    if cos >= threshold:
                        src1_to_src2[ent1] = {'ent': ent2, 'cos': cos}
                else:
                    assert 0
        return src1_to_src2

    EA = {kg: dict() for kg in kgs}
    for kg1 in kgs:
        for kg2 in kgs:
            if kg1 == kg2:
                continue
            EA[kg1][kg2] = _read_align_dict(os.path.join(data_path, f'{kg1}_{kg2}.txt'))
    return EA


def mk_dirs_with_timestamp(prefix: str):
    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join(prefix, timestamp)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return output_dir


def get_logger(log_dir: str):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
