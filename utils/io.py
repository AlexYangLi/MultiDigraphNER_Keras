# -*- coding: utf-8 -*-

import os
import json
import pickle


def pickle_load(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        print('Logging Info - Loaded:', filename)
    except EOFError:
        print('Logging Error - Cannot load:', filename)
        obj = None

    return obj


def pickle_dump(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

    print('Logging Info - Saved:', filename)


def write_log(filename, log, mode='w'):
    with open(filename, mode) as writer:
        writer.write('\n')
        json.dump(log, writer, indent=4, ensure_ascii=False)


def format_filename(_dir, filename_template, **kwargs):
    """Obtain the filename of data base on the provided template and parameters"""
    filename = os.path.join(_dir, filename_template.format(**kwargs))
    return filename
