import json

"""

Some simple logging functionality, inspired by rllab's logginself.
Assumes that each diagnostic gets logged each iteration

Call logz.configure_output_dir() to start logging to a 
tab-separated-values file (some_folder_name/loself.txt)

To load the learning curves, you can do, for example

A = np.genfromtxt('/tmp/expt_1468984536/loself.txt', delimiter='\t', dtype=None, names=True)
A['EpRewMean']

"""

import os.path as osp
import time
import atexit
import os
import pickle
# import tensorflow as tf

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class G(object):
    def __init__(self):
        self.output_dir = None
        self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.path_model = None

    def configure_output_dir(self, d=None):
        """
        Set output directory to d, or to /tmp/somerandomnumber if d is None
        """
        self.output_dir = d or "/tmp/experiments/%i" % int(time.time())
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, "loself.txt"), 'w')
        atexit.register(self.output_file.close)
        print(colorize("Logging data to %s" % self.output_file.name, 'green', bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, \
                "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        assert key not in self.log_current_row, \
            "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val

    def save_params(self, params):
        with open(osp.join(self.output_dir, "params.json"), 'w') as out:
            out.write(json.dumps(params, separators=(',\n', '\t:\t'), sort_keys=True))

    # def pickle_tf_vars(self):
    #     """
    #     Saves tensorflow variables
    #     Requires them to be initialized first, also a default session must exist
    #     """
    #     _dict = {v.name: v.eval() for v in tf.global_variables()}
    #     with open(osp.join(self.output_dir, "vars.pkl"), 'wb') as f:
    #         pickle.dump(_dict, f)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        key_str = '%'+'%d' % max_key_len
        fmt = "| " + key_str + "s | %15s |"
        n_slashes = 22 + max_key_len
        # print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            if hasattr(val, "__float__"):
                val_str = "%8.3g" % val
            else:
                val_str = val
            # print(fmt % (key, val_str))
            vals.append(val)
        # print("-"*n_slashes)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers))
                self.output_file.write("\n")
            self.output_file.write("\t".join(map(str, vals)))
            self.output_file.write("\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False
