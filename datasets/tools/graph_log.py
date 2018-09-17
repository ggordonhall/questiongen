import sys
import matplotlib.pyplot as plt


from typing import List, Tuple, Union

"""Functions to graph perplexity and accuracy of
training schedule with Matplotlib.pyplot.

Accepts type of graph to generate from standard input.
Write graph to file.
"""


def read_log_ppl(filename: str) -> Tuple[List[int],
                                         List[int], List[int], List[int]]:
    """Read the log file and extract perplexity

        Args:
            filename {str}: the name of the logfile

        Returns:
            train_steps {List[int]}: number of trainsteps
            train_ppl {List[int]}: train perplexity
            valid_steps {List[int]}: number of validation steps
            valid_ppl {List[int]}: validation perplexity
    """
    train_ppl, valid_ppl = [], []
    train_steps, valid_steps = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        logs = f.readlines()
        for idx, log in enumerate(logs[1:]):
            # logs are printed twice
            if log == logs[idx]:
                continue
            # split each record into tokenss
            log = [l for l in log.split(' ') if l != '']
            if len(log) < 5:
                continue

            if log[3] == 'Step':
                step, ppl = log[4].split('/')[0], log[8][:-1]
                train_steps.append(int(step))
                try:
                    train_ppl.append(float(ppl))
                except ValueError:
                    print(log)

            elif log[4] == 'perplexity:':
                step, ppl = train_steps[-1], log[5][:-1]
                valid_steps.append(int(step))
                try:
                    valid_ppl.append(float(ppl))
                except ValueError:
                    print(log)

    return (train_steps, train_ppl, valid_steps, valid_ppl)


def read_log_acc(filename: str) -> Tuple[List[int],
                                         List[int], List[int], List[int]]:
    """Read the log file and extract accuracy

        Args:
            filename {str}: the name of the logfile

        Returns:
            train_steps {List[int]}: number of trainsteps
            train_ppl {List[int]}: train accuracy
            valid_steps {List[int]}: number of validation steps
            valid_ppl {List[int]}: validation accuracy
    """
    train_acc, valid_acc = [], []
    train_steps, valid_steps = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        logs = f.readlines()
        for idx, log in enumerate(logs[1:]):
            # logs are printed twice
            if log == logs[idx]:
                continue
            # split each record into tokens
            log = [l for l in log.split(' ') if l != '']
            if len(log) < 5:
                continue

            if log[3] == 'Step':
                step, acc = log[4].split('/')[0], log[6][:-1]
                train_steps.append(int(step))
                try:
                    train_acc.append(float(acc))
                except ValueError:
                    print(acc)

            elif log[4] == 'accuracy:':
                step, acc = train_steps[-1], log[5][:-1]
                valid_steps.append(int(step))
                try:
                    valid_acc.append(float(acc))
                except ValueError:
                    print(acc)

    return (train_steps, train_acc, valid_steps, valid_acc)


def graph_log(data: Tuple[List[int], List[int], List[int], List[int]],
              type: Union['Accuracy', 'Perplexity']):
    """Plot the data"""
    train_steps, train_data, valid_steps, valid_data = data
    plt.plot(train_steps, train_data, 'r', valid_steps, valid_data, 'b')
    plt.ylabel(type)
    plt.xlabel('Steps')
    plt.grid(True)
    plt.legend(('Train Set', 'Validation Set'))
    plt.savefig('{}.png'.format(type), dpi=300)


if __name__ == '__main__':
    filename = 'log_file.log'
    if sys.argv[1] == 'acc':
        data = read_log_acc(filename)
        graph_log(data, 'Accuracy')
    else:
        data = read_log_ppl(filename)
        graph_log(data, 'Perplexity')
