import sys, re, os
import subprocess
import multiprocessing

MKDSSP = './mkdssp'

def get_ext(fname):
    return str.split(fname, os.extsep)[-1]


def get_name(fname):
    return str.split(fname, os.extsep)[0]


def multicall(calls: list):
    """
    Spawn multiple concurrent subprocesses to run through all calls more rapidly. T
his will run no
    more processes than there are threads on the system.
    :param calls: Array of call arrays, see subprocess.run()
    :return: None
    """
    processes = []
    init_count = len(calls)
    target_processes = int(multiprocessing.cpu_count() * 1.5)
    while len(calls) > 0 or len(processes) > 0:
        while len(processes) < target_processes and len(calls) > 0:
            call = calls.pop()
            print("Calling {:06} of {:06}: {}".format(init_count - len(calls), init_count, call))
            processes.append(subprocess.Popen(call))
        tmp = []
        for process in processes:
            code = process.poll()
            if code is None:
                tmp.append(process)
        if len(tmp) != 0 and len(tmp) == len(processes):
            os.waitpid(-1, 0)
        processes = tmp
    print("All processes finished")

def run_dssp(input, output=None):
    args = "{} -i {}".format(MKDSSP, input)
    dssp = os.popen(args, 'r')
    ss = []
    for line in dssp:
        match = re.match(r'^ +\d+ +\d+ +([A-Z]) {2}([A-Z]?)', line)
        if match is not None:
            ss.append(match.group(2))
    dssp.close()
    if output is not None:
        output = open(output,"w")
        output.write('> ' + get_name(os.path.basename(input)) + '\n')
        for i in ss:
            if i is '':
                output.write('C')
            else:
                output.write(i)
        output.write('\n')
    else:
        print('> ' + get_name(os.path.basename(input)))
        for i in ss:
            if i is '':
                print('C', end='')
            else:
                print(i, end='')
        print()



if __name__ == '__main__':
    if len(sys.argv) == 3:
        MKDSSP = os.path.abspath(sys.argv[1])
        run_dssp(sys.argv[2])
    elif len(sys.argv) == 4:
        MKDSSP = os.path.abspath(sys.argv[1])
        run_dssp(sys.argv[2], sys.argv[3])
    else:
        print("Usage: ./run_dssp.py <Path of dssp> <input> <optional output>\n"
              " <Path of dssp> mkdssp file to run dssp.\n"
              " <input>  PDB to assess.\n"
              " <output> File to write of secondary structures formatted for UniCon3D.\n"
              "          If unspecified, it will write to stdout.")
