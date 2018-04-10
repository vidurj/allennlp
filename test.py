import atexit
import os
import signal
import subprocess
import time
import sys
import cmd
import sys


class Interpreter(cmd.Cmd):
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = '(teacher)'
    file = None

    def do_solve(self, arg):
        print('solving', arg)

    def do_learn(self, arg):
        print('learning', arg)

    def do_print_foo(self, _):
        print('foo')
        while sys.stdin.read() == '':
            time.sleep(0.1)

    def complete_solve(self, text, line, begidx, endidx):
        print('-' * 10)
        print('text:', text)
        print('line:', line)
        print('begidx:', begidx)
        print('endidx:', endidx)
        print('-' * 10)
        return []

Interpreter().cmdloop()
