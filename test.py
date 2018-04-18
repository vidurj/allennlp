import atexit
import os
import signal
import subprocess
import time
import sys
import cmd
import sys
import torch

# class Interpreter(cmd.Cmd):
#     intro = 'Welcome. Type help or ? to list commands.\n'
#     prompt = '(teacher)'
#     file = None
#
#     def do_solve(self, arg):
#         print('solving', arg)
#
#     def do_learn(self, arg):
#         print('learning', arg)
#
#     def do_print_foo(self, _):
#         print('foo')
#         while sys.stdin.read() == '':
#             time.sleep(0.1)
#
#     def complete_solve(self, text, line, begidx, endidx):
#         print('-' * 10)
#         print('text:', text)
#         print('line:', line)
#         print('begidx:', begidx)
#         print('endidx:', endidx)
#         print('-' * 10)
#         return []
#
# Interpreter().cmdloop()
#
# print(os.environ)
# print('agent_name' in os.environ)
# print('AGENT_NAME' in os.environ)



A = torch.FloatTensor([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5]]])
ind = torch.LongTensor([0, 1, 0])


# def batched_index_select(t, dim, inds):
#     dummy = inds.unsqueeze(1).unsqueeze(2).expand(inds.size(0), 1, t.size(2))
#     print(dummy.size())
#     out = t.gather(dim, dummy)  # b x e x f
#     return out.squeeze(1)
#
# dim = 1
# print(batched_index_select(A, dim, ind))

tensor_1 = torch.FloatTensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
tensor_2 = torch.FloatTensor([])
print((tensor_1 * tensor_2).sum(dim=-1))
