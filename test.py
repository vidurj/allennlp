import atexit
import os
import signal
import subprocess
import time


class Task:
    """
    Runs turbo evaluations & cleanup thereof.
    """

    def __init__(self, cwd_copy_name, log_file_name):
        self.process = None
        self.p = None
        self.cwd_copy_name = cwd_copy_name
        self.log_file_name = log_file_name

    def kill_process(self):
        """
        Kill process if parent exits.
        NOTE: this will not work if the parent exits ungracefully
        https://stackoverflow.com/a/14128476
        process.kill() is insufficient to processes spawned by the child process.
        See
        https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
        """
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except OSError:
            # Process is already dead
            pass

    def start(self):
        """
        Runs turbo evaluations
        """
        self.process = subprocess.Popen(['echo', '"foo\nbar \n bax"'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        preexec_fn=os.setsid,
                                        cwd=self.cwd_copy_name)
        # This ensures that the process is killed if the main process fails.
        atexit.register(self.kill_process)

    def wait(self):
        """
        Waits for the process to complete, while streaming its logs. When the process finishes,
        it deletes the copy of the cwd the process was run in.
        """
        log_lines = []

        while True:
            line = self.process.stdout.readline()
            if line != '':
                print(line)
                log_lines.append(line)
            elif self.process.poll() is not None:
                break
            else:
                time.sleep(1)

        line = self.process.stdout.readline()
        while line != '':
            print(line)
            log_lines.append(line)
            line = self.process.stdout.readline()

        with open(self.log_file_name, 'w') as f:
            f.write('\n'.join(log_lines))

        subprocess.check_call(['rm', '-rf', self.cwd_copy_name])
        return self.process.poll()



t = Task('data', 'logs.txt')
t.start()
time.sleep(1)
t.wait()