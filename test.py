import atexit
import os
import signal
import subprocess
import time

print(subprocess.check_call(['mkdir', '-p', 'test']))