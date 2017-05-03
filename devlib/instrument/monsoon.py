import devlib
import json
import os
import os.path
import psutil
import time
import logging
from subprocess import Popen, STDOUT
from threading import Timer
import pandas as pd
import StringIO
from collections import namedtuple
from subprocess import Popen, PIPE, STDOUT
from devlib.instrument import Instrument, CONTINUOUS, MeasurementsCsv
from devlib.exception import HostError
from devlib.utils.misc import which

class MonsoonInstrument(Instrument):
    MONSOON_BIN = "./tools/scripts/monsoon.par"

    def log_error(self, errmsg, exception=0):
	print "ERROR: " + errmsg
	self.logger.error(errmsg)
	if exception == 1:
		raise(Exception(errmsg))

    def log_info(self, msg):
	print "INFO: " + msg
	self.logger.info(msg)

    def __init__(self, target, device_entry=None):
	super(MonsoonInstrument, self).__init__(target)

	if not os.path.exists(MonsoonInstrument.MONSOON_BIN):
		self.log_error("monsoon.par not found", exception=1)

	if device_entry != None and not os.path.exists(device_entry):
		self.log_error(str(device_entry) + " doesn't exist", exception=1)

	self.device_entry = device_entry
	self.process = None

	self.log_info('Found monsoon.par')

    def reset(self):
        self.log_info('Initiailzing MonsoonInstrument')
	# Waiting for monsoon to get ready
	self.run_monsoon(['--status'], timeout=3)
	self.log_info("MonsoonInstrument ready for action")

    def kill_monsoon(self, p):
	try:
		p.kill()
	except OSError:
		pass # ignore

    # Args should be a list of arguments ex, ['--status']
    # In !stream mode, we continously kill monsoon till it behaves (that is it doesn't block)
    # This is for the initial --status command and other commands where we're not streaming
    # but just running a simple monsoon command and returning.
    # For the streaming case (where we are confident monsoon is working after passing it --status),
    # we kill monsoon after timeout but this isn't an error, just termination of data collection.
    def run_monsoon(self, args=None, stream=0, async_mode=0, timeout=5, report=0):
	if report == 0 and args == None:
		raise(Exception("Can't run monsoon without args"))

	if report == 0:
		popen_args = [MonsoonInstrument.MONSOON_BIN]
		if self.device_entry:
			popen_args += ['--device', self.device_entry]
		popen_args += args

	# Give up after 20 trials for !stream case (every attempt is timeout seconds)
	n_trys_max = 20
	n_trys = n_trys_max

	while n_trys > 0:
		# When reporting, don't try to run monsoon
		if report == 0:
			msg = "Running monsoon command: " + str(popen_args)
			self.log_info(msg)
			p = Popen(popen_args, stdout=PIPE, stderr=PIPE)

		if async_mode == 1:
			self.process = p
			return "ASYNC"

		if report == 1:
			stream = 1
			p = self.process
			if p == None:
				self.log_error("Monsoon not running") 
			p.kill()
		else:
			# In sync mode (async_mode = 0), we kill monsoon after sometime
			t = Timer(timeout, self.kill_monsoon, [p])
			t.start()

		# Get return code for task after either it quit or it was killed
		waitret = p.wait()
		if waitret != 0:
			if report == 0:
				t.cancel()

			# If we received an error and we're !stream, do a retry
			if stream == 0:
				n_trys = n_trys - 1
				errmsg="timed out waiting or error for monsoon, try resetting it"
				self.log_error(errmsg)

			errmsg = p.stderr.read()
			stderrmsg = ""
			for l in errmsg.split(os.linesep):
				if l == "":
					continue
				stderrmsg += "STDERR: " + l + os.linesep

			# Always print error buffer
			self.log_error(stderrmsg)

			if stream == 1:
				# if stream, we don't consider -9 as an error, its just test completion timeout
				# if we do return for any other reason, raise an exception to catch it later
				if waitret != -9:
					self.log_error(stderrmsg, exception=1)
				self.log_info("MonsoonInstrument stream completed, returning")
				return p.stdout.read()

			if n_trys == 0:
				errmsg = "MonsoonInstrument maximum retries exceeded, please POWER CYCLE your monsoon."
				self.log_error(errmsg, exception=1)

			errmsg = ("Retrying attempt " + str(n_trys_max - n_trys)) + ", last output:\n" + p.stdout.read() + "\n"
			self.log_error(errmsg)
		else:
			if report == 0:
				t.cancel()
			self.log_info("MonsoonInstrument commands completed, returning")
			return p.stdout.read()
	self.log_error("monsoon unreachable code path", exception=1)

    def get_status(self):
        return self.run_monsoon(['--status'], timeout=3)

    def set_voltage(self, volt):
	if (volt > 5):
		self.log_error("Too high voltage requested: " + str(volt))
	return self.run_monsoon(['--voltage', str(volt)])

    def set_current(self, current):
	return self.run_monsoon(['--current', str(current)])

    def set_start_current(self, current):
	return self.run_monsoon(['--startcurrent', str(current)])

    # Returns pandas dataframe for monsoon output (time and current in mA)
    def get_samples_sync(self, duration):
	txt = self.run_monsoon(['--hz', '5', '--timestamp', '--samples', '2000000'], stream=1, timeout=duration)
	so = StringIO.StringIO(txt)
	return pd.read_csv(so, sep=' ', names=['time',	'current'])

    # Async stuff
    def start(self):
	self.run_monsoon(['--hz', '5', '--timestamp', '--samples', '2000000'], async_mode=1)

    def report(self):
	txt = self.run_monsoon(self, report=1)
	so = StringIO.StringIO(txt)
	return pd.read_csv(so, sep=' ', names=['time',	'current'])

