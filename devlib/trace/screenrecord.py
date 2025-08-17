import os
import signal

from devlib.trace import TraceCollector
from devlib.host import kill_children

DEFAULT_MAX_DURATION = 180

class ScreenrecordCollector(TraceCollector):

    def __init__(self, target,
                 verbose=False,
                 size=None,
                 bit_rate=None,
                 time_limit_s=None,
                 debug=False,
                 rotate=False,
                 output_format=None,
    ):
        super(ScreenrecordCollector, self).__init__(target)

        self.verbose = verbose
        self.size = size
        self.bit_rate = bit_rate
        self.time_limit_s = time_limit_s
        self.debug = debug
        self.rotate = rotate
        self.output_format = output_format

        self.screenrecord = None

        args = []
        if self.verbose:
            args.append('--verbose')
        if self.size:
            args.append('--size {}'.format(self.size))
        if self.bit_rate:
            args.append('--bit-rate {}'.format(self.bit_rate))
        if self.time_limit_s and self.time_limit_s < DEFAULT_MAX_DURATION:
            args.append('--time-limit {}'.self.time_limit_s)
        if self.debug:
            args.append('--bugreport')
        if self.rotate:
            args.append('--rotate')
        if self.output_format:
            args.append('--output-format {}'.format(self.output_format))

        self._cmd = 'screenrecord {}'.format(' '.join(args))

    def reset(self):
        """

        """
        # Remove file(s)
        target.remove(self._remote_file)

    def start(self):
        """
        Start the video recording
        """
        # We'd need a set of e.g. numbered tmp files
        self._remote_file = self.target.path.join(
            self.target.working_directory,
            'screenrecord_tmp'
        )

        cmd = '{} {}'.format(self._cmd, self._remote_file)

        self._screenrecord = self.target.background(cmd)

    def stop(self):
        """
        Stop the video recording
        """
        if not self._screenrecord:
            raise RuntimeError('Logcat monitor not running, nothing to stop')

        kill_children(self._screenrecord.pid, signal.SIGINT)
        self._screenrecord.terminate()

    def get_trace(self, outfile):
        """
        Output collected video recording to designated file
        """
        # Files would be pulled and concat'd here
        self.target.pull(self._remote_file, outfile)
