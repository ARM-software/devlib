#    Copyright 2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import zlib
from io import FileIO, UnsupportedOperation, TextIOBase
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO


class BufferedZippedTextReader(TextIOBase):

    def __init__(self, file_path, window_size=16384, wbits=47):
        '''
        parameters:
            :file_path: path to the archive.
            :window_size: size of buffer holding compressed data (bytes).
            :wbits: configuration parameter for zlib decompressor.
        '''
        self.file_path = file_path
        self.in_stream = FileIO(self.file_path, 'r')
        self.window_size = window_size
        self.wbits = wbits
        self._reset()
        self.mode = 'r'

    def tell(self):
        '''
        Returns the current posititon of cursor in the extracted file. 
        Returns -1 when EOF has been reached.
        '''
        return self.cur_pos
   
    def seek(self, offset, whence=0):
        '''
        Moves cursor to specified offset in extracted file.

        parameters:
            :offset: the absolute or relative offset to which the file cursor
                     must be moved.
            :whence: default value is 0 meaning absolute file positionning. 
                     1 means seek() is relative to current position. Other 
                     strategies are not yet implemented.
        '''
        if whence == 0:
            target_pos = offset
        elif whence == 1:
            target_pos = self.tell() + offset
        else:
            raise ValueError('Invalid whence: {}'.format(whence))
        if target_pos < self.tell():
            self._reset()
        while self.tell() != target_pos and self.tell() != -1:
            self.char_iterator.next()
        return self.tell()

    def readline(self, size=-1):
        line_writer = StringIO()
        char_count = 0
        while char_count < size or size < 0:
            c = self.char_iterator.next()
            if c == '\n' or not c:
                break
            line_writer.write(c)
            char_count += 1
        return line_writer.getvalue()

    def readlines(self, size=-1):
        return self.read(size).split('\n')

    def readall(self):
        res = StringIO()
        for c in self.char_iterator:
            res.write(c)
        return res.getvalue()

    def read(self, size=-1):
        if size == -1 or size == None:
            return self.readall()
        i = 0
        res = StringIO()
        while self.tell() != -1 and i < size:
            c = self.char_iterator.next()
            res.write(c)
            i += 1
        return res.getvalue()
    
    def close(self):
        super(TextIOBase, self).close()
        self.in_stream.close()

    def __enter__(self):
        self._reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self

    def next(self):
        return self.readline()

    def fileno(self):
        return self.in_stream.fileno()

    def isatty(self):
        return self.in_stream.isatty()

    def readable(self):
        return True

    def seekable(self):
        return True

    def detach(self, *args, **kwargs):
        raise UnsupportedOperation('Cannot detach underlying buffer.')

    # Stub write-related methods of TextIOBase
    def writable(self):
        return False

    def write(self, *args, **kwargs):
        raise IOError('Cannot write in read-only stream')
    
    def writelines(self, *args, **kwargs):
        raise IOError('Cannot write in read-only stream')
    
    def truncate(self, *args, **kwargs):
        raise IOError('Cannot truncate read-only stream')

    def flush(self):
        pass

    def _iter_buff(self):
        while True:
            to_unzip = self.in_stream.read(self.window_size)
            buff_writer = StringIO()
            while to_unzip:
                buff_writer.write(self.decompressor.decompress(to_unzip))
                to_unzip = self.decompressor.unconsumed_tail
            buff = buff_writer.getvalue()
            if buff:
                yield buff
            else:
                return

    def _iter_char(self):
        for buff in self.buff_iterator:
            for c in buff:
                self.cur_pos += 1
                yield c
        self.cur_pos = -1
    
    def _reset(self):
        self.cur_pos = 0
        self.in_stream.seek(0) 
        self.decompressor = zlib.decompressobj(self.wbits)
        self.buff_iterator = self._iter_buff()
        self.char_iterator = self._iter_char()

