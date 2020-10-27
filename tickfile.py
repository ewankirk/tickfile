'''
A set of classes to compress tick files
'''

import struct
import math
import os
import io
import collections
import time
#import pstats
# import cProfile
from typing import NamedTuple, IO, Text, Tuple, ByteString

_BLOCK_SIZE = 4096
_NO_BYTE_FLAG = 0
_ONE_BYTE_FLAG = 1
_TWO_BYTE_FLAG = 2
_FOUR_BYTE_FLAG = 3

TEST_DATA = 'es01c.small'
COMPRESSED_DATA = 'es01c.small.z'


#each header
#    magic: 8 double  or maybe the first date 8 bytes
#    tick_res: 8 byte float
#    dimension: 2 bytes unsigned short (dimensions in the file
#    last_index: 2 byte unsigned short (max blocsize = 0xFFFF)
#    time_res: 4 bytes unsigned long (how many milliseconds 10e-3 of time
#    resolution)
#
#    this is a total of 24 bytes
class BlockHeader(NamedTuple):
    magic: float
    tick_res: float
    dimension: int
    last_index: int
    time_res: int

_COMPRESSION_SCHEMA_ONE = -1.0


class Point(NamedTuple):
    date: float
    values: Tuple[float]


class CompressException(Exception):
    ''' A compression exception '''
    pass

class BaseDataFile(object):
    ''' The abstract base class for compressed files '''

    def get_points(self, start=0) -> Point:
        ''' Get points from the file starting at date '''
        raise NotImplementedError()

    def insert_points(self, at_date, points) -> None:
        ''' Insert the points into the file at date '''
        raise NotImplementedError()

    def delete_points(self, at_date, count):
        ''' Delete the points at date '''
        raise NotImplementedError()

    def append_points(self, points):
        ''' Append the points to the file '''
        raise NotImplementedError()

class CompressedTickFile(BaseDataFile):
    ''' A compressed file which handles multidimensional tick points '''

    _double_packer = struct.Struct('<d')
    _byte_packer = struct.Struct('<B')
    _short_packer = struct.Struct('<H')
    _long_packer = struct.Struct('<L')
    _header_packer = struct.Struct('<ddHHL')

    def __init__(self, filename: str, dimension: int = 1) -> None:
        
        self._fd: IO = io.open(filename, 'rb')
        self._reader = io.BufferedReader(self._fd)
        buf: bytes = self._reader.read(_BLOCK_SIZE)
        self._compressed: int = self._is_block_compressed(buf)
        if self._compressed:
            # read in the nan and the next short which is the dimension.  We
            # need to know this before we read in everything else
            self._dimension = struct.unpack_from('<H', buf, 16)[0]
            header = self._get_header(buf)
            self._time_res = header.time_res
            self._tick_res = header.tick_res
        else:
            self._dimension = dimension
            self._time_res = None
            self._tick_res = None

        self._double_dimension_format = str(self._dimension) + 'd'
        self._byte_dimension_format = '<' + str(self._dimension) + 'b'
        self._short_dimension_format = '<' + str(self._dimension) + 'h'
        self._long_dimension_format = '<' + str(self._dimension) + 'l'
        self._header_length = 32 + 8 * self._dimension
        self._position_cache: dict = {}
        self._reader.seek(0)
        # Precreate the struct writers to save time
        self._double_dim_packer = struct.Struct(self._double_dimension_format)
        self._byte_dim_packer = struct.Struct(self._byte_dimension_format)
        self._short_dim_packer = struct.Struct(self._short_dimension_format)
        self._long_dim_packer = struct.Struct(self._long_dimension_format)
        
    def __str__(self) -> str:
        return ("CompressedRTSTreamFile: "
                "compressed = {} time_res = {} tick_res {} "
                "dimension {}").format(self.compressed, self._time_res,
                                       self._tick_res, self._dimension)

    def close(self) -> None:
        ''' Close this file '''
        self._reader.close()
        self._fd.close()

    @property
    def compressed(self) -> int:
        ''' Is this file compressed? '''
        return self._compressed

    @property
    def tick_resolution(self) -> float:
        ''' What is the tick resolution '''
        return self._tick_res

    @property
    def time_resolution(self) -> int:
        ''' What is the time resolution '''
        return self._time_res

    def _compress(self, point: Point, prev: Point, time_res: int, tick_res: float) -> ByteString:
        ''' Compress a point relative to the previous point
            This function calculates the delta of pt to prev using the supplied
            date resolution (in milliseconds) and tick_resolution (a double).
            Returns a bytearray containing the compressed bytestream
        '''
        delta_t: int = int((point.date - prev.date) * 1000.0 / time_res)
        delta_v: Tuple[int] = tuple([int((x - y)/tick_res) for x, y in zip(point.values, prev.values)])

        if delta_t == 0:
            date_bit_size = 1
        else:
            date_bit_size = int(math.log(delta_t, 2)) + 1
        # Special case single dimensional data because it is so prevalent
        if self._dimension == 1:
            if delta_v == (0,):
                value_bit_size = 1
            else:
                value_bit_size = int(math.log(abs(delta_v[0]), 2)) + 1
        else:
            # This is multidimensional so harder.
            # This piece of code is a bit unclear but we add 127 to the tuple so that
            # we are guaranteed to be at least one byte for each multidimensional thingy
            # we are not going to mess about with nibbles and shit like that.
            multi = max([abs(x) for x in delta_v + (127,)])
            value_bit_size = int(math.log(abs(multi), 2)) + 1
        # We preallocate the byte array.  Could use extend but why bother?
        buf = bytearray(5 + 4 * self._dimension)
        first_byte = 0
        # date is easier since it is always positive
        if date_bit_size <= 8:
            # We could do much much better here for very high frequency data because we could
            # compress everything < 32ms into a nibble...
            date_bytes = _ONE_BYTE_FLAG
            self._byte_packer.pack_into(buf, 1, delta_t)
            i = 2
        elif date_bit_size <= 16:
            date_bytes = _TWO_BYTE_FLAG
            self._short_packer.pack_into(buf, 1, delta_t)
            i = 3
        else:
            date_bytes = _FOUR_BYTE_FLAG
            self._long_packer.pack_into(buf, 1, delta_t)
            i = 5
        # value is a bit harder because it can be negative
        # If the tick change is less than 4 ticks in any direction we can fit it into the first byte
        if value_bit_size < 4:
            # This can only be single dimensional data
            value_bytes = _NO_BYTE_FLAG
            first_byte = delta_v[0] + 7
        elif value_bit_size < 8:
            value_bytes = _ONE_BYTE_FLAG
            self._byte_dim_packer.pack_into(buf, i, *delta_v)
            i += self._dimension
        elif value_bit_size < 16:
            value_bytes = _TWO_BYTE_FLAG
            self._short_dim_packer.pack_into(buf, i, *delta_v)
            i += self._dimension * 2
        else:
            value_bytes = _FOUR_BYTE_FLAG
            self._long_dim_packer.pack_into(buf, i, *delta_v)
            i += self._dimension * 4
        # Now encode up the first byte
        # The first byte in teh frame is important
        # The first two bits are the number of bytes in the date
        # the next two bits are the number of bytes in the tick
        # the last four bits are the tick movement +4 if and only if we have
        # smaller than 4 tick movement.
        self._byte_packer.pack_into(buf, 0, first_byte | (date_bytes << 6) | (value_bytes << 4))
        return buf[0:i]

    def _get_block(self, block_num):
        ''' read the _BLOCK_SIZE block from the file into a buffer and return it
        '''
        self._reader.seek(block_num * _BLOCK_SIZE)
        buf = self._reader.read(_BLOCK_SIZE)
        return buf

    def _get_header(self, buf):
        ''' read the header out of a compressed file
        '''
        return BlockHeader(*self._header_packer.unpack_from(buf, 0))

    def _get_date_pos(self, block):
        ''' get the date at the beginning of this block

            The first 32 bytes have enough info for us to work out what's happening.
            This critically depends on the format of the header which can't really
            change now
            There is a dict which caches the date at the beginning of the point
            This saves considerable IO on files
        '''
        if block in self._position_cache:
            return self._position_cache[block]
        self._reader.seek(block * _BLOCK_SIZE)
        buf = self._reader.read(32)
        if self._is_block_compressed(buf):
            date = self._double_packer.unpack_from(buf, 24)[0] # double at byte 24 is the date
        else:
            date = self._double_packer.unpack_from(buf, 0)[0] # first double is the date
        self._position_cache[block] = date
        return date

    def _is_block_compressed(self, buf):
        '''Check if this block is compressed

        '''
        return math.isnan(*self._double_packer.unpack_from(buf, 0))

    def _compressed_point_generator(self, buf):
        ''' Generate points from a compressed buffer

            The generator starts at the header_length in the buffer (which
            should be bytes or bytearray and then steps through uncompressing
            the data as deltas to the previous point.  The first point
            is the key that starts this buffer

            Remember that every block can potentially have a different
            tick and date resolution
        '''
        # The first point is yielded immedidately because we know it
        head = self._get_header(buf)
        point = Point(self._double_packer.unpack_from(buf, 24)[0],
                      self._double_dim_packer.unpack_from(buf, 32))
        i = self._header_length
        while i < head.last_index:
            yield point
            # i points to the current point in the buffer so unpack this one
            byte1 = self._byte_packer.unpack_from(buf, i)[0]
            value_bytes = (byte1 >> 4) & 0x03
            date_bytes = (byte1 >> 6) & 0x03
            value = byte1 & 0x0F
            i += 1
            # if there really are data bytes then unpack them
            if date_bytes == _ONE_BYTE_FLAG:
                date = self._byte_packer.unpack_from(buf, i)[0]
                i += 1
            elif date_bytes == _TWO_BYTE_FLAG:
                date = self._short_packer.unpack_from(buf, i)[0]
                i += 2
            elif date_bytes == _FOUR_BYTE_FLAG:
                date = self._long_packer.unpack_from(buf, i)[0]
                i += 4
            else:
                raise CompressException("bad date format in uncompress: ", date_bytes)
            # second is the value
            if value_bytes == _NO_BYTE_FLAG:
                values = (value - 7,)
            elif value_bytes == _ONE_BYTE_FLAG:
                values = self._byte_dim_packer.unpack_from(buf, i)
                i += self._dimension
            elif value_bytes == _TWO_BYTE_FLAG:
                values = self._short_dim_packer.unpack_from(buf, i)
                i += self._dimension * 2
            elif value_bytes == _FOUR_BYTE_FLAG:
                values = self._long_dim_packer.unpack_from(buf, i)
                i += self._dimension * 4
            else:
                raise CompressException("bad values format in uncompress: ", value_bytes)
            # now turn back into proper units
            values = tuple([x * head.tick_res for x in values])
            date = date * float(head.time_res) / 1000.0
            point = Point(date + point.date,
                          tuple([x + y for x, y in zip(values, point.values)]))

    def _uncompressed_point_generator(self, buf):
        ''' generates points from an uncompressed buffer starting at 0

            create unpack objects
        '''
        length = len(buf)
        index = 0
        while index < length:
            try:
                point = Point(self._double_packer.unpack_from(buf, index)[0],
                                self._double_dim_packer.unpack_from(buf, index + 8))
                index += (self._dimension + 1) * 8
            except struct.error as error:
                print(error)
            yield point
   
    def _get_points_from_block(self, buf):
        ''' Get points from a block

            Deals with compressed or uncompressed blocks
        '''
        if self._is_block_compressed(buf):
            for point in self._compressed_point_generator(buf):
                yield point
        else:
            # rewind the buffer
            for point in self._uncompressed_point_generator(buf):
                yield point

    def _file_binary_search(self, find, low, high):
        ''' bisection search a file for the block which contains the date

        '''
        if high < low:
            return low # this is the one
        mid = (high - low) // 2
        date = self._get_date_pos(mid)
        if date > find:
            return self._file_binary_search(find, low, mid - 1)
        elif date < find:
            return self._file_binary_search(find, mid + 1, high)
        else:
            return mid



    def get_points(self, start=0):
        ''' get points starting at the optional start

            do a bisection search of the file to find the block
            that the seek date is in then we start reading points
            from this block until the date is bigger than the seek date
            and then start yielding the points. This implementation
            just passes in a double.  Dates would be better but
            leave for real implementation
        '''
        self._reader.seek(0, io.SEEK_END)
        max_block = self._reader.tell() // _BLOCK_SIZE
        pos = self._file_binary_search(start, 0, max_block)
        block = pos
        while True:
            buf = self._get_block(block)
            block += 1
            if len(buf) == 0:
                break
            points = []
            for point in self._get_points_from_block(buf):
                points.append(point)
            for point in points:
                if point.date > start:
                    yield point


    # This function compresses the current file to a new named file
    def compress_file(self, filename, time_res=None, tick_res=None):
        ''' compresses the current data file into a new file given by filename

        '''
        if tick_res is None:
            tick_res = self._tick_res
        if time_res is None:
            time_res = self._time_res

        writer = io.BufferedWriter(io.open(filename, "wb"))
        outbuf = bytearray(_BLOCK_SIZE)
        index = 0
        first_packer = struct.Struct('<d' + str(self._dimension) + 'd')
        max_size_compressed = 4 * self._dimension + 5

        # seek to the beginning of ourself
        self._reader.seek(0)
        block = 0

        while True:
            buf = self._get_block(block)
            if len(buf) == 0:
                # we are done
                break
            points = self._get_points_from_block(buf)
            #now loop and compress
            for point in points:
                if index == 0: # this is the first point so just take it
                    first_packer.pack_into(outbuf, 24, point.date, *point.values)
                    index = 32 + self._dimension * 8
                    prev = point
                    continue
                comp = self._compress(point, prev, time_res, tick_res)
                prev = point
                outbuf[index:index + len(comp)] = comp
                index = index + len(comp)
                # now check for end of block condition
                if index + max_size_compressed < _BLOCK_SIZE:
                    continue
                # if we get here then there isn't enough space for
                # another point so write this one and reset the buffer
                self._header_packer.pack_into(outbuf, 0, float('nan'),
                                              float(tick_res), int(self._dimension),
                                              int(index), int(time_res))
                writer.write(outbuf)
                index = 0
            # now there are no points left so get the next block
            block += 1
        # and we're done getting points so flush the buffer
        self._header_packer.pack_into(outbuf, 0, float('nan'),
                                      float(tick_res), int(self._dimension),
                                      int(index), int(time_res))
        # note that this sends the full _BLOCK_SIZE to the file
        # this is to ensure that the file is block aligned after compression
        # so simple append works
        writer.write(outbuf)
        writer.close()

# Some test stuff
class Timer():
    ''' A utility class to time the files '''

    def __init__(self, prompt=None):
        self.prompt = prompt
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        formatted_time = '{:8.5f}'.format(time.time() - self.start)
        if self.prompt is None:
            print('{}'.format(formatted_time))
        else:
            print('{}: {}'.format(self.prompt, formatted_time))

def _test():
    filename = TEST_DATA
    compressed_name = COMPRESSED_DATA
    print(compressed_name)

    with Timer('Just read data in raw'):
        file_ = open(filename, "rb")
        for _ in file_.read(1024):
            pass

    csr = CompressedTickFile(filename)
    with Timer('Compress:'):
        csr.compress_file(compressed_name, 1, 0.125)

    csr = CompressedTickFile(filename)
    amount = 100000
    count = 0
    uncomp = []
    avg = 0
    with Timer('Read from uncompressed file'):
        for point in csr.get_points():
            uncomp.append(point)
            avg = avg + point.values[0]
            count = count + 1
            if count > amount:
                break
        print(avg/count)

    csr2 = CompressedTickFile(compressed_name)
    count = 0
    avg = 0
    comp = []
    with Timer('Read from compressed file'):
        for point in csr2.get_points():
            if count % 10000 == 0:
                print(point)
            comp.append(point)
            avg = avg + point.values[0]
            count = count + 1
            if count > amount:
                break
        print(avg / count)
    avg = 0
    with Timer('Loop through points'):
        for point in comp:
            avg = avg + point.values[0]
    print(avg / amount)

if __name__ == '__main__':
    # os.chdir('/Users/ewan/src/ccptickfiles')
    print('foo')
    _test()
    
