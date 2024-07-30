import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from dataclasses import dataclass
from app_global import BLANK
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TextIO
import struct
import re
import io

from app_global import NO_DATA


class FileFormat(Enum):
    ASCII = auto()
    BINARY = auto()
    GEOTIFF = auto()


@dataclass
class TSLAMMFile:
    file_name: str
    converted_file_name: str = ""
    file_format: FileFormat = FileFormat.ASCII
    file_no_data: int = -9999  # Default no-data value, update as needed
    chunk_size = 65536  # Define the size of each chunk for GeoTiff
    text_file: TextIO = None
    row: int = 0
    col: int = 0
    xll_corner: float = 0.0
    yll_corner: float = 0.0
    cell_size: float = 1.0


@dataclass
class TSLAMMInputFile(TSLAMMFile):
    read_version: float = 1.0
    is_repeat: bool = False
    repeat_num: float = 1.0
    iter_to_read: int = 0
    data = None  # Current GeoTiff chunk data
    current_index = 0  # Current index in flat data array for GeoTiff
    dataset = None  # rasterio dataset for GeoTiff
    current_window = None  # Current data window for GeoTiff

    def __init__(self, filen: str):
        self.file_name = filen
        self.read_stream: Optional[io.BufferedReader] = None
        if filen.lower().endswith("asc") or filen.lower().endswith("txt"):
            self.file_format = FileFormat.ASCII
        elif filen.lower().endswith("slb"):
            self.file_format = FileFormat.BINARY
        else:
            self.file_format = FileFormat.GEOTIFF

    def load_next_chunk(self):
        """Load the next chunk of data from the GeoTIFF."""
        if self.current_window is None:
            row_offset = 0
        else:
            row_offset = self.current_window.row_off + self.chunk_size

        if row_offset >= self.dataset.height:
            self.data = None  # No more data to read
            return False

        # Calculate the number of rows to read without exceeding the dataset
        rows = min(self.chunk_size, self.dataset.height - row_offset)
        self.current_window = Window(0, row_offset, self.dataset.width, rows)
        self.data = self.dataset.read(window=self.current_window, masked=True)
        self.current_index = 0  # Reset index for new chunk
        return True

    def prepare_file_for_reading(self):
        try:
            if self.file_format == FileFormat.GEOTIFF:
                self.dataset = rasterio.open(self.file_name)
                self.col, self.row = self.dataset.width, self.dataset.height
                transform = self.dataset.transform
                self.xll_corner, self.yll_corner = transform.c, transform.f
                self.cell_size = self.dataset.res[0]  # Assuming square pixels
                self.file_no_data = self.dataset.nodatavals[0] if self.dataset.nodatavals else self.file_no_data
                self.load_next_chunk()  # Load the first chunk

            elif self.file_format == FileFormat.ASCII:
                self.text_file = open(self.file_name, 'r')
                headers = {}
                for _ in range(6):  # Assume six header lines
                    line = self.text_file.readline().strip()
                    key, value = re.split(r'\s+', line)
                    headers[key.lower()] = value

                self.col = int(headers.get('ncols', 0))
                self.row = int(headers.get('nrows', 0))
                self.xll_corner = float(headers.get('xllcorner', 0.0))
                self.yll_corner = float(headers.get('yllcorner', 0.0))
                self.cell_size = float(headers.get('cellsize', 1.0))
                self.file_no_data = int(headers.get('nodata_value', self.file_no_data))

            elif self.file_format == FileFormat.BINARY:
                self.read_stream = open(self.file_name, 'rb')
                read_version = struct.unpack('d', self.read_stream.read(8))[0]
                if not (0.999 < read_version < 9.999):
                    raise ValueError(f"Error Reading Version Number for data File '{self.file_name}'")

                self.col, = struct.unpack('i', self.read_stream.read(4))
                self.row, = struct.unpack('i', self.read_stream.read(4))
                self.xll_corner, = struct.unpack('d', self.read_stream.read(8))
                self.yll_corner, = struct.unpack('d', self.read_stream.read(8))
                self.cell_size, = struct.unpack('d', self.read_stream.read(8))

            return True
        except Exception as e:
            print(f"Error preparing file '{self.file_name}': {str(e)}")
            return False

    def get_next_number(self):
        try:
            if self.file_format == FileFormat.ASCII:
                while True:
                    s: str = ''
                    while True:
                        char = self.text_file.read(1)
                        if not char:  # End of file check
                            return None  # Indicate end of file
                        if char.isspace() and s:
                            break
                        elif char and not char.isspace():
                            s += char
                    num = float(s)
                    if num == self.file_no_data:
                        num = NO_DATA
                    return num

            elif self.file_format == FileFormat.BINARY:
                if self.iter_to_read == 0:
                    self.is_repeat = struct.unpack('?', self.read_stream.read(1))[0]
                    self.iter_to_read = struct.unpack('H', self.read_stream.read(2))[0]  # Read as unsigned short
                    num = struct.unpack('f', self.read_stream.read(4))[0]
                    if self.is_repeat:
                        self.repeat_num = num
                    self.iter_to_read -= 1
                else:
                    if self.is_repeat:
                        num = self.repeat_num
                    else:
                        num = struct.unpack('f', self.read_stream.read(4))[0]
                    self.iter_to_read -= 1
                return num

            elif self.file_format == FileFormat.GEOTIFF:
                if self.data is None or self.current_index >= self.data.size:
                    if not self.load_next_chunk():
                        return None  # No more data to load
                # Retrieve the next number from the flat array of the current chunk
                num = self.data.flat[self.current_index]
                self.current_index += 1

                if np.ma.is_masked(num):
                    return BLANK  # Return default value for no-data cells
                return num
            else:
                return None

        except (EOFError, struct.error, IndexError):
            return None

    def close_file(self):
        if self.file_format == 'text':
            self.text_file.close()
        elif self.file_format == 'binary':
            self.read_stream.close()
        elif self.file_format == 'geotiff':
            self.dataset.close()


class TSLAMMOutputFile(TSLAMMFile):
    def __init__(self, nc, nr, xll, yll, sz, fn, fformat):
        self.col = nc
        self.row = nr
        self.xll_corner = xll
        self.yll_corner = yll
        self.cell_size = sz
        self.file_name = fn
        self.write_stream = None
        self.num_written = 0
        self.num_list = []
        self.buffer_size = 65535
        self.file_format = fformat
        if self.file_format == FileFormat.GEOTIFF:
            # Initialize the data array and setup for writing
            self.data = np.empty((nr, nc), dtype=np.float32)  # fixme assuming float32 for simplicity
            self.current_index = 0

    def write_header(self, prompt=False):

        while True:
            if prompt and Path(self.file_name).exists():
                response = input(f'Overwrite {self.file_name}? (y/n): ')
                if response.lower() != 'y':
                    new_fn = input('Enter new file name or press enter to cancel:')
                    if new_fn:
                        self.file_name = new_fn
                    else:
                        err_msg = 'No alternative filename chosen'
                        return False, err_msg

            try:

                if self.file_format == FileFormat.GEOTIFF:
                    # Define the data type and the no data value
                    data_type = 'float32'    # fixme modify for elevations (float) vs. categories (Int)
                    file_no_data = NO_DATA

                    # Set up the affine transformation for the raster
                    transform = from_origin(self.xll_corner, self.yll_corner + self.row * self.cell_size, self.cell_size,
                                            self.cell_size)

                    # Set up the metadata dictionary
                    meta = {
                        'driver': 'GTiff',
                        'dtype': data_type,
                        'nodata': file_no_data,
                        'width': self.col,
                        'height': self.row,
                        'count': 1,  # Number of bands; change if multi-band
                        'compress': 'deflate',  # Using DEFLATE compression
                        'crs': None,  # CRS is set to None, which means unspecified  fixme
                        'transform': transform
                    }
                    self.write_stream = rasterio.open(self.file_name, 'w', **meta)
                elif self.file_format == FileFormat.ASCII:
                    self.text_file = open(self.file_name, 'w')
                    self.text_file.write(f'ncols {self.col}\n')
                    self.text_file.write(f'nrows {self.row}\n')
                    self.text_file.write(f'xllcorner {self.xll_corner}\n')
                    self.text_file.write(f'yllcorner {self.yll_corner}\n')
                    self.text_file.write(f'cellsize {self.cell_size}\n')
                    self.text_file.write(f'nodata_value {NO_DATA}\n')
                elif self.file_format == FileFormat.BINARY:
                    self.write_stream = open(self.file_name, 'wb')
                    slb_version = 1.0
                    self.write_stream.write(struct.pack('d', slb_version))  # d is 8 bytes
                    self.write_stream.write(struct.pack('i', self.col))   # i is 4 bytes
                    self.write_stream.write(struct.pack('i', self.row))
                    self.write_stream.write(struct.pack('d', self.xll_corner))
                    self.write_stream.write(struct.pack('d', self.yll_corner))
                    self.write_stream.write(struct.pack('d', self.cell_size))

                return True, ''

            except Exception as e:

                print(f'Error writing header for {self.file_name}: {str(e)}')
                retry_response = input('Would you like to retry? (y/n): ')
                if retry_response.lower() != 'y':
                    return False, f'Error writing header for {self.file_name}: {str(e)}'

    def cr(self):
        if self.file_format == FileFormat.ASCII:
            self.text_file.write('\n')

    def write_next_number(self, num, last_number):
        if self.file_format == FileFormat.GEOTIFF:
            row, col = divmod(self.current_index, self.col)
            # Write directly to file in chunks
            if row % self.chunk_size == 0 and col == 0 and self.current_index != 0:
                # Flush current chunk
                start_row = row - self.chunk_size
                window = Window(0, start_row, self.col, self.chunk_size)
                self.write_stream.write(self.data[start_row:start_row + self.chunk_size, :], 1, window=window)
                self.data[start_row:start_row + self.chunk_size, :] = 0  # Clear memory if needed

            # Store the number in the local array
            self.data[row, col] = num
            self.current_index += 1

            if last_number:
                start_row = row - (row % self.chunk_size)
                num_rows_in_last_chunk = (row % self.chunk_size) + 1
                window = Window(0, start_row, self.col, num_rows_in_last_chunk)
                self.write_stream.write(self.data[start_row:start_row + num_rows_in_last_chunk, :], 1, window=window)
                # dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
                # dst.update_tags(ns='rio_overview', resampling='nearest')
                self.write_stream.close()  # Close file

            return True

        elif self.file_format == FileFormat.ASCII:
            if abs(num - round(num)) < 1e-8:  # within 1e-8 of an integer so write an integer
                self.text_file.write(f"{int(round(num))} ")
            else:
                self.text_file.write(f"{num:.6f} ")
            if last_number:
                self.cr()
            return True

        self.num_list.append(num)

        def write_buffer():
            self.in_chain = False
            j = 0
            while j <= self.num_written:
                if not self.in_chain:
                    self.reading_repeat = (j < self.num_written and self.num_list[j] == self.num_list[j + 1])
                    self.chain_size = 0
                    self.in_chain = True

                if j == self.num_written:
                    self.chain_done = True
                else:
                    if self.reading_repeat:
                        self.chain_done = self.num_list[j] != self.num_list[j + 1]
                    else:
                        self.chain_done = self.num_list[j] == self.num_list[j + 1]
                        if j < self.num_written - 1:
                            self.chain_done = self.chain_done or (self.num_list[j + 1] == self.num_list[j + 2])

                self.chain_size += 1

                if self.chain_done:
                    self.write_stream.write(struct.pack('?', self.reading_repeat))
                    self.write_stream.write(struct.pack('H', self.chain_size))
                    if self.reading_repeat:
                        self.write_stream.write(struct.pack('f', self.num_list[j - 1]))
                    else:
                        for k in range(self.chain_size):
                            self.write_stream.write(struct.pack('f', self.num_list[j - k]))

                    self.chain_done = False
                    self.in_chain = False

                j += 1

        if self.num_written == self.buffer_size or last_number:
            write_buffer()
            self.num_written = 0
        else:
            self.num_written += 1

        return True
