
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from raster_files import TSLAMMInputFile, TSLAMMOutputFile, FileFormat 


"""
This script converts raster files from one format to another.
Supported input and output formats:
- ASCII (.asc, .txt)
- Binary (.slb)
- GeoTIFF (.tif, .tiff)
Functions:
    determine_file_format(file_path: str) -> Optional[FileFormat]:
        Determine the file format based on file extension.
    prompt_overwrite(file_path: str) -> bool:
        Prompt the user to confirm overwriting an existing file.
    main(input_file: str, output_file: str):
        Main function to handle the conversion process.
Usage:
    python convert_raster.py input_file output_file
Arguments:
    input_file: Path to the input raster file.
    output_file: Path to the output raster file.
Example:
    python convert_raster.py input.asc output.tif
"""



def determine_file_format(file_path: str) -> Optional[FileFormat]:
    """Determine the file format based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == '.asc' or ext == '.txt':
        return FileFormat.ASCII
    elif ext == '.slb':
        return FileFormat.BINARY
    elif ext == '.tif' or ext == '.tiff':
        return FileFormat.GEOTIFF
    return None

def prompt_overwrite(file_path: str) -> bool:
    while True:
        response = input(f"File '{file_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

def main(input_file: str, output_file: str):
    input_format = determine_file_format(input_file)
    output_format = determine_file_format(output_file)

    if input_format is None or output_format is None:
        print("Unsupported file format. Supported formats are: .asc, .txt, .slb, .tif")
        sys.exit(1)

    # Check if the output file already exists and prompt for overwrite
    if Path(output_file).exists():
        if not prompt_overwrite(output_file):
            print("Operation canceled by the user.")
            sys.exit(0)
    # Load the input file
    input_raster = TSLAMMInputFile(input_file)
    if not input_raster.prepare_file_for_reading():
        print(f"Error preparing input file: {input_file}")
        sys.exit(1)

    # Prepare the output file
    output_raster = TSLAMMOutputFile(input_raster.col, input_raster.row, input_raster.xll_corner,
                                     input_raster.yll_corner, input_raster.cell_size, output_file, output_format)

    success, err_msg = output_raster.write_header()
    if not success:
        print(f"Error preparing output file: {err_msg}")
        sys.exit(1)

    # Read data from input and write to output
    for row in range(input_raster.row):
        for col in range(input_raster.col):
            last_number = (row == input_raster.row - 1) and (col == input_raster.col - 1)
            num = input_raster.get_next_number()
            output_raster.write_next_number(num, last_number)

    # Clean up
    input_raster.close_file()

    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_raster.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
