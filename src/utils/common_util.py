"""Define the utility functions for general purpose."""

from contextlib import contextmanager
import shutil



def copy_file(src_path, dest_dir):
    """Copy file from source path to destination directory.

    Args:
        src_path: A string indicates the source file path.
        dest_dir: A string indicates the destination file directory.

    Raises:
        FileNotFoundError: If the `src_path` or `dest_dir` is not existing.
    """
    try:
        shutil.copy(src_path, dest_dir)
    except FileNotFoundError as e:
        logger.exception(
            f'Source file {src_path} or destination file directory {dest_dir} is not existing.'
        )
        raise e
    except shutil.SameFileError:
        logger.warning(f'{src_path} and {dest_dir} are the same file.')
        pass


