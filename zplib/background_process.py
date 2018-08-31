# This code is licensed under the MIT License (see LICENSE file for details)

import os
import pathlib
import sys
import traceback

import lockfile

def run_in_background(function, *args, logfile=None, nice=None, delete_logfile=True, lock=None, **kws):
    """Run a function in a background process (by forking the foreground process)

    Parameters:
        function: function to run.
        *args: arguments to function.
        logfile: if not None, redirect stderr and stdout to this file. The file
            will be opened in append mode (so existing logs will be added to)
            and the PID of the background process will be written to the file
            before running the function.
        nice: if not None, level to renice the forked process to.
        delete_logfile: if True, the logfile will be deleted after the function
            exits, except in case of an exception, where the file will be retained
            to aid in debugging. (It will contain the traceback information.)
        lock: if not None, path to a file lock to acquire before running the
            function (to prevent multiple backround jobs from running at once.)
        **kws: keyword arguments to function.
    """
    if _detach_process_context():
        # _detach_process_context returns True in the parent process and False
        # in the child process.
        return
    try:
        if logfile is None:
            log = None
            delete_logfile = False
        else:
            logfile = pathlib.Path(logfile)
            log = logfile.open('a')
            log.write(str(os.getpid())+'\n')
        # close standard streams so the process can still run even if the
        # controlling terminal window is closed.
        _redirect_stream(sys.stdin, None)
        _redirect_stream(sys.stdout, log)
        _redirect_stream(sys.stderr, log)
        if nice is not None:
            os.nice(nice)
        if lock is not None:
            lock = lockfile.LockFile(lock)
            lock.acquire()
        function(*args, **kws)
    except:
        # don't remove the logfile...
        traceback.print_exc()
        delete_logfile = False
    finally:
        log.close()
        if delete_logfile:
            logfile.unlink()
        if lock is not None:
            lock.release()
        # if we don't exit with os._exit(), then if this function was called from
        # ipython, the child will try to return back to the ipython shell, with all
        # manner of hilarity ensuing.
        os._exit(0)

def _detach_process_context():
    """Detach the process context from parent and session.

    Detach from the parent process and session group, allowing the parent to
    keep running while the child continues running. Uses the standard UNIX
    double-fork strategy to isolate the child.
    """
    if _fork_carefully() > 0:
        # parent: return now
        return True
    # child: fork again
    os.setsid()
    if _fork_carefully() > 0:
        # exit parent
        os._exit(0)
    return False

def _fork_carefully():
    try:
        return os.fork()
    except OSError as e:
        raise RuntimeError('Fork failed: [{}] {}'.format(e.errno, e.strerror))

def _redirect_stream(src, dst):
    if dst is None:
        dst_fd = os.open(os.devnull, os.O_RDWR)
    else:
        dst_fd = dst.fileno()
    os.dup2(dst_fd, src.fileno())