"""Parent-process deadlines and resource checks include preparation and scoring."""
import os
import signal
import subprocess
import time


class ProcessTree:
    """Own a child tree using a Windows Job or a POSIX process group.

    Windows venv launchers spawn another python.exe. Terminating just the
    launcher leaves that interpreter alive; a Job retains ownership of both.
    """
    def __init__(self, command, *, memory_limit_bytes=0, **kwargs):
        self.job = None
        self._memory_handles = {}
        if os.name == 'nt':
            import ctypes as ct
            from ctypes import wintypes as wt
            class Basic(ct.Structure):
                _fields_ = [('process_time', ct.c_longlong), ('job_time', ct.c_longlong),
                            ('flags', wt.DWORD), ('min_ws', ct.c_size_t), ('max_ws', ct.c_size_t),
                            ('active_processes', wt.DWORD), ('affinity', ct.c_size_t),
                            ('priority', wt.DWORD), ('scheduling', wt.DWORD)]
            class Extended(ct.Structure):
                _fields_ = [('basic', Basic), ('io', ct.c_ulonglong*6),
                            ('process_memory', ct.c_size_t), ('job_memory', ct.c_size_t),
                            ('peak_process_memory', ct.c_size_t), ('peak_job_memory', ct.c_size_t)]
            self._extended = Extended
            self.kernel = ct.WinDLL('kernel32', use_last_error=True)
            self.kernel.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
            self.kernel.CreateJobObjectW.restype = wt.HANDLE
            self.kernel.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
            self.kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
            self.kernel.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
            self.kernel.CloseHandle.argtypes = [wt.HANDLE]
            self.kernel.QueryInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p]
            self.job = self.kernel.CreateJobObjectW(None, None)
            if not self.job:
                raise ct.WinError(ct.get_last_error())
            limits = Extended()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if memory_limit_bytes:
                limits.basic.flags |= 0x200  # JOB_OBJECT_LIMIT_JOB_MEMORY: committed memory, entire tree
                limits.job_memory = memory_limit_bytes
            if not self.kernel.SetInformationJobObject(self.job, 9, ct.byref(limits), ct.sizeof(limits)):
                self.close()
                raise ct.WinError(ct.get_last_error())
        try:
            self.proc = subprocess.Popen(command, start_new_session=os.name != 'nt', **kwargs)
            if self.job and not self.kernel.AssignProcessToJobObject(self.job, int(self.proc._handle)):
                import ctypes as ct
                error = ct.WinError(ct.get_last_error())
                self.proc.terminate()
                self.proc.wait(timeout=5)
                raise error
        except BaseException:
            self.close()
            raise

    def memory_sample(self):
        """Query only owned processes. RSS includes shared pages; private is commit."""
        if not self.job:
            raise NotImplementedError('Native tree memory telemetry requires Windows')
        import ctypes as ct
        from ctypes import wintypes as wt
        class Pids(ct.Structure):
            _fields_ = [('assigned', wt.DWORD), ('count', wt.DWORD), ('pids', ct.c_size_t*256)]
        class Counters(ct.Structure):
            _fields_ = [('cb', wt.DWORD), ('faults', wt.DWORD)]+[(name, ct.c_size_t) for name in
                ('peak_rss', 'rss', 'peak_paged_pool', 'paged_pool', 'peak_nonpaged_pool', 'nonpaged_pool',
                 'pagefile', 'peak_pagefile', 'private')]
        pids, limits = Pids(), self._extended()
        for kind, value in ((3, pids), (9, limits)):
            if not self.kernel.QueryInformationJobObject(self.job, kind, ct.byref(value), ct.sizeof(value), None):
                raise ct.WinError(ct.get_last_error())
        self.kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        self.kernel.OpenProcess.restype = wt.HANDLE
        self.kernel.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ct.c_void_p, wt.DWORD]
        self.kernel.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD)]
        self.kernel.GetProcessHandleCount.argtypes = [wt.HANDLE, ct.POINTER(wt.DWORD)]
        self.kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        rows = []
        for pid in list(pids.pids)[:pids.count]:
            handle = self._memory_handles.get(pid)
            if not handle:
                handle = self.kernel.OpenProcess(0x1000 | 0x100000, False, pid)
            if not handle:
                error = ct.get_last_error()
                # A process can become non-openable while its final Job entry
                # is being removed. Only ignore it after membership disappears.
                for _ in range(5):
                    current = Pids()
                    if not self.kernel.QueryInformationJobObject(self.job, 3, ct.byref(current), ct.sizeof(current), None):
                        raise ct.WinError(ct.get_last_error())
                    if pid not in list(current.pids)[:current.count]:
                        break
                    time.sleep(.01)
                else:
                    raise ct.WinError(error)
                continue
            self._memory_handles[pid] = handle
            if self.kernel.WaitForSingleObject(handle, 0) == 0:
                continue
            value = Counters()
            value.cb = ct.sizeof(value)
            name = ct.create_unicode_buffer(32768)
            size, handles = wt.DWORD(len(name)), wt.DWORD()
            queries = (
                lambda: self.kernel.K32GetProcessMemoryInfo(handle, ct.byref(value), ct.sizeof(value)),
                lambda: self.kernel.QueryFullProcessImageNameW(handle, 0, name, ct.byref(size)),
                lambda: self.kernel.GetProcessHandleCount(handle, ct.byref(handles)),
            )
            for query in queries:
                succeeded = False
                exited = False
                for _ in range(3):
                    if query():
                        succeeded = True
                        break
                    error = ct.get_last_error()
                    # PEB initialization and process teardown can briefly deny
                    # queries before the process handle becomes signalled.
                    if self.kernel.WaitForSingleObject(handle, 100) == 0:
                        exited = True
                        break
                if exited:
                    break
                if not succeeded:
                    raise ct.WinError(error)
            else:
                rows.append(dict(pid=pid, name=os.path.basename(name.value), handles=handles.value,
                                 rss_bytes=value.rss, private_commit_bytes=value.private))
        return dict(processes=rows, rss_bytes=sum(r['rss_bytes'] for r in rows),
                    private_commit_bytes=sum(r['private_commit_bytes'] for r in rows),
                    peak_job_commit_bytes=limits.peak_job_memory,
                    job_commit_limit_bytes=limits.job_memory)

    def stop(self):
        if self.job:
            if not self.kernel.TerminateJobObject(self.job, 1):
                import ctypes as ct
                raise ct.WinError(ct.get_last_error())
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name != 'nt':
                os.killpg(self.proc.pid, signal.SIGKILL)
            else:
                self.proc.kill()
            self.proc.wait(timeout=5)

    def close(self):
        for handle in self._memory_handles.values():
            self.kernel.CloseHandle(handle)
        self._memory_handles.clear()
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None
        elif hasattr(self, 'proc') and os.name != 'nt':
            # Also stop descendants if their parent exited before polling.
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def supervise(command, *, seconds, log, check=None):
    if seconds < 1:
        raise ValueError('Positive wall-clock budget required')
    start = time.monotonic()
    tree = ProcessTree(command, stdout=log, stderr=subprocess.STDOUT)
    proc = tree.proc
    status = 'FINISHED'
    try:
        while proc.poll() is None:
            remaining = seconds-(time.monotonic()-start)
            if remaining <= 0:
                status = 'HARD_TIME_LIMIT'
                break
            if check:
                check()
            try:
                proc.wait(timeout=min(1., remaining))
            except subprocess.TimeoutExpired:
                pass
    except Exception as error:
        status = 'RESOURCE_LIMIT_OR_INTERRUPTION'
        failure = type(error).__name__
    finally:
        try:
            if proc.poll() is None:
                tree.stop()
        finally:
            tree.close()
    return dict(status=status if proc.returncode == 0 or status != 'FINISHED' else 'FAILED',
                exit_code=proc.returncode, elapsed_seconds=time.monotonic()-start, limit_seconds=seconds,
                resource_error=locals().get('failure'))
