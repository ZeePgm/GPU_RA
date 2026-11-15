// Windows process ownership for experiment launchers. No Python stays resident.
// Native contracts: CreateProcessW, STARTUPINFOEX HANDLE_LIST, job accounting,
// JOBOBJECT_EXTENDED_LIMIT_INFORMATION, and TerminateJobObject (Microsoft Learn).
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace GPU_RA
{
    public sealed class Result
    {
        public int ExitCode;
        public bool Cancelled;
        public uint ProcessId;
        public uint TotalProcesses;
        public uint ActiveProcessesAtEnd;
    }

    public static class RunJob
    {
        private const uint CreateSuspended = 0x00000004;
        private const uint CreateNoWindow = 0x08000000;
        private const uint ExtendedStartupInfoPresent = 0x00080000;
        private const uint StartfUseStdHandles = 0x00000100;
        private const uint JobLimitKillOnJobClose = 0x00002000;
        private const uint GenericRead = 0x80000000;
        private const uint GenericWrite = 0x40000000;
        private const uint FileShareRead = 0x00000001;
        private const uint FileShareWrite = 0x00000002;
        private const uint CreateNew = 1;
        private const uint OpenExisting = 3;
        private const uint FileAttributeNormal = 0x00000080;
        private const uint Infinite = 0xFFFFFFFF;
        private const uint WaitObject0 = 0;
        private const uint CancelExitCode = 130;
        private static readonly IntPtr InvalidHandle = new IntPtr(-1);
        private static readonly UIntPtr HandleListAttribute = new UIntPtr(0x00020002);
        private static readonly object RunLock = new object();

        // A launcher must not start its Python completion stage unless true.
        public static bool LastDrainConfirmed { get; private set; } = true;

        private enum JobInformationClass
        {
            BasicAccountingInformation = 1,
            ExtendedLimitInformation = 9
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct SecurityAttributes
        {
            public uint Length;
            public IntPtr SecurityDescriptor;
            [MarshalAs(UnmanagedType.Bool)] public bool InheritHandle;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct StartupInfo
        {
            public uint Size;
            public IntPtr Reserved;
            public IntPtr Desktop;
            public IntPtr Title;
            public uint X, Y, XSize, YSize, XCountChars, YCountChars;
            public uint FillAttribute, Flags;
            public ushort ShowWindow, Reserved2Size;
            public IntPtr Reserved2;
            public IntPtr StdInput, StdOutput, StdError;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct StartupInfoEx
        {
            public StartupInfo StartupInfo;
            public IntPtr AttributeList;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ProcessInformation
        {
            public IntPtr Process, Thread;
            public uint ProcessId, ThreadId;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JobBasicLimitInformation
        {
            public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters
        {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JobExtendedLimitInformation
        {
            public JobBasicLimitInformation BasicLimitInformation;
            public IoCounters IoInfo;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed, PeakJobMemoryUsed;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JobBasicAccountingInformation
        {
            public long TotalUserTime, TotalKernelTime;
            public long ThisPeriodTotalUserTime, ThisPeriodTotalKernelTime;
            public uint TotalPageFaultCount, TotalProcesses;
            public uint ActiveProcesses, TotalTerminatedProcesses;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObjectW(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetInformationJobObject(IntPtr job, JobInformationClass informationClass,
            ref JobExtendedLimitInformation information, uint length);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool QueryInformationJobObject(IntPtr job, JobInformationClass informationClass,
            out JobBasicAccountingInformation information, uint length, out uint returnedLength);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool TerminateJobObject(IntPtr job, uint exitCode);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CreateProcessW(string applicationName, StringBuilder commandLine,
            IntPtr processAttributes, IntPtr threadAttributes,
            [MarshalAs(UnmanagedType.Bool)] bool inheritHandles, uint creationFlags,
            IntPtr environment, string currentDirectory, ref StartupInfoEx startupInfo,
            out ProcessInformation processInformation);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern uint ResumeThread(IntPtr thread);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool TerminateProcess(IntPtr process, uint exitCode);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateFileW(string name, uint access, uint share,
            ref SecurityAttributes attributes, uint disposition, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FlushFileBuffers(IntPtr file);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool InitializeProcThreadAttributeList(IntPtr attributes, int count,
            uint flags, ref UIntPtr bytes);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool UpdateProcThreadAttribute(IntPtr attributes, uint flags,
            UIntPtr attribute, IntPtr value, UIntPtr bytes, IntPtr previousValue, IntPtr returnSize);
        [DllImport("kernel32.dll")]
        private static extern void DeleteProcThreadAttributeList(IntPtr attributes);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr handle);

        private static Win32Exception NativeError(string operation)
        {
            return new Win32Exception(Marshal.GetLastWin32Error(), operation);
        }

        private static void Close(ref IntPtr handle)
        {
            if (handle != IntPtr.Zero && handle != InvalidHandle) CloseHandle(handle);
            handle = IntPtr.Zero;
        }

        private static JobBasicAccountingInformation Accounting(IntPtr job)
        {
            JobBasicAccountingInformation information;
            uint returned;
            uint bytes = checked((uint)Marshal.SizeOf(typeof(JobBasicAccountingInformation)));
            if (!QueryInformationJobObject(job, JobInformationClass.BasicAccountingInformation,
                                           out information, bytes, out returned))
                throw NativeError("QueryInformationJobObject");
            if (returned < bytes) throw new InvalidOperationException("Incomplete job accounting information");
            return information;
        }

        private static JobBasicAccountingInformation Drain(IntPtr job)
        {
            for (;;)
            {
                JobBasicAccountingInformation information = Accounting(job);
                if (information.ActiveProcesses == 0)
                {
                    LastDrainConfirmed = true;
                    return information;
                }
                Thread.Sleep(25);
            }
        }

        private static bool CancelRequested(string cancelPath)
        {
            return !String.IsNullOrEmpty(cancelPath) &&
                   (File.Exists(cancelPath) || Directory.Exists(cancelPath));
        }

        private static IntPtr OpenOutput(string path, ref SecurityAttributes attributes)
        {
            IntPtr handle = CreateFileW(path, GenericWrite, FileShareRead, ref attributes,
                                        CreateNew, FileAttributeNormal, IntPtr.Zero);
            if (handle == InvalidHandle) throw NativeError("CreateFileW(CREATE_NEW): " + path);
            return handle;
        }

        // windowsArguments is list2cmdline(argv[1:]); argv[0] is added here.
        public static Result Run(string executable, string windowsArguments, string cwd,
                                 string stdoutPath, string stderrPath, string cancelPath)
        {
            lock (RunLock)
            {
                return RunOwned(executable, windowsArguments, cwd, stdoutPath, stderrPath, cancelPath);
            }
        }

        private static Result RunOwned(string executable, string windowsArguments, string cwd,
                                       string stdoutPath, string stderrPath, string cancelPath)
        {
            LastDrainConfirmed = true; // No child exists yet.
            if (Environment.OSVersion.Platform != PlatformID.Win32NT || IntPtr.Size != 8)
                throw new PlatformNotSupportedException("RunJob requires 64-bit Windows");
            if (String.IsNullOrWhiteSpace(executable) || !Path.IsPathFullyQualified(executable))
                throw new ArgumentException("Executable must be an absolute path", "executable");
            if (String.IsNullOrWhiteSpace(cwd) || !Path.IsPathFullyQualified(cwd))
                throw new ArgumentException("Working directory must be an absolute path", "cwd");
            executable = Path.GetFullPath(executable);
            cwd = Path.GetFullPath(cwd);
            if (!Directory.Exists(cwd)) throw new DirectoryNotFoundException(cwd);
            if (String.IsNullOrWhiteSpace(stdoutPath) || String.IsNullOrWhiteSpace(stderrPath))
                throw new ArgumentException("Two fresh output paths are required");
            stdoutPath = Path.GetFullPath(stdoutPath);
            stderrPath = Path.GetFullPath(stderrPath);
            if (String.Equals(stdoutPath, stderrPath, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("stdout and stderr require different fresh files");
            if (!String.IsNullOrEmpty(cancelPath)) cancelPath = Path.GetFullPath(cancelPath);
            windowsArguments = windowsArguments ?? String.Empty;
            if (executable.IndexOf('"') >= 0 || windowsArguments.IndexOf('\0') >= 0)
                throw new ArgumentException("Invalid executable or command line");
            string text = "\"" + executable + "\"" +
                          (windowsArguments.Length == 0 ? String.Empty : " " + windowsArguments);
            if (text.Length >= 32767) throw new ArgumentException("Windows command line exceeds 32766 characters");

            IntPtr job = IntPtr.Zero, stdout = IntPtr.Zero, stderr = IntPtr.Zero, stdin = IntPtr.Zero;
            IntPtr attributes = IntPtr.Zero, handleArray = IntPtr.Zero;
            bool attributesInitialized = false, assigned = false, resumed = false;
            ProcessInformation process = new ProcessInformation();
            try
            {
                job = CreateJobObjectW(IntPtr.Zero, null); // Unnamed, non-inheritable.
                if (job == IntPtr.Zero) throw NativeError("CreateJobObjectW");
                JobExtendedLimitInformation limits = new JobExtendedLimitInformation();
                // BREAKAWAY_OK and SILENT_BREAKAWAY_OK remain unset.
                limits.BasicLimitInformation.LimitFlags = JobLimitKillOnJobClose;
                if (!SetInformationJobObject(job, JobInformationClass.ExtendedLimitInformation,
                    ref limits, checked((uint)Marshal.SizeOf(typeof(JobExtendedLimitInformation)))))
                    throw NativeError("SetInformationJobObject(KILL_ON_JOB_CLOSE)");

                SecurityAttributes inherited = new SecurityAttributes();
                inherited.Length = checked((uint)Marshal.SizeOf(typeof(SecurityAttributes)));
                inherited.InheritHandle = true;
                stdout = OpenOutput(stdoutPath, ref inherited);
                stderr = OpenOutput(stderrPath, ref inherited);
                stdin = CreateFileW("NUL", GenericRead, FileShareRead | FileShareWrite, ref inherited,
                                    OpenExisting, FileAttributeNormal, IntPtr.Zero);
                if (stdin == InvalidHandle) throw NativeError("CreateFileW(NUL)");

                UIntPtr attributeBytes = UIntPtr.Zero;
                InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref attributeBytes);
                if (attributeBytes == UIntPtr.Zero) throw NativeError("InitializeProcThreadAttributeList(size)");
                attributes = Marshal.AllocHGlobal(checked((int)attributeBytes.ToUInt64()));
                if (!InitializeProcThreadAttributeList(attributes, 1, 0, ref attributeBytes))
                    throw NativeError("InitializeProcThreadAttributeList");
                attributesInitialized = true;
                IntPtr[] inheritedHandles = { stdin, stdout, stderr };
                int handleBytes = checked(inheritedHandles.Length * IntPtr.Size);
                handleArray = Marshal.AllocHGlobal(handleBytes);
                for (int index = 0; index < inheritedHandles.Length; ++index)
                    Marshal.WriteIntPtr(handleArray, index * IntPtr.Size, inheritedHandles[index]);
                if (!UpdateProcThreadAttribute(attributes, 0, HandleListAttribute, handleArray,
                    new UIntPtr(checked((uint)handleBytes)), IntPtr.Zero, IntPtr.Zero))
                    throw NativeError("UpdateProcThreadAttribute(HANDLE_LIST)");

                StartupInfoEx startup = new StartupInfoEx();
                startup.StartupInfo.Size = checked((uint)Marshal.SizeOf(typeof(StartupInfoEx)));
                startup.StartupInfo.Flags = StartfUseStdHandles;
                startup.StartupInfo.StdInput = stdin;
                startup.StartupInfo.StdOutput = stdout;
                startup.StartupInfo.StdError = stderr;
                startup.AttributeList = attributes;
                StringBuilder mutableCommandLine = new StringBuilder(text, text.Length + 1);
                if (!CreateProcessW(executable, mutableCommandLine, IntPtr.Zero, IntPtr.Zero, true,
                    CreateSuspended | CreateNoWindow | ExtendedStartupInfoPresent,
                    IntPtr.Zero, cwd, ref startup, out process))
                    throw NativeError("CreateProcessW(CREATE_SUSPENDED)");
                LastDrainConfirmed = false;
                if (!AssignProcessToJobObject(job, process.Process))
                    throw NativeError("AssignProcessToJobObject");
                assigned = true;

                bool cancelled = CancelRequested(cancelPath);
                if (cancelled)
                {
                    if (!TerminateJobObject(job, CancelExitCode)) throw NativeError("TerminateJobObject(cancel)");
                }
                else
                {
                    if (ResumeThread(process.Thread) == UInt32.MaxValue) throw NativeError("ResumeThread");
                    resumed = true;
                }

                JobBasicAccountingInformation final;
                for (;;)
                {
                    final = Accounting(job);
                    if (final.ActiveProcesses == 0) break;
                    if (!cancelled && CancelRequested(cancelPath))
                    {
                        cancelled = true;
                        if (!TerminateJobObject(job, CancelExitCode)) throw NativeError("TerminateJobObject(cancel)");
                    }
                    Thread.Sleep(25);
                }
                LastDrainConfirmed = true;
                if (WaitForSingleObject(process.Process, Infinite) != WaitObject0)
                    throw NativeError("WaitForSingleObject(root after job drained)");
                uint exitCode;
                if (!GetExitCodeProcess(process.Process, out exitCode)) throw NativeError("GetExitCodeProcess");
                if (!FlushFileBuffers(stdout)) throw NativeError("FlushFileBuffers(stdout)");
                if (!FlushFileBuffers(stderr)) throw NativeError("FlushFileBuffers(stderr)");
                return new Result { ExitCode = unchecked((int)exitCode), Cancelled = cancelled,
                    ProcessId = process.ProcessId, TotalProcesses = final.TotalProcesses,
                    ActiveProcessesAtEnd = final.ActiveProcesses };
            }
            catch (Exception original)
            {
                original.Data["ProcessId"] = process.ProcessId;
                try
                {
                    if (assigned)
                    {
                        if (!TerminateJobObject(job, CancelExitCode))
                        {
                            int terminationError = Marshal.GetLastWin32Error();
                            if (Accounting(job).ActiveProcesses != 0)
                                throw new Win32Exception(terminationError, "TerminateJobObject(exception cleanup)");
                        }
                        Drain(job); // Confirm the complete job, not only its root.
                    }
                    else if (process.Process != IntPtr.Zero)
                    {
                        // Assignment failed before ResumeThread: no user code or
                        // descendants could run. Terminate and await this root.
                        if (resumed) throw new InvalidOperationException("Unowned process was resumed");
                        if (WaitForSingleObject(process.Process, 0) != WaitObject0)
                        {
                            if (!TerminateProcess(process.Process, CancelExitCode))
                                throw NativeError("TerminateProcess(unassigned suspended root)");
                            if (WaitForSingleObject(process.Process, Infinite) != WaitObject0)
                                throw NativeError("WaitForSingleObject(unassigned suspended root)");
                        }
                        LastDrainConfirmed = true;
                    }
                }
                catch (Exception cleanup)
                {
                    LastDrainConfirmed = false;
                    throw new AggregateException("Execution failed and process drain could not be confirmed", original, cleanup);
                }
                throw;
            }
            finally
            {
                if (attributesInitialized) DeleteProcThreadAttributeList(attributes);
                if (attributes != IntPtr.Zero) Marshal.FreeHGlobal(attributes);
                if (handleArray != IntPtr.Zero) Marshal.FreeHGlobal(handleArray);
                Close(ref process.Thread);
                Close(ref process.Process);
                Close(ref stdin);
                Close(ref stdout);
                Close(ref stderr);
                Close(ref job); // Last-resort kill-on-close if confirmation failed.
            }
        }
    }
}
