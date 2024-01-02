#! /usr/bin/env python3

import os
environment = {
    "ALLUSERSPROFILE":r"C:\ProgramData",
    "APPDATA":r"C:\Users\christopher.lorton\AppData\Roaming",
    "CHROME_CRASHPAD_PIPE_NAME":r"\\.\pipe\crashpad_7276_HXQLODAPRVOIRLBF",
    "CommandPromptType":"Native",
    "CommonProgramFiles":r"C:\Program Files\Common Files",
    "CommonProgramFiles(x86)":r"C:\Program Files (x86)\Common Files",
    "CommonProgramW6432":r"C:\Program Files\Common Files",
    "COMPUTERNAME":"BMGF-R913QDD7",
    "ComSpec":r"C:\windows\system32\cmd.exe",
    "CUDA_PATH":r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
    "CUDA_PATH_V12_3":r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
    "DevEnvDir":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\\",
    "DriverData":r"C:\Windows\System32\Drivers\DriverData",
    "ExtensionSdkDir":r"C:\Program Files (x86)\Microsoft SDKs\Windows Kits\10\ExtensionSDKs",
    "Framework40Version":"v4.0",
    "FrameworkDir":r"C:\Windows\Microsoft.NET\Framework64\\",
    "FrameworkDir64":r"C:\Windows\Microsoft.NET\Framework64\\",
    "FrameworkVersion":"v4.0.30319",
    "FrameworkVersion64":"v4.0.30319",
    "FSHARPINSTALLDIR":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\FSharp\Tools",
    "HOMEDRIVE":"C:",
    "HOMEPATH":r"\Users\christopher.lorton",
    "INCLUDE":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\ATLMFC\include;C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\include;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\ucrt;C:\Program Files (x86)\Windows Kits\10\\include\10.0.19041.0\\shared;C:\Program Files (x86)\Windows Kits\10\\include\10.0.19041.0\\um;C:\Program Files (x86)\Windows Kits\10\\include\10.0.19041.0\\winrt;C:\Program Files (x86)\Windows Kits\10\\include\10.0.19041.0\\cppwinrt",
    "LIB":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\ATLMFC\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\lib\x64;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\lib\um\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.19041.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\\lib\10.0.19041.0\\um\x64",
    "LIBPATH":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\ATLMFC\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\lib\x86\store\references;C:\Program Files (x86)\Windows Kits\10\UnionMetadata\10.0.19041.0;C:\Program Files (x86)\Windows Kits\10\References\10.0.19041.0;C:\Windows\Microsoft.NET\Framework64\v4.0.30319",
    "LOCALAPPDATA":r"C:\Users\christopher.lorton\AppData\Local",
    "LOGONSERVER":r"\\BMGF-R913QDD7",
    "NETFXSDKDir":r"C:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\\",
    "NUMBER_OF_PROCESSORS":"12",
    "OneDrive":r"C:\Users\christopher.lorton\OneDrive - Bill & Melinda Gates Foundation",
    "OneDriveCommercial":r"C:\Users\christopher.lorton\OneDrive - Bill & Melinda Gates Foundation",
    "ORIGINAL_XDG_CURRENT_DESKTOP":"undefined",
    "OS":"Windows_NT",
    "Path":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\\Extensions\Microsoft\IntelliCode\CLI;C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\bin\HostX64\x64;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\VC\VCPackages;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\TestWindow;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\bin\Roslyn;C:\Program Files\Microsoft Visual Studio\2022\Professional\Team Tools\Performance Tools\x64;C:\Program Files\Microsoft Visual Studio\2022\Professional\Team Tools\Performance Tools;C:\Program Files (x86)\Microsoft SDKs\Windows\v10.0A\bin\NETFX 4.8 Tools\x64\;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\FSharp\Tools;C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\\x64;C:\Program Files (x86)\Windows Kits\10\bin\\x64;C:\Program Files\Microsoft Visual Studio\2022\Professional\\MSBuild\Current\Bin\amd64;C:\Windows\Microsoft.NET\Framework64\v4.0.30319;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\;c:\src\piecoodah\.venv\Scripts;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\libnvvp;C:\Program Files\Python312\Scripts\;C:\Program Files\Python312\;C:\Program Files (x86)\Razer\ChromaBroadcast\bin;C:\Program Files\Razer\ChromaBroadcast\bin;C:\windows\system32;C:\windows;C:\windows\System32\Wbem;C:\windows\System32\WindowsPowerShell\v1.0\;C:\windows\System32\OpenSSH\;C:\Program Files\Git\cmd;C:\Program Files\dotnet\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\windows\system32\config\systemprofile\AppData\Local\Microsoft\WindowsApps;;C:\Program Files\Docker\Docker\resources\bin;C:\Program Files\Cloudflare\Cloudflare WARP\;C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.1\;C:\Users\christopher.lorton\AppData\Local\Programs\Python\Python39\Scripts\;C:\Users\christopher.lorton\AppData\Local\Programs\Python\Python39\;C:\Users\christopher.lorton\AppData\Local\Microsoft\WindowsApps;;C:\Users\christopher.lorton\.dotnet\tools;C:\Users\christopher.lorton\AppData\Local\Programs\Microsoft VS Code\bin;C:\Users\christopher.lorton\AppData\Local\Pandoc\;C:\Users\christopher.lorton\AppData\Local\Programs\Julia-1.8.4\bin;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\VC\Linux\bin\ConnectionManagerExe",
    "PATHEXT":".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.PY;.PYW",
    "Platform":"x64",
    "PROCESSOR_ARCHITECTURE":"AMD64",
    "PROCESSOR_IDENTIFIER":"Intel64 Family 6 Model 165 Stepping 2, GenuineIntel",
    "PROCESSOR_LEVEL":"6",
    "PROCESSOR_REVISION":"a502",
    "ProgramData":r"C:\ProgramData",
    "ProgramFiles":r"C:\Program Files",
    "ProgramFiles(x86)":r"C:\Program Files (x86)",
    "ProgramW6432":r"C:\Program Files",
    "PROMPT":"(.venv) $P$G",
    "PSModulePath":r"C:\Program Files\WindowsPowerShell\Modules;C:\windows\system32\WindowsPowerShell\v1.0\Modules",
    "PUBLIC":r"C:\Users\Public",
    "RTOOLS42_HOME":r"C:\rtools42",
    "SESSIONNAME":"Console",
    "SystemDrive":"C:",
    "SystemRoot":r"C:\windows",
    "TEMP":r"C:\Users\CHRIST~1.LOR\AppData\Local\Temp",
    "TMP":r"C:\Users\CHRIST~1.LOR\AppData\Local\Temp",
    "UCRTVersion":"10.0.19041.0",
    "UniversalCRTSdkDir":r"C:\Program Files (x86)\Windows Kits\10\\",
    "USERDOMAIN":"BMGF-R913QDD7",
    "USERDOMAIN_ROAMINGPROFILE":"BMGF-R913QDD7",
    "USERNAME":"christopher.lorton",
    "USERPROFILE":r"C:\Users\christopher.lorton",
    "VCIDEInstallDir":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\VC\\",
    "VCINSTALLDIR":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\\",
    "VCToolsInstallDir":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.30.30705\\",
    "VCToolsRedistDir":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Redist\MSVC\14.30.30704\\",
    "VCToolsVersion":"14.30.30705",
    "VIRTUAL_ENV":r"c:\src\piecoodah\.venv",
    "VIRTUAL_ENV_PROMPT":"(.venv) ",
    "VisualStudioVersion":"17.0",
    "VS170COMNTOOLS":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\\",
    "VSCMD_ARG_app_plat":"Desktop",
    "VSCMD_ARG_HOST_ARCH":"x64",
    "VSCMD_ARG_TGT_ARCH":"x64",
    "VSCMD_VER":"17.0.4",
    "VSINSTALLDIR":r"C:\Program Files\Microsoft Visual Studio\2022\Professional\\",
    "windir":r"C:\windows",
    "WindowsLibPath":r"C:\Program Files (x86)\Windows Kits\10\UnionMetadata\10.0.19041.0;C:\Program Files (x86)\Windows Kits\10\References\10.0.19041.0",
    "WindowsSdkBinPath":r"C:\Program Files (x86)\Windows Kits\10\bin\\",
    "WindowsSdkDir":r"C:\Program Files (x86)\Windows Kits\10\\",
    "WindowsSDKLibVersion":r"10.0.19041.0\\",
    "WindowsSdkVerBinPath":r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\\",
    "WindowsSDKVersion":r"10.0.19041.0\\",
    "WindowsSDK_ExecutablePath_x64":r"C:\Program Files (x86)\Microsoft SDKs\Windows\v10.0A\bin\NETFX 4.8 Tools\x64\\",
    "WindowsSDK_ExecutablePath_x86":r"C:\Program Files (x86)\Microsoft SDKs\Windows\v10.0A\bin\NETFX 4.8 Tools\\",
    "WSLENV":"WT_SESSION:WT_PROFILE_ID:",
    "WT_PROFILE_ID":"{0caa0dad-35be-5f56-a8ff-afceeeaa6101}",
    "WT_SESSION":"81d70815-e9da-4a82-b588-f9278dd01ccb",
    "ZES_ENABLE_SYSMAN":"1",
    "_OLD_VIRTUAL_PATH":r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\libnvvp;C:\Program Files\Python312\Scripts\;C:\Program Files\Python312\;C:\Program Files (x86)\Razer\ChromaBroadcast\bin;C:\Program Files\Razer\ChromaBroadcast\bin;C:\windows\system32;C:\windows;C:\windows\System32\Wbem;C:\windows\System32\WindowsPowerShell\v1.0\;C:\windows\System32\OpenSSH\;C:\Program Files\Git\cmd;C:\Program Files\dotnet\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\windows\system32\config\systemprofile\AppData\Local\Microsoft\WindowsApps;;C:\Program Files\Docker\Docker\resources\bin;C:\Program Files\Cloudflare\Cloudflare WARP\;C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.1\;C:\Users\christopher.lorton\AppData\Local\Programs\Python\Python39\Scripts\;C:\Users\christopher.lorton\AppData\Local\Programs\Python\Python39\;C:\Users\christopher.lorton\AppData\Local\Microsoft\WindowsApps;;C:\Users\christopher.lorton\.dotnet\tools;C:\Users\christopher.lorton\AppData\Local\Programs\Microsoft VS Code\bin;C:\Users\christopher.lorton\AppData\Local\Pandoc\;C:\Users\christopher.lorton\AppData\Local\Programs\Julia-1.8.4\bin",
    "_OLD_VIRTUAL_PROMPT":"$P$G",
    "__DOTNET_ADD_64BIT":"1",
    "__DOTNET_PREFERRED_BITNESS":"64",
    "__VSCMD_PREINIT_PATH":r"c:\src\piecoodah\.venv\Scripts;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\libnvvp;C:\Program Files\Python312\Scripts\;C:\Program Files\Python312\;C:\Program Files (x86)\Razer\ChromaBroadcast\bin;C:\Program Files\Razer\ChromaBroadcast\bin;C:\windows\system32;C:\windows;C:\windows\System32\Wbem;C:\windows\System32\WindowsPowerShell\v1.0\;C:\windows\System32\OpenSSH\;C:\Program Files\Git\cmd;C:\Program Files\dotnet\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\windows\system32\config\systemprofile\AppData\Local\Microsoft\WindowsApps;;C:\Program Files\Docker\Docker\resources\bin;C:\Program Files\Cloudflare\Cloudflare WARP\;C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.1\;C:\Users\christopher.lorton\AppData\Local\Programs\Python\Python39\Scripts\;C:\Users\christopher.lorton\AppData\Local\Programs\Python\Python39\;C:\Users\christopher.lorton\AppData\Local\Microsoft\WindowsApps;;C:\Users\christopher.lorton\.dotnet\tools;C:\Users\christopher.lorton\AppData\Local\Programs\Microsoft VS Code\bin;C:\Users\christopher.lorton\AppData\Local\Pandoc\;C:\Users\christopher.lorton\AppData\Local\Programs\Julia-1.8.4\bin",
    "__VSCMD_script_err_count":"0",
    "TERM_PROGRAM":"vscode",
    "TERM_PROGRAM_VERSION":"1.85.1",
    "LANG":"en_US.UTF-8",
    "COLORTERM":"truecolor",
    "GIT_ASKPASS":r"c:\Users\christopher.lorton\AppData\Local\Programs\Microsoft VS Code\resources\app\extensions\git\dist\askpass.sh",
    "VSCODE_GIT_ASKPASS_NODE":r"C:\Users\christopher.lorton\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "VSCODE_GIT_ASKPASS_EXTRA_ARGS":"--ms-enable-electron-run-as-node",
    "VSCODE_GIT_ASKPASS_MAIN":r"c:\Users\christopher.lorton\AppData\Local\Programs\Microsoft VS Code\resources\app\extensions\git\dist\askpass-main.js",
    "VSCODE_GIT_IPC_HANDLE":r"\\.\pipe\vscode-git-5499a98bf4-sock",
}
os.environ.update(environment)

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import numba as nb
import polars as pl
from tqdm import tqdm

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel

def kernels():

    filename = Path(__name__).parent / "fullseir.cu"
    mod = SourceModule(filename.read_text())
    infect = mod.get_function("infect")
    infection_update = mod.get_function("infection_update")
    exposure_update = mod.get_function("exposure_update")
    transmission_update = mod.get_function("transmission_update")

    sumnz = ReductionKernel(
        np.uint32,                      # output type
        neutral="0",                    # neutral element
        reduce_expr="a+b",              # reduce expression
        map_expr="(x[i] != 0) ? 1 : 0", # map expression
        arguments="unsigned char *x"    # arguments
        )

    return infect, infection_update, exposure_update, transmission_update, sumnz


def initialize_prng_states(num_states, seed):

    """Initialize the PRNG states on the CPU (it cannot run in parallel) and copy to GPU."""

    @nb.jit((nb.uint64[:,:], nb.int64, nb.uint64), nopython=True, cache=True, nogil=True, fastmath=True)
    def init_state(states, index, seed):
        z = seed + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint32(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint32(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint32(31))
        states[index,:] = z
        return

    @nb.jit((nb.uint64, nb.uint32), nopython=True, cache=True, nogil=True, fastmath=True)
    def rotl(x, k):
        return (x << k) | (x >> np.uint32(64 - k))

    @nb.jit((nb.uint64[:,:], nb.int64), nopython=True, cache=True, nogil=True, fastmath=True)
    def gonext(states, index):
        s0, s1 = states[index]
        result = s0 + s1

        s1 ^= s0
        states[index,0] = np.uint64(rotl(s0, np.uint32(55))) ^ s1 ^ (s1 << np.uint32(14))
        states[index,1] = np.uint64(rotl(s1, np.uint32(36)))

        return result

    @nb.jit((nb.uint64[:,:], nb.int64), nopython=True, cache=True, nogil=True, fastmath=True)
    def jumpforward(states, index):
        jump = np.array((np.uint64(0xbeac0467eba5facb), np.uint64(0xd86b048b86aa9922)), dtype=np.uint64)
        s0 = np.uint64(0)
        s1 = np.uint64(0)
        for i in range(2):
            for b in range(64):
                if jump[i] & (np.uint64(1) << np.uint32(b)):
                    s0 ^= states[index,0]
                    s1 ^= states[index,1]
                gonext(states, index)

        states[index,0] = s0
        states[index,1] = s1
        return

    states_cpu = np.empty((num_states, 2), dtype=np.uint64)
    init_state(states_cpu, 0, seed)
    for i in tqdm(range(1, num_states)):
        states_cpu[i] = states_cpu[i-1]
        jumpforward(states_cpu, i)
    states_gpu = gpuarray.to_gpu(states_cpu)

    return states_gpu


def record_state(timestep, sumnz, sus_gpu, exp_gpu, inf_gpu, pop_size, results):

    sus = np.uint32(sumnz(sus_gpu).get())
    exp = np.uint32(sumnz(exp_gpu).get())
    inf = np.uint32(sumnz(inf_gpu).get())
    rec = np.uint32(pop_size - sus - exp - inf)
    results[timestep,:] = np.array([timestep, sus, exp, inf, rec])

    return


def main(params):

    # compile the GPU kernels
    infect, infection_update, exposure_update, transmission_update, sumnz = kernels()

    # initialize the PRNG states on GPU
    num_states = np.ceil(params.pop_size / params.prng_stride).astype(np.uint32)
    print(f"Allocating {num_states:,} PRNG states on GPU for {params.pop_size:,} agents...")
    prng_states = initialize_prng_states(num_states, params.seed)

    # allocate properties on the GPU
    sus_gpu = gpuarray.ones(params.pop_size, dtype=np.uint8)
    exp_gpu = gpuarray.zeros(params.pop_size, dtype=np.uint8)
    inf_gpu = gpuarray.zeros(params.pop_size, dtype=np.uint8)

    # allocate results on the CPU
    # 5 columns: timestep, S, E, I, R
    results = np.zeros((params.timesteps+1, 5), dtype=np.uint32)

    NUM_THREADS = 256
    NUM_BLOCKS = int(np.ceil(params.pop_size / NUM_THREADS))

    # infect initial agents
    targets_gpu = gpuarray.to_gpu(np.random.choice(params.pop_size, size=params.initial, replace=False).astype(np.uint32).reshape((10,1)))
    infect(
        params.initial, targets_gpu,
        prng_states,
        sus_gpu,
        inf_gpu, params.inf_mean, params.inf_std,
        block=(int(params.initial), 1, 1), grid=(1, 1, 1), shared=int(params.initial*8)
        )

    # run the simulation
    tstart = datetime.now()
    record_state(0, sumnz, sus_gpu, exp_gpu, inf_gpu, params.pop_size, results)
    for timestep in tqdm(range(params.timesteps)):

        infection_update(
            params.pop_size,
            inf_gpu,
            block=(NUM_THREADS, 1, 1), grid=(NUM_BLOCKS, 1, 1))

        exposure_update(
            params.pop_size,
            exp_gpu,
            prng_states, params.prng_stride,
            inf_gpu, params.inf_mean, params.inf_std,
            block=(NUM_THREADS, 1, 1), grid=(NUM_BLOCKS, 1, 1), shared=NUM_THREADS * 16)

        contagion = np.uint32(sumnz(inf_gpu).get())
        force = params.beta * contagion / params.pop_size
        transmission_update(
            np.uint64(params.pop_size),
            prng_states, np.uint64(params.prng_stride),
            np.float64(force),
            sus_gpu,
            exp_gpu, np.float64(params.exp_mean), np.float64(params.exp_std),
            block=(NUM_THREADS, 1, 1), grid=(NUM_BLOCKS, 1, 1), shared=NUM_THREADS * 16)

        contagion = np.uint32(sumnz(inf_gpu).get())
        record_state(timestep+1, sumnz, sus_gpu, exp_gpu, inf_gpu, params.pop_size, results)
    tfinish = datetime.now()
    print(f"Simulation took {tfinish - tstart}.")

    # convert results to a Polars DataFrame
    df = pl.DataFrame(results, schema=["timestep", "S", "E", "I", "R"])
    # write to CSV
    print(f"Writing results to fullseir.csv...")
    df.write_csv("fullseir.csv")

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--timesteps", type=np.uint32, default=np.uint32(365), help="number of timesteps to run")
    parser.add_argument("-p", "--pop_size", type=np.uint32, default=np.uint32(1_000_000), help="population size")
    parser.add_argument("-i", "--initial", type=np.uint32, default=np.uint32(10), help="initial number of infected agents")
    parser.add_argument("-s", "--prng_stride", type=np.uint32, default=np.uint32(16), help="stride for PRNG states (# of agents per PRNG)")
    parser.add_argument("-S", "--seed", type=np.uint32, default=np.uint32(20231231), help="seed for PRNG")
    parser.add_argument("-r", "--r_naught", type=np.float32, default=np.float32(1.52), help="basic reproduction number")
    parser.add_argument("-m", "--inf_mean", type=np.float32, default=np.float32(5.0), help="mean of incubation period (days)")
    parser.add_argument("-d", "--inf_std", type=np.float32, default=np.float32(1.0), help="standard deviation of incubation period (days)")
    parser.add_argument("-M", "--exp_mean", type=np.float32, default=np.float32(4.0), help="mean of infectious period (days)")
    parser.add_argument("-D", "--exp_std", type=np.float32, default=np.float32(1.0), help="standard deviation of infectious period (days)")
    args = parser.parse_args()
    args.beta = args.r_naught / args.inf_mean
    main(args)
