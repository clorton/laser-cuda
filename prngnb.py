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

from datetime import datetime
from pathlib import Path

timport = datetime.now()
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import numba as nb
from tqdm import tqdm

from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

tinit = datetime.now()
NUM_PRNGS = 1 << (27 - 5)
PRNG_SEED = 20231229

# initialize PRNG states
# 1. allocate locally (on CPU)
states_cpu = np.empty((NUM_PRNGS, 2), dtype=np.uint64)

# 2. initialize locally (on CPU)
n = np.uint64(PRNG_SEED)

@nb.jit((nb.uint64[:,:], nb.int64, nb.uint64), nopython=True, cache=True, nogil=True, fastmath=True)
def init_state(states, index, seed):
    # index = np.int64(index)
    # seed = np.uint64(seed)
    z = seed + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint32(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint32(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint32(31))
    states[index,:] = z
    return

@nb.jit((nb.uint64, nb.uint32), nopython=True, cache=True, nogil=True, fastmath=True)
def rotl(x, k):
    '''Left rotate x by k bits.'''
    # x = np.uint64(x)
    # k = np.uint32(k)
    return (x << k) | (x >> np.uint32(64 - k))

@nb.jit((nb.uint64[:,:], nb.int64), nopython=True, cache=True, nogil=True, fastmath=True)
def gonext(states, index):
    # index = np.int64(index)
    s0, s1 = states[index]
    result = s0 + s1

    s1 ^= s0
    states[index,0] = np.uint64(rotl(s0, np.uint32(55))) ^ s1 ^ (s1 << np.uint32(14))
    states[index,1] = np.uint64(rotl(s1, np.uint32(36)))

    return result

@nb.jit((nb.uint64[:,:], nb.int64), nopython=True, cache=True, nogil=True, fastmath=True)
def jumpforward(states, index):
    # index = np.int64(index)
    jump = (np.uint64(0xbeac0467eba5facb), np.uint64(0xd86b048b86aa9922))
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

tsetup = datetime.now()
init_state(states_cpu, 0, PRNG_SEED)
for i in tqdm(range(1, NUM_PRNGS)):
    states_cpu[i] = states_cpu[i-1]
    jumpforward(states_cpu, i)

# 3. copy to GPU
states_gpu = gpuarray.to_gpu(states_cpu)

# 4. return gpuarray from 3

# CUDA kernel to draw from a uniform distribution
tkernel = datetime.now()
mod = SourceModule("""
__device__ uint64_t rotl(uint64_t x, uint32_t k) {
    return (x << k) | (x >> (64 - k));
}

#define FACTOR (double(1.0) / 9007199254740992)

__device__ float uint64_to_unit_float32(uint64_t x) {
    return float((x >> 11) * FACTOR);
}

__device__ uint64_t get_next(uint64_t *states, uint32_t index) {
    uint64_t s0 = states[index*2];
    uint64_t s1 = states[index*2+1];
    uint64_t result = s0 + s1;

    s1 ^= s0;
    s0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);
    s1 = rotl(s1, 36);

    states[index*2] = s0;
    states[index*2+1] = s1;

    return result;
}

__global__ void uniform(uint64_t *states, float *draws, uint32_t n_draws) {
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < n_draws; i += stride) {
        draws[i] = uint64_to_unit_float32(get_next(states, i));
    }
}

#define TWO_PI  (float(2.0 * 3.14159265))

__global__ void normal(uint64_t *states, float mean, float std, float *draws, uint32_t n_draws) {
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < n_draws; i += stride) {
        float u1 = uint64_to_unit_float32(get_next(states, i));
        float u2 = uint64_to_unit_float32(get_next(states, i));
        float z0 = sqrt(-2 * log(u1)) * cos(TWO_PI * u2);
        draws[i] = z0 * std + mean;
    }
}
""")
uniform = mod.get_function("uniform")
normal = mod.get_function("normal")
tdraw = datetime.now()
draws = gpuarray.empty((NUM_PRNGS,), dtype=np.float32)
NUM_THREAD = 128
NUM_BLOCKS = NUM_PRNGS // NUM_THREAD + 1
uniform(states_gpu, draws, np.uint32(NUM_PRNGS), block=(NUM_THREAD,1,1), grid=(NUM_BLOCKS,1,1))
pycuda.autoinit.context.synchronize()
tuniform = datetime.now()

with Path("uniform.csv").open("w") as file:
    for value in draws.get():
        file.write(f"{value}\n")

tdraw2 = datetime.now()
normal(states_gpu, np.float32(0), np.float32(1), draws, np.uint32(NUM_PRNGS), block=(NUM_THREAD,1,1), grid=(NUM_BLOCKS,1,1))
pycuda.autoinit.context.synchronize()
tfinish = datetime.now()

print(f"{NUM_PRNGS:_} draws from {NUM_PRNGS} PRNGs (support {NUM_PRNGS * 1 << 5} agents?)")
print(f"import: {tinit - timport}")
print(f"init: {tsetup - tinit}")
print(f"setup: {tkernel - tsetup}")
print(f"kernel: {tdraw - tkernel}")
print(f"uniform: {tuniform - tdraw}")
print(f"normal: {tfinish - tdraw2}")

with Path("normal.csv").open("w") as file:
    for value in draws.get():
        file.write(f"{value}\n")

pass
