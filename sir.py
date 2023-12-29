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

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import polars as pl
from tqdm import tqdm

from pycuda.compiler import SourceModule
# from pycuda import gpuarray as GPUArray
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel

POP_SIZE = 1 << 28  # 1_048_576
INIT_COUNT = 10
INF_MEAN = 5
INF_STD = 1

# allocate space for POP_SIZE uint8s on the GPU
sus_gpu = gpuarray.ones(POP_SIZE, dtype=np.uint8)

# allocate space for POP_SIZE uint8s on the GPU
inf_gpu = gpuarray.zeros(POP_SIZE, dtype=np.uint8)

# draw INIT_COUNT random indices from the population (POP_SIZE) without replacement
init_cpu = np.random.choice(POP_SIZE, size=INIT_COUNT, replace=False)
init_gpu = gpuarray.to_gpu(init_cpu)

# draw INIT_COUNT values from a normal distribution with mean INF_MEAN and standard deviation INF_STD
itimer_cpu = np.round(np.random.normal(INF_MEAN, INF_STD, INIT_COUNT)).astype(np.uint8)
itimer_gpu = gpuarray.to_gpu(itimer_cpu)

# run initialization kernel
mod = SourceModule("""
__global__ void init(uint8_t *sus, uint8_t *inf, uint32_t *init, uint8_t *itimer)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  sus[init[i]] = 0;
  inf[init[i]] = itimer[i];
}
""", cache_dir=Path(r"c:\src\piecoodah\.cache"))
init = mod.get_function("init")
init(sus_gpu, inf_gpu, init_gpu, itimer_gpu, block=(INIT_COUNT,1,1), grid=(1,1))

# def do_debug():
#     sus_cpu = np.empty(sus_gpu.shape, dtype=sus_gpu.dtype)
#     inf_cpu = np.empty(inf_gpu.shape, dtype=inf_gpu.dtype)
#     sus_gpu.get(sus_cpu)
#     inf_gpu.get(inf_cpu)
#     print(sus_cpu[init_cpu])
#     print(inf_cpu[init_cpu])
#     ... # do something with sus_cpu and inf_cpu
#     pass

# do_debug()

# infection update kernel
mod = SourceModule("""
__global__ void update(uint8_t *inf)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (inf[i] > 0) {
    inf[i] -= 1;
  }
}
""", cache_dir=Path(r"c:\src\piecoodah\.cache"))

update = mod.get_function("update")
BLK_SIZE = 1024
GRD_SIZE = int(np.ceil(POP_SIZE / BLK_SIZE))

# update(inf_gpu, block=(BLK_SIZE,1,1), grid=(GRD_SIZE,1))
# do_debug()

sumc = ReductionKernel(np.uint32, neutral="0",
        reduce_expr="a+b", map_expr="(inf[i] ? 1 : 0)",
        arguments="unsigned char *inf")

# contagion = np.uint32(sumc(inf_gpu).get())
# print(f"Initial contagion: {contagion}")

R_ZERO = 2.5
BETA = R_ZERO / INF_MEAN

# run transmission kernel
# mod = SourceModule("""
# __global__ void transmit(uint8_t *sus, uint8_t *inf, float force, float inf_mean, float inf_std)
# {
#   const int i = blockIdx.x*blockDim.x+threadIdx.x;
#   // Every 16th thread will draw random numbers for itself and the next 15 threads
#   // We also draw from the normal for any threads that need it.
#   // Code comes from Numba's implementation.
#   float draw = curand_normal(&state);
#   if (draw < force) {
    # sus[i] = 0;
    # inf[i] = round(curand_normal(&state) * inf_std + inf_mean);
#   }
# }
# """)
# transmit = mod.get_function("transmit")

sumr = ReductionKernel(np.uint32, neutral="0",
        reduce_expr="a+b", map_expr="(((sus[i] == 0) && (inf[i] == 0)) ? 1 : 0)",
        arguments="unsigned char *sus, unsigned char *inf")

TIMESTEPS = 180

results = np.zeros((TIMESTEPS+1, 4), dtype=np.uint32)

def record(timestep, susg, infg, results):
    results[timestep, 0] = timestep
    results[timestep, 1] = sumc(susg).get()
    results[timestep, 2] = sumc(infg).get()
    results[timestep, 3] = sumr(susg, infg).get()

record(0, sus_gpu, inf_gpu, results)

start = datetime.now()
for i in tqdm(range(TIMESTEPS)):
    update(inf_gpu, block=(BLK_SIZE,1,1), grid=(GRD_SIZE,1))
    contagion = np.uint32(sumc(inf_gpu).get())
    force = BETA * contagion / POP_SIZE
    # transmit(sus_gpu, inf_gpu, force, INF_MEAN, INF_STD, block=(BLK_SIZE,1,1), grid=(GRD_SIZE,1))
    record(i+1, sus_gpu, inf_gpu, results)
finish = datetime.now()

print(f"Time elapsed: {finish - start}")

df = pl.DataFrame(data=results, schema=["timestep", "susceptible", "infected", "recovered"])
df.write_csv("sir.csv")
print(df.head(8))
print(f"Results for {POP_SIZE:_} agents for {TIMESTEPS} timesteps written to 'sir.csv'.")

pass
