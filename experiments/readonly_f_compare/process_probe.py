"""Observe process ownership without querying the CUDA/NVML driver during timing."""
import ctypes
from ctypes import wintypes as W
def processes():
 k=ctypes.WinDLL('kernel32',use_last_error=True)
 class Entry(ctypes.Structure):
  _fields_=[('dwSize',W.DWORD),('cntUsage',W.DWORD),('th32ProcessID',W.DWORD),('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',W.DWORD),('cntThreads',W.DWORD),('th32ParentProcessID',W.DWORD),('pcPriClassBase',W.LONG),('dwFlags',W.DWORD),('szExeFile',W.WCHAR*260)]
 k.CreateToolhelp32Snapshot.argtypes=[W.DWORD,W.DWORD];k.CreateToolhelp32Snapshot.restype=W.HANDLE
 k.Process32FirstW.argtypes=[W.HANDLE,ctypes.POINTER(Entry)];k.Process32NextW.argtypes=[W.HANDLE,ctypes.POINTER(Entry)]
 k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];k.OpenProcess.restype=W.HANDLE
 k.QueryFullProcessImageNameW.argtypes=[W.HANDLE,W.DWORD,W.LPWSTR,ctypes.POINTER(W.DWORD)]
 k.CloseHandle.argtypes=[W.HANDLE]
 snap=k.CreateToolhelp32Snapshot(2,0)
 if snap==W.HANDLE(-1).value:raise ctypes.WinError(ctypes.get_last_error())
 try:
  item=Entry();item.dwSize=ctypes.sizeof(item);ok=k.Process32FirstW(snap,ctypes.byref(item));result=[]
  while ok:
   path='';h=k.OpenProcess(0x1000,False,item.th32ProcessID)
   if h:
    try:
     buf=ctypes.create_unicode_buffer(32768);n=W.DWORD(len(buf))
     if k.QueryFullProcessImageNameW(h,0,buf,ctypes.byref(n)):path=buf.value
    finally:k.CloseHandle(h)
   result.append({'pid':item.th32ProcessID,'name':item.szExeFile,'path':path})
   ok=k.Process32NextW(snap,ctypes.byref(item))
  return result
 finally:k.CloseHandle(snap)
if __name__=='__main__':
 import json
 rows=processes();print(json.dumps({'process_count':len(rows),'workspace':[p for p in rows if 'v1_update' in p['path'].lower()],'python':[p for p in rows if p['name'].lower()=='python.exe']},indent=2))
