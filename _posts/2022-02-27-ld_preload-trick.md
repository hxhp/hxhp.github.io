---
title: Pitfalls encountered in the LD_PRELOAD trick
date: 2022-02-27
tags: [LD\_PRELOAD, CUDA intercept, segmentation fault]
excerpt: |
  A few pitfalls encountered in applying the LD_PRELOAD trick to CUDA intercept.
---

## Motivation
Recently in the [alnair device plugin](https://github.com/CentaurusInfra/alnair/tree/main/intercept-lib) 
project, I was trying to use the [LD\_PRELOAD trick](https://osterlund.xyz/posts/2018-03-12-interceptiong-functions-c.html)
to intercept all the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) calls issued by 
CUDA applications. Here I document a few pitfalls encounted in applying this technique.

## Pitfall 1: could not intercept cuInit()?
While the LD\_PRELOAD trick in its original form works well for contrived examples as in the above 
[link](https://osterlund.xyz/posts/2018-03-12-interceptiong-functions-c.html), a simple attempt to intercept 
cuInit() in a trivial pytorch script failed:
```c
// foo.cpp
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>

CUresult cuInit (unsigned int Flags)
{
    CUresult (*lcuInit) (unsigned int) = (CUresult (*) (unsigned int)) dlsym(RTLD_NEXT, "cuInit");
    fprintf(stderr, "cuInit hooked\n");
    return lcuInit(Flags);
}
```
```bash
g++ -I/usr/local/cuda/include -fPIC -shared -o libfoo.so foo.cpp -ldl -lcuda

```
```python
# test.py
import torch
device = torch.cuda.current_device()
x = torch.randn(1024, 1024).to(device)
y = torch.randn(1024, 1024).to(device)
z = torch.matmul(x, y)
```
```bash
conda activate pytorch_gpu
LD_PRELOAD=./libfoo.so python test.py
```
There was nothing coming out of the above command! But I was expecting "cuInit hooked". This perplexed me for quite a 
while until I found these stackoverflow posts [1](https://stackoverflow.com/questions/49971717/hooking-into-cuda-driver-api-calls),
[2](https://stackoverflow.com/questions/37792037/ld-preload-doesnt-affect-dlopen-with-rtld-now). 

I came to realize that this technique might not be working in some cases. These "some" cases turn out to be quite
common in the CUDA runtime (a higher level library than the CUDA driver API), i.e. 
```c
dlsym(dlopen("...", RTLD_NOW), "...");
```
If a user application used dlsym(dlopen()) to call a library function, our technique would fail because
it can only overwrite those explicit calls.

How do we solve the problem? Fortunately the Nvidia example "/usr/local/cuda/samples/7\_CUDALibraries/cuHook/libcuhook.cpp" 
already gave a solution to this: to intercept dlsym().

Here is a quote from that example:
>We need to interpose dlsym since anyone using dlopen+dlsym to get the CUDA driver symbols will bypass
the hooking mechanism (this includes the CUDA runtime). Its tricky though, since if we replace the
real dlsym with ours, we can't dlsym() the real dlsym. To get around that, call the 'private'
libc interface called \_\_libc\_dlsym to get the real dlsym.

Following the Nvidia example, dlsym is intercepted in the same 
[way](https://github.com/CentaurusInfra/alnair/blob/ec63f740528dd60b0f703944a547669134f1608f/intercept-lib/src/cuda_interpose.cc#L144).


## Pitfall 2: segmentation fault
After intercepting dlsym(), I was happy to see that cuInit can now be interposed successfully in the above example. 
But very soon I found an even more weird problem: segmentation fault.

I thought it might be just my implementation was not good, but the library compiled from the Nvidia example also had such problems. 

Google search led to only similar problems but no solution. Then I added lots of log statements trying to pinpoint where the problem was. 

Finally I was able to isolate the issue with a simple example:
```c
// foo.cpp
#define _USE_GNU
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>

extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

typedef void* (*fnDlsym)(void*, const char*);

static void* real_dlsym(void *handle, const char* symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    fprintf(stderr, "real_dlsym %s\n", symbol);
    void* tmp =  (*internal_dlsym)(handle, symbol);
    if(tmp == NULL) fprintf(stderr, "cannot find symbol %s\n", symbol);
    return tmp;
}

void* dlsym(void *handle, const char *symbol)
{
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }

    if (strcmp(symbol, "cuInit") == 0) {
        fprintf(stderr, "attempt to dlsym cuInit\n"); 
        return (void*)(&cuInit);
    }

    return (real_dlsym(handle, symbol));
}

CUresult CUDAAPI cuInit (unsigned int Flags)
{
    fprintf(stderr, "cuInit hooked\n"); 
    static void* real_func = (void*)real_dlsym(RTLD_NEXT, "cuInit");
    CUresult result = ((CUresult CUDAAPI (*)(unsigned int))real_func)(Flags);
    return (result);
}

```
```bash
g++ -I/usr/local/cuda/include -fPIC -shared -o libfoo.so foo.cpp -ldl -lcuda
```
```c
// main.cu
#include <stdio.h>

int main()
{
  int a, *d_a;
  cudaMalloc(&d_a, sizeof(d_a[0]));
  cudaMemcpyAsync(d_a, &a, sizeof(a), cudaMemcpyHostToDevice);
  printf("Done\n");
}
```
```bash
nvcc main.cu -cudart shared
```
```bash
LD_PRELOAD=./libfoo.so ./a.out
```
>attempt to dlsym cuInit<br>
real_dlsym cuDeviceGet<br>
......<br>
real_dlsym cuGraphKernelNodeSetAttribute<br>
cuInit hooked<br>
real_dlsym cuInit<br>
cannot find symbol cuInit<br>
cuInit not found!<br>
Segmentation fault (core dumped)

Let me explain the output by listing the events happened in time order:
1. The dlsym() overwrite function was triggered to look up the "cuInit" symbol:
>attempt to dlsym cuInit 
2. The cuInit() overwrite function (instead of the real CUDA cuInit()) was returned to the caller.
3. Lots of other "cu*" symbols were looked up by dlsym() overwrite function and subsequently by real_dlsym():
>real_dlsym cuDeviceGet<br>
......<br>
real_dlsym cuGraphKernelNodeSetAttribute
4. The cuInit() overwrite function was triggered by the user application:
> cuInit hooked
5. The real_dlsym() was triggered by the cuInit() overwrite for the "cuInit" symbol: 
>real_dlsym cuInit<br>
cannot find symbol cuInit<br>
cuInit not found!<br>
Segmentation fault (core dumped)

We can clearly see that real_dlsym could not find "cuInit" and thus returned NULL at event 5. 
And then segmentation fault occurred when this NULL value was used to trigger the "cuInit" function call. 

One interesting obeservation is: the program was able to look up other "cu*" symbols at event 3, but failed to look up "cuInit" at event 5.
But these symbols are in the same shared library libcuda.so:
```bash
readelf -Ws /lib/x86_64-linux-gnu/libcuda.so
```
>...<br>
   611: 0000000000224b20    38 FUNC    GLOBAL DEFAULT   12 cuInit<br>
   ...<br>
   136: 0000000000224ac0    38 FUNC    GLOBAL DEFAULT   12 cuDeviceGet   

Why would this happen?

It's probably related to how the dynamic linker ld-linux.so works. For a symbol like "cuInit" to be looked up successfully, the corresponding 
info in the libcuda.so must be already loaded somewhere in the memory. We can see this was indeed the case at event 3. However, it looks that 
this was not the case at event 5. My guess is that ld-linux.so already loaded the info from libcuda.so before event 1, but later removed the 
info from the memory, perhaps for efficiency reasons. And after that, at event 5, "cuInit" could no longer be found.

Based on my guess, I tried to add a static variable "real_cuInit" to record the real "cuInit" at event 1. And it works!
```c
#define _USE_GNU
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>

extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

typedef void* (*fnDlsym)(void*, const char*);
static void* real_cuInit = NULL;

static void* real_dlsym(void *handle, const char* symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

void* dlsym(void *handle, const char *symbol)
{
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }

    if (strcmp(symbol, "cuInit") == 0) {
        if(real_cuInit == NULL) real_cuInit = real_dlsym(handle, symbol);
        return (void*)(&cuInit);
    }

    return (real_dlsym(handle, symbol));
}

CUresult CUDAAPI cuInit (unsigned int Flags)
{
    fprintf(stderr, "cuInit hooked\n");
    CUresult result = ((CUresult CUDAAPI (*)(unsigned int))real_cuInit)(Flags);
    return (result);
}
```
```bash
g++ -g -I/usr/local/cuda/include -fPIC -shared -o libfoo.so foo.cpp -ldl -lcuda
LD_PRELOAD=./libfoo.so ./a.out
```
>cuInit hooked<br>
Done

Yay, I solved it! And this solution was also applied in our 
[code](https://github.com/CentaurusInfra/alnair/blob/ec63f740528dd60b0f703944a547669134f1608f/intercept-lib/src/cuda_interpose.cc#L134).

## Pitfall 3: the program hangs...
Using the above libfoo.so, I encounted another problem when running this program:
```python
# test.py
import torch
device = torch.cuda.current_device()
x = torch.randn(1024, 1024).to(device)
y = torch.randn(1024, 1024).to(device)
z = torch.matmul(x, y)
```
The program simply hangs there, forever. Again let me add a couple of debugging statements:
```c
...
static void* real_dlsym(void *handle, const char* symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    void* tmp =  (*internal_dlsym)(handle, symbol);
    if(tmp == NULL) fprintf(stderr, "cannot find symbol %s\n", symbol);
    return tmp;
}
 
void* dlsym(void *handle, const char *symbol)
{
    if (strncmp(symbol, "cu", 2) != 0) {
        fprintf(stderr, "trace 1 %s\n", symbol);
        return (real_dlsym(handle, symbol));
    }
...
``` 
```bash
g++ -g -I/usr/local/cuda/include -fPIC -shared -o libfoo.so foo.cpp -ldl -lcuda
LD_PRELOAD=./libfoo.so python test.py >/tmp/log
less /tmp/log
```
>......
trace 1 omp_get_num_threads<br>
cannot find symbol omp_get_num_threads<br>
......<br>
trace 1 ompt_start_tool<br>
trace 1 ompt_start_tool<br>
trace 1 ompt_start_tool<br>
.......

We have two observations here:
1. The real_dlsym failed to look up "omp_get_num_threads"
2. There was an infinite loop to dlsym "ompt_start_tool"

Using a similar method as the fix in the segmentation fault, I added a variable "real_omp_get_num_threads"
and manually opened libgomp.so.1 to find the symbol "omp_get_num_threads".
```c
#define _USE_GNU
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>

extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

typedef void* (*fnDlsym)(void*, const char*);
static void* real_cuInit = NULL;
static void* real_omp_get_num_threads = NULL;

static void* real_dlsym(void *handle, const char* symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    return  (*internal_dlsym)(handle, symbol);
}

void* dlsym(void *handle, const char *symbol)
{
    if (strcmp(symbol, "omp_get_num_threads") == 0) {
        if(real_omp_get_num_threads == NULL)
            real_omp_get_num_threads = (void*)__libc_dlsym(__libc_dlopen_mode("libgomp.so.1", RTLD_LAZY), "omp_get_num_threads");
        return real_omp_get_num_threads;
    }

    if (strncmp(symbol, "cu", 2) != 0) {
        return real_dlsym(handle, symbol);
    }

    if (strcmp(symbol, "cuInit") == 0) {
        if(real_cuInit == NULL) real_cuInit = real_dlsym(handle, symbol);
        return (void*)(&cuInit);
    }

    return (real_dlsym(handle, symbol));
}

CUresult CUDAAPI cuInit (unsigned int Flags)
{
    fprintf(stderr, "cuInit hooked\n");
    CUresult result = ((CUresult CUDAAPI (*)(unsigned int))real_cuInit)(Flags);
    return (result);
}
```
```bash
g++ -I/usr/local/cuda/include -fPIC -shared -o libfoo.so foo.cpp -ldl -lcuda
LD_PRELOAD=./libfoo.so python test.py
```
>cuInit hooked<br>
cuInit hooked

## Summary
The LD\_PRELOAD trick is not as easy as it seems. Hard to debug pitfalls were encounted in my attempt to interpose the CUDA driver 
API. There might be more to come and we need to put the [code](https://github.com/CentaurusInfra/alnair/tree/main/intercept-lib) to
comprehensive testing.
