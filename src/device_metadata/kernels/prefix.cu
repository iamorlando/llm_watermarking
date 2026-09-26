typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long long wm_u64;
#define WM_DEVICE
#define WM_PRIVATE
#define WM_CONSTANT __device__ __constant__
#define WM_FN __device__ __forceinline__
#define WM_LOG logf
#define WM_ATOMIC_OR(p,v) atomicOr(p,v)
#define WM_ATOMIC_EXCHANGE(p,v) atomicExch(p,v)
