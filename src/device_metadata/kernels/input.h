// Device inputs never become host seeds. p = seed_len,input_len,size,mode,param,auxiliary.
WM_FN void wm_input(WM_DEVICE const uchar* seed, WM_DEVICE const uint* input,
                    WM_DEVICE uchar* output, uint len, uint input_len, uint size,
                    uint mode, uint param, uint repeat, uint tid) {
  if (tid>=size) return;
  if (mode==6 || mode==7) {
    WM_DEVICE uint* values=(WM_DEVICE uint*)output;
    if (mode==6) {
      values[tid]=tid<input_len ? input[tid] : 0xffffffffu;
      values[(wm_u64)size+tid]=tid;
    } else {
      uint other=tid^param;
      uint a=input[tid],b=input[other];
      uint ai=input[(wm_u64)size+tid],bi=input[(wm_u64)size+other];
      bool less=a<b || (a==b && ai<bi);
      bool take_min=((tid & repeat)==0)==((tid & param)==0);
      bool take_self=less==take_min;
      values[tid]=take_self ? a : b;
      values[(wm_u64)size+tid]=take_self ? ai : bi;
    }
    return;
  }
  if (mode==2) {
    uint length=min(input[0],input_len-2), prompt=min(input[1],length);
    if (tid<len) output[tid]=seed[tid];
    else if (tid<size-1) {
      uint offset=tid-len;
      uint token=length>=param ? input[2+length-param+offset/4] : 0;
      output[tid]=(uchar)(token>>(8*(offset%4)));
    } else {
      bool active=length>=param;
      if (active && repeat) {
        uint start=max(max(prompt,param),length>1024 ? length-1024 : 0);
        for (uint end=start;end<length;++end) {
          bool equal=true;
          for (uint j=0;j<param;++j) {
            if (input[2+end-param+j]!=input[2+length-param+j]) { equal=false; break; }
          }
          if (equal) { active=false; break; }
        }
      }
      output[tid]=(uchar)active;
    }
    return;
  }
  if (mode==3) {
    if (tid<len) output[tid]=seed[tid];
    else output[tid]=(uchar)(((wm_u64)(input[0]%param))>>(8*(tid-len)));
    return;
  }
  uint h[8];
  if (mode==4 || mode==5) {
    wm_u64 cursor=0,value,bound=param;
    wm_u64 threshold=mode==5 ? ((wm_u64)0-bound)%bound : 0;
    do {
      wm_digest(seed,len,3,cursor/4,h);
      uint j=(uint)(cursor%4)*2;
      value=(wm_u64)wm_swap(h[j]) | ((wm_u64)wm_swap(h[j+1])<<32);
      ++cursor;
    } while (value<threshold);
    if (mode==5) ((WM_DEVICE uint*)output)[0]=(uint)(value%bound);
    else ((WM_DEVICE float*)output)[0]=min(((float)(value>>12)+0.5f)*0x1p-52f,0x1.fffffep-1f);
    return;
  }
  wm_digest(seed,len,mode,input[tid],h);
  if (mode==0) {
    for (uint b=0;b<param;++b) output[(wm_u64)b*size+tid]=(uchar)(h[b/4]>>(8*(3-b%4)));
  } else {
    wm_u64 word=(wm_u64)wm_swap(h[0]) | ((wm_u64)wm_swap(h[1])<<32);
    wm_u64 n=word>>12;
    float cost;
    if (n>=((wm_u64)1<<51)) {
      float tail=((float)((((wm_u64)1<<52)-1)-n)+0.5f)*0x1p-52f;
      float x=tail/(2.0f-tail),x2=x*x,term=x,sum=x;
      for (uint k=1;k<12;++k) { term*=x2; sum+=term/(float)(2*k+1); }
      cost=2.0f*sum;
    } else cost=-WM_LOG(((float)n+0.5f)*0x1p-52f);
    ((WM_DEVICE float*)output)[tid]=-WM_LOG(cost);
  }
}
