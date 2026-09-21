#pragma once
// Host transport only. Exact stable device packet bytes, no file output.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <climits>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <openssl/evp.h>
#include "memory_common.h"

namespace sgpipe {
inline void require(bool b,const char *s){if(!b)throw std::runtime_error(s);}
static_assert(sizeof(fast_mem_access_t)==616,"stable packet size differs");
static_assert(offsetof(fast_mem_access_t,mem_addrs1)==88,"stable address offset differs");
static_assert(offsetof(fast_mem_access_t,curr_clk)==600,"stable clock offset differs");
static_assert(offsetof(fast_mem_access_t,predicate_mask)==612,"stable mask offset differs");
enum Type:uint32_t {HELLO=1,STATIC=2,BEGIN=3,RECORD=4,END=5,CLOSED=6};

// Scope only the writing thread's SIGPIPE. A broken consumer throws instead of
// terminating the target before its sampler error receipt can be written.
struct NoSigpipe {
  sigset_t one,old;bool previously_pending=false;
  NoSigpipe(){sigemptyset(&one);sigaddset(&one,SIGPIPE);sigset_t p;sigpending(&p);previously_pending=sigismember(&p,SIGPIPE);require(pthread_sigmask(SIG_BLOCK,&one,&old)==0,"SIGPIPE mask");}
  ~NoSigpipe(){sigset_t p;if(!previously_pending&&sigpending(&p)==0&&sigismember(&p,SIGPIPE)){int result;sigwait(&one,&result);}pthread_sigmask(SIG_SETMASK,&old,nullptr);}
};
struct Writer {
  int fd=-1;uint64_t bytes=0,limit=0;EVP_MD_CTX* hash=nullptr;bool closed=false;
  void open(int inherited_fd,uint64_t cap){
    require(fd==-1&&cap>=1024&&cap<=(8ull<<30),"pipe/budget state");
    struct stat st{};require(fstat(inherited_fd,&st)==0&&S_ISFIFO(st.st_mode),"sample output must be a pipe, never a regular file");
    int flags=fcntl(inherited_fd,F_GETFL);require(flags>=0&&(flags&O_ACCMODE)==O_WRONLY,"sample pipe requires write endpoint");
    // The controller creates a dedicated pipe; this owned endpoint is set
    // nonblocking so even a stalled consumer obeys the ten-second poll bound.
    require(fcntl(inherited_fd,F_SETFL,flags|O_NONBLOCK)==0,"nonblocking sample pipe");
    fd=inherited_fd;limit=cap;hash=EVP_MD_CTX_new();require(hash&&EVP_DigestInit_ex(hash,EVP_sha256(),nullptr)==1,"pipe SHA init");
    uint16_t little=1;require(*reinterpret_cast<unsigned char*>(&little)==1,"packet wire requires little endian host");
    write("SGPKT001",8);
  }
  void write(const void* data,size_t n){
    require(!closed&&fd>=0&&n<=limit-bytes,"sample pipe byte quota");
    NoSigpipe block;const char *p=static_cast<const char*>(data);size_t offset=0;
    // Bound the whole write call; repeated EINTR cannot reset this deadline.
    timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);uint64_t deadline=uint64_t(t.tv_sec)*1000+t.tv_nsec/1000000+10000;
    while(offset<n){
      clock_gettime(CLOCK_MONOTONIC,&t);uint64_t current=uint64_t(t.tv_sec)*1000+t.tv_nsec/1000000;require(current<deadline,"sample consumer stalled for ten seconds");
      ssize_t k=::write(fd,p+offset,n-offset);
      if(k>0){require(EVP_DigestUpdate(hash,p+offset,size_t(k))==1,"pipe SHA update");bytes+=uint64_t(k);offset+=size_t(k);continue;}
      if(k<0&&errno==EINTR)continue;
      if(k<0&&(errno==EAGAIN||errno==EWOULDBLOCK)){pollfd wait{fd,POLLOUT,0};int r=poll(&wait,1,int(deadline-current));if(r<0&&errno==EINTR)continue;require(r>0&&!(wait.revents&(POLLERR|POLLHUP|POLLNVAL)),"sample consumer closed/stalled");continue;}
      throw std::runtime_error("sample pipe write failed/broken consumer");
    }
  }
  void frame(Type kind,const void* data,size_t n){require(n>0&&n<=(1u<<20)&&bytes<=limit&&n+8<=limit-bytes,"bounded complete frame");uint32_t h[2]={uint32_t(kind),uint32_t(n)};write(h,sizeof(h));write(data,n);}
  void json(Type kind,const std::string&s){frame(kind,s.data(),s.size());}
  void packet(uint64_t received,uint64_t selected,const fast_mem_access_t &p){
    // No struct padding exists under the validated 616-byte ABI. capture_seq
    // stays zero; the two host ordinals have explicitly different meanings.
    require(p.capture_seq==0,"device packet was renumbered");unsigned char payload[632];
    std::memcpy(payload,&received,8);std::memcpy(payload+8,&selected,8);std::memcpy(payload+16,&p,616);frame(RECORD,payload,sizeof(payload));
  }
  std::string digest()const{
    EVP_MD_CTX*c=EVP_MD_CTX_new();require(c&&EVP_MD_CTX_copy_ex(c,hash)==1,"pipe SHA copy");unsigned char d[32];unsigned n=0;bool ok=EVP_DigestFinal_ex(c,d,&n)==1;EVP_MD_CTX_free(c);require(ok&&n==32,"pipe SHA final");
    static const char h[]="0123456789abcdef";std::string s;for(unsigned char b:d){s+=h[b>>4];s+=h[b&15];}return s;
  }
  void close(){if(fd>=0){require(::close(fd)==0,"sample pipe close");fd=-1;}closed=true;if(hash){EVP_MD_CTX_free(hash);hash=nullptr;}}
};
} // namespace sgpipe
