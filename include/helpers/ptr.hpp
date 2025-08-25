#pragma once

template<typename T>
struct Ptr{
  T t;
  Ptr(T in_t){
    t = in_t;
  }
  operator T*(){
    return &t;
  }
  operator const T*(){
    return &t;
  }
};
