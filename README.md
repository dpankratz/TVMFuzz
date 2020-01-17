# CMPUT664Project

## Attack vectors to explore
1. What happens when FFR is all 0's
2. What happens when FFR is reset to always suppress exceptions
3. What new hardware structures have contention due to SVE
4. Can the vector accesses be denied by coherence attacks
5. Can link 2 be reproduced with SVE?


## Links to explore
1. http://www.numberworld.org/blogs/2018_6_16_avx_spectre/ 
  This article shows how clock throttling in the cpu caused by execution of an AVX instruction in a speculative path can be used to leak 1 bit of information. 
2. http://www.numberworld.org/blogs/2018_6_16_avx_spectre/ 
  Same as 1 but over the network. Of significance is illustrating the attack works on ARM.
