#pragma once

//#if defined(_WIN32)  // HACK(LTE): Disable export/imoort since EasyCL is directly linked with lucia
#if 0
# if defined(EasyCL_EXPORTS)
#  define EasyCL_EXPORT __declspec(dllexport)
# else
#  define EasyCL_EXPORT __declspec(dllimport)
# endif 
#else 
# define EasyCL_EXPORT
#endif
