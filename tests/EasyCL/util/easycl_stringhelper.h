// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>

#include "../EasyCL_export.h"

namespace easycl {
    EasyCL_EXPORT class IHasToString {
    public:
        virtual ~IHasToString() {}
        virtual std::string toString() = 0;
    };

    //std::string toString(IHasToString *val); // { // not terribly efficient, but works...
    //   std::ostringstream myostringstream;
    //   myostringstream << val->toString();
    //   return myostringstream.str();
    //}

    template<typename T>
    std::string toString(T val) { // not terribly efficient, but works...
       std::ostringstream myostringstream;
       myostringstream << val;
       return myostringstream.str();
    }

    EasyCL_EXPORT std::vector<std::string> split(const std::string &str, const std::string &separator = " ");
    EasyCL_EXPORT std::string trim(const std::string &target);

    inline float atof(std::string stringvalue) {
       return (float)std::atof(stringvalue.c_str());
    }
    inline int atoi(std::string stringvalue) {
       return std::atoi(stringvalue.c_str());
    }

    // returns empty string if off the end of the number of available tokens
    inline std::string getToken(std::string targetstring, int tokenIndexFromZero, std::string separator = " ") {
       std::vector<std::string> splitstring = split(targetstring, separator);
       if(tokenIndexFromZero < (int)splitstring.size()) {
          return splitstring[tokenIndexFromZero];
       } else {
          return "";
       }
    }

    EasyCL_EXPORT std::string replace(std::string targetString, std::string oldValue, std::string newValue);
    EasyCL_EXPORT std::string replaceGlobal(std::string targetString, std::string oldValue, std::string newValue);

    EasyCL_EXPORT std::string toLower(std::string in);

    EasyCL_EXPORT void strcpy_safe(char *destination, char const*source, int maxLength);
}
