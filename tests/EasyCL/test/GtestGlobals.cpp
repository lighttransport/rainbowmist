// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "GtestGlobals.h"

GtestGlobals *GtestGlobals::instance() {
    static GtestGlobals *thisInstance = new GtestGlobals();
    return thisInstance;
}


