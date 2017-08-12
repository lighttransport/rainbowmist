#include "gtest/gtest.h"
#include "util/easycl_stringhelper.h"

inline ::testing::AssertionResult AssertFloatsNear(const char *expr_one, const char *expr_two,
    float one, float two) {
    float diff = one - two;
    float absdiff = diff > 0 ? diff : - diff;
    float absone = one > 0 ? one : -one;
    if(absdiff <= absone * 0.0001f) {
        return ::testing::AssertionSuccess();
    }
    if(absdiff < 0.000000000000001f) {
        return ::testing::AssertionSuccess();
    }
    std::string onestr = easycl::toString(one);
    std::string twostr = easycl::toString(two);
    return ::testing::AssertionFailure()
        << expr_one << " and " << expr_two << " differ: " << onestr << " vs " << twostr;
} 

#define EXPECT_FLOAT_NEAR(one, two) EXPECT_PRED_FORMAT2(AssertFloatsNear, one, two)
#define ASSERT_FLOAT_NEAR(one, two) ASSERT_PRED_FORMAT2(AssertFloatsNear, one, two)


