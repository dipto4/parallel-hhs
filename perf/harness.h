#pragma once
#include<iostream>
#include<string>


#define TEST_EQUAL(test_name, actual, expected)                       \
    do {                                                              \
        if ((actual) != (expected)) {                                 \
            std::cerr << "[FAIL] " << test_name << "\n"              \
                      << "  " #actual " = " << (actual) << "\n"       \
                      << "  " #expected " = " << (expected) << "\n"; \
        } else {                                                      \
            std::cout << "[PASS] " << test_name << "\n";             \
        }                                                             \
    } while (0)




#define TEST_ASSERT(test_name, condition)                            \
    do {                                                             \
        if (!(condition)) {                                          \
            std::cerr << "[FAIL] " << test_name << "\n"             \
                      << "  Condition failed: " #condition "\n";    \
        } else {                                                     \
            std::cout << "[PASS] " << test_name << "\n";            \
        }                                                            \
    } while (0)


