find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})


add_executable(test_nToken helloTest.cpp) # test name

target_link_libraries(test_nToken ${GTEST_BOTH_LIBRARIES} pthread) #gtest need pthread


gtest_discover_tests(test_nToken) #add to test list