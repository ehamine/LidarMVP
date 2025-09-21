add_library(lidar_manager_core STATIC
    LidarAcquisition.cpp
    OusterDriverEnhanced.cpp
    PointTypes.cpp
)

target_include_directories(lidar_manager_core PUBLIC ../include)
target_link_libraries(lidar_manager_core PUBLIC fmt::fmt OusterSDK::OusterSDK)

add_executable(lidar_manager
    main.cpp
)

add_executable(scan_demo
    ../test_scan_demo.cpp
)

add_executable(offline_test
    ../test_offline_acquisition.cpp
)

target_link_libraries(lidar_manager PRIVATE lidar_manager_core)
target_link_libraries(scan_demo PRIVATE lidar_manager_core)
target_link_libraries(offline_test PRIVATE lidar_manager_core)

