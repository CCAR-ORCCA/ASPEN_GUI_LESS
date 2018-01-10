
cmake_host_system_information(RESULT _host_name QUERY HOSTNAME)
message("-- _host_name variable is set to: " ${_host_name})


if ( ${_host_name} STREQUAL "fortuna")
	message("THIS IS FORTUNA")

endif()


set(ASPEN_INCLUDE_HEADER /usr/local/include/ASPEN)
set(ASPEN_INCLUDE_GNUPLOT /usr/local/include/ASPEN/external/Gnuplot)
set(ASPEN_INCLUDE_CGAL /usr/local/include/ASPEN/external/CGAL_interface)



if (APPLE)
	set(ASPEN_LIBRARY /usr/local/lib/libASPEN.dylib)
elseif(UNIX AND NOT APPLE)
	set(ASPEN_LIBRARY /usr/local/lib/libASPEN.so)
else()
	message(FATAL_ERROR "Unsupported platform")
endif()




message("-- Found ASPEN: " ${ASPEN_LIBRARY})
