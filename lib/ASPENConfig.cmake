
if (EXISTS /home/bebe0705/.am_fortuna)
	set(IS_FORTUNA ON)
	message("-- This is Fortuna")

else()
	set(IS_FORTUNA OFF)
endif()


if(${IS_FORTUNA})

	set(ASPEN_INCLUDE_HEADER /home/bebe0705/libs/local/include/ASPEN)
	set(ASPEN_INCLUDE_GNUPLOT /home/bebe0705/libs/local/include/ASPEN/external/Gnuplot)
	set(ASPEN_INCLUDE_CGAL /home/bebe0705/libs/local/include/ASPEN/external/CGAL_interface)
	set(ASPEN_LIBRARY /home/bebe0705/libs/local/lib/libASPEN.so)

else()
	
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
endif()



message("-- Found ASPEN: " ${ASPEN_LIBRARY})
