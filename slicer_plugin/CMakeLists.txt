cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(NeuroStrip)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/dyollb/neurostrip")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Bryn Lloyd (Slicer Community)")
set(EXTENSION_DESCRIPTION "CNN based skull stripping (brain masking) from MRI using the NeuroStrip library")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/dyollb/neurostrip/main/slicer_plugin/Resources/Icons/NeuroStrip.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/dyollb/neurostrip/main/docs/slicer_screenshot.png")
set(EXTENSION_DEPENDS "neurostrip; onnxruntime")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(NeuroStrip)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
