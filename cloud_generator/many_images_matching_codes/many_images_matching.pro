QT += core
QT -= gui

TARGET = many_images_matching
CONFIG += console
CONFIG -= app_bundle
CONFIG += link_pkgconfig
CONFIG += c++11 console
PKGCONFIG += eigen3
PKGCONFIG += pcl_io-1.9
#QMAKE_CXXFLAGS += -std=c++11

TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv4
INCLUDEPATH += /usr/local/include/pcl-1.9
INCLUDEPATH += /usr/local/include/vtk-8.1

LIBS += -L/usr/lib/ -lboost_system
LIBS += -L/usr/local/lib -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_xfeatures2d -lopencv_imgcodecs -lopencv_calib3d
LIBS += `pkg-config \
    --cflags \
    --libs`

SOURCES += main.cpp \
    view.cpp

HEADERS += \
    stats.h \
    utils.h \
    many_images_matching.h \
    view.h \
    sfm.h \
    cloudpoint.h \
    cloud_to_pcldatastruct.h

