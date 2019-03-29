QT += core
QT -= gui

TARGET = many_images_matching
CONFIG += console
CONFIG -= app_bundle
CONFIG += link_pkgconfig
PKGCONFIG += eigen3
PKGCONFIG += pcl_io-1.8
QMAKE_CXXFLAGS += -std=c++11

TEMPLATE = app

INCLUDEPATH += /usr/local/include/pcl-1.8
INCLUDEPATH += /usr/include/vtk-6.2

LIBS += -L/usr/lib/ -lboost_system
LIBS += `pkg-config \
    opencv \
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

