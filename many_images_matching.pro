QT += core
QT -= gui

TARGET = many_images_matching
CONFIG += console
CONFIG -= app_bundle
CONFIG += link_pkgconfig
PKGCONFIG += eigen3

TEMPLATE = app

LIBS += `pkg-config \
    opencv \
    --cflags \
    --libs`

INCLUDEPATH += /usr/local/include/pcl-1.8

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

